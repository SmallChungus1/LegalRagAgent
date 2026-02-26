import functools
import logging
import os
import time
import numpy as np
import pandas as pd
import threading
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Global lock to prevent Apple Silicon MPS crashes when multiple threads hit the GPU
_gpu_lock = threading.Lock()

def gpu_locked(func):
    """Decorator to ensure only one thread accesses the GPU at a time on MPS."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import torch
        # Only lock if we are on Apple Silicon MPS (known to crash with concurrent access)
        if torch.backends.mps.is_available():
            with _gpu_lock:
                return func(*args, **kwargs)
        return func(*args, **kwargs)
    return wrapper

CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "legal_passages"
QA_MEMORY_COLLECTION = "qa_memory"

# Embedding model — configurable via env var. Default: gte-large-en-v1.5 (1024d, 8192 tokens)
# Other options: "all-MiniLM-L6-v2" (384d, fast), "freelawproject/modernbert-embed-base_finetune_8192" (768d, legal-specific)
DEFAULT_EMBEDDING_MODEL = "Alibaba-NLP/gte-large-en-v1.5"

# Source-aware retrieval — when True, retrieves from study (mbe/wex) and caselaw pools
# separately and interleaves results. When False, retrieves from the full corpus and lets
# the cross-encoder pick the best passages regardless of source.
SOURCE_DIVERSE_RETRIEVAL = os.getenv("SOURCE_DIVERSE_RETRIEVAL", "0") == "1"


@functools.lru_cache(maxsize=1)
def get_embeddings():
    model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[rag_utils] Loading embedding model: {model_name} on {device}")
    
    # Some models (gte-large, modernbert) use custom architectures that need trust_remote_code
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "trust_remote_code": True,
            "device": device
        },
    )

@functools.lru_cache(maxsize=1)
def get_cross_encoder():
    """Returns a cached cross-encoder model for reranking.

    ms-marco-MiniLM-L-6-v2 scores (query, document) pairs with full
    cross-attention, catching semantic nuances that bi-encoder embeddings miss.
    """
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[rag_utils] Loading cross-encoder on {device}")
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)


def rerank_with_cross_encoder(
    query: str,
    docs: List[Document],
    top_k: int = 5,
) -> List[Document]:
    """Rerank documents using a cross-encoder model."""
    if not docs or len(docs) <= 1:
        return docs[:top_k]

    cross_encoder = get_cross_encoder()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)

    scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:top_k]]


_vectorstore_instances: Dict[str, Chroma] = {}

def get_vectorstore(collection_name: str = COLLECTION_NAME) -> Chroma:
    """Returns a Chroma vector store singleton for the given collection."""
    if collection_name not in _vectorstore_instances:
        embeddings = get_embeddings()
        _vectorstore_instances[collection_name] = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_DIR,
        )
    return _vectorstore_instances[collection_name]

def load_passages_to_chroma(passages_csv_path: str, max_passages: int = 0,
                            collection_name: str = COLLECTION_NAME):
    """Loads passages from a CSV into ChromaDB if not already loaded.

    Args:
        passages_csv_path: Path to the passages CSV (columns: idx, text, source, ...).
        max_passages: Max passages to load (0 = all). Useful for quick testing.
        collection_name: ChromaDB collection to load into (default: legal_passages).
    """
    print(f"Loading passages from {passages_csv_path}...")
    df = pd.read_csv(passages_csv_path)

    if max_passages > 0:
        df = df.head(max_passages)

    documents = []
    for _, row in df.iterrows():
        if 'idx' not in row or 'text' not in row or pd.isna(row['text']):
            continue

        metadata = {
            "faiss_id": str(row.get('faiss_id', '')),
            "idx": str(row['idx']),
            "source": str(row.get('source', ''))
        }

        doc = Document(page_content=str(row['text']), metadata=metadata)
        documents.append(doc)

    print(f"Prepared {len(documents)} documents. Initializing vectorstore...")
    vectorstore = get_vectorstore(collection_name)

    existing_count = vectorstore._collection.count()
    if existing_count >= len(documents):
        print(f"Vectorstore already contains {existing_count} documents (>= {len(documents)}). Skipping.")
        return vectorstore

    if existing_count > 0:
        print(f"Vectorstore has {existing_count} docs but need {len(documents)}. Clearing and reloading...")
        ids = vectorstore._collection.get()["ids"]
        if ids:
            for i in range(0, len(ids), 5000):
                vectorstore._collection.delete(ids=ids[i:i+5000])
        print("Cleared existing collection.")

    batch_size = 500
    total = len(documents)
    for i in range(0, total, batch_size):
        batch = documents[i:i+batch_size]
        vectorstore.add_documents(batch)
        done = min(i + batch_size, total)
        if done % 5000 == 0 or done == total:
            print(f"  Progress: {done}/{total} ({done/total*100:.1f}%)")

    print(f"Finished loading {total} documents into ChromaDB.")
    return vectorstore

def get_retriever(k: int = 5, vectorstore: Chroma = None):
    """Returns a retriever interface for the vector store."""
    vs = vectorstore or get_vectorstore()
    return vs.as_retriever(search_kwargs={"k": k})


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _retrieve_unified(query: str, k: int, vectorstore: Chroma) -> List[Document]:
    """Unified retrieval: over-retrieve from full corpus, then cross-encoder rerank.

    No source splitting — the cross-encoder picks the best passages regardless
    of whether they're MBE, wex, or caselaw.
    """
    fetch_k = k * 4
    retriever = get_retriever(k=fetch_k, vectorstore=vectorstore)
    candidates = retriever.invoke(query)
    return rerank_with_cross_encoder(query, candidates, top_k=k)


def _retrieve_source_diverse(query: str, k: int, vectorstore: Chroma) -> List[Document]:
    """Source-diverse retrieval: separate pools for study material vs caselaw.

    Over-retrieves from each pool, reranks within each, then interleaves.
    Preserves source diversity but may miss the best passages if they're
    concentrated in one pool.
    """
    fetch_k = k * 4

    try:
        study_results = vectorstore.similarity_search_with_relevance_scores(
            query, k=fetch_k, filter={"source": {"$in": ["mbe", "wex"]}}
        )
    except Exception as e:
        logger.warning("retrieval pool error (study): %s", e)
        study_results = []

    try:
        case_results = vectorstore.similarity_search_with_relevance_scores(
            query, k=fetch_k, filter={"source": "caselaw"}
        )
    except Exception as e:
        logger.warning("retrieval pool error (caselaw): %s", e)
        case_results = []

    study_candidates = _dedup_docs(study_results)
    case_candidates = _dedup_docs(case_results)

    study_k = (k + 1) // 2
    case_k = k - study_k

    reranked_study = rerank_with_cross_encoder(query, study_candidates, top_k=study_k)
    reranked_case = rerank_with_cross_encoder(query, case_candidates, top_k=case_k)

    result = list(reranked_study)
    seen = {doc.metadata.get("idx", "") for doc in result}
    for doc in reranked_case:
        if len(result) >= k:
            break
        idx = doc.metadata.get("idx", "")
        if idx not in seen:
            seen.add(idx)
            result.append(doc)

    # Backfill if either pool was too small
    if len(result) < k:
        all_candidates = study_candidates + case_candidates
        all_reranked = rerank_with_cross_encoder(query, all_candidates, top_k=k * 2)
        for doc in all_reranked:
            if len(result) >= k:
                break
            idx = doc.metadata.get("idx", "")
            if idx not in seen:
                seen.add(idx)
                result.append(doc)

    return result


def _dedup_docs(scored_results) -> List[Document]:
    """Deduplicate (doc, score) results by idx."""
    candidates = []
    seen = set()
    for doc, _ in scored_results:
        idx = doc.metadata.get("idx", "")
        if idx not in seen:
            seen.add(idx)
            candidates.append(doc)
    return candidates


@gpu_locked
def retrieve_documents(query: str, k: int = 5, vectorstore: Chroma = None,
                       exclude_ids: set = None) -> List[Document]:
    """Two-stage retrieval: bi-encoder over-retrieve + cross-encoder rerank.

    When SOURCE_DIVERSE_RETRIEVAL is True, splits retrieval into study/caselaw pools.
    When False (default), retrieves from the full corpus and lets the cross-encoder decide.

    Args:
        exclude_ids: Set of document idx strings to exclude (for cross-step dedup).
    """
    vectorstore = vectorstore or get_vectorstore()
    total_docs = vectorstore._collection.count()

    # Small corpus: always use simple retrieval
    if total_docs < 5000:
        fetch_k = k * 4
        retriever = get_retriever(k=fetch_k, vectorstore=vectorstore)
        candidates = retriever.invoke(query)
        if exclude_ids:
            candidates = [d for d in candidates if d.metadata.get("idx", "") not in exclude_ids]
        return rerank_with_cross_encoder(query, candidates, top_k=k)

    if SOURCE_DIVERSE_RETRIEVAL:
        results = _retrieve_source_diverse(query, k, vectorstore)
    else:
        results = _retrieve_unified(query, k, vectorstore)

    if exclude_ids:
        results = [d for d in results if d.metadata.get("idx", "") not in exclude_ids]
    return results


@gpu_locked
def retrieve_documents_multi_query(queries: List[str], k: int = 5,
                                   vectorstore: Chroma = None,
                                   exclude_ids: set = None) -> List[Document]:
    """Multi-query retrieval: pool candidates from multiple query variants, then rerank.

    For each query variant, bi-encoder over-retrieves candidates.
    All candidates are pooled and deduplicated. The cross-encoder reranks the
    full pool against the PRIMARY query (first in list).

    Args:
        exclude_ids: Set of document idx strings to exclude (for cross-step dedup).
    """
    if not queries:
        return []
    if len(queries) == 1:
        return retrieve_documents(queries[0], k=k, vectorstore=vectorstore,
                                  exclude_ids=exclude_ids)

    vectorstore = vectorstore or get_vectorstore()
    total_docs = vectorstore._collection.count()

    # For small corpora, simple pooled retrieval + rerank
    if total_docs < 5000:
        all_candidates = []
        seen_idx = set(exclude_ids) if exclude_ids else set()
        fetch_k = k * 3
        for q in queries:
            retriever = get_retriever(k=fetch_k, vectorstore=vectorstore)
            for doc in retriever.invoke(q):
                idx = doc.metadata.get("idx", "")
                if idx not in seen_idx:
                    seen_idx.add(idx)
                    all_candidates.append(doc)
        return rerank_with_cross_encoder(queries[0], all_candidates, top_k=k)

    fetch_k = k * 3

    if SOURCE_DIVERSE_RETRIEVAL:
        # Pool from both source pools across all query variants
        study_candidates = []
        study_seen = set(exclude_ids) if exclude_ids else set()
        case_candidates = []
        case_seen = set(exclude_ids) if exclude_ids else set()

        for q in queries:
            try:
                study_results = vectorstore.similarity_search_with_relevance_scores(
                    q, k=fetch_k, filter={"source": {"$in": ["mbe", "wex"]}}
                )
                for doc, _ in study_results:
                    idx = doc.metadata.get("idx", "")
                    if idx not in study_seen:
                        study_seen.add(idx)
                        study_candidates.append(doc)
            except Exception as e:
                logger.warning("multi-query retrieval pool error (study): %s", e)

            try:
                case_results = vectorstore.similarity_search_with_relevance_scores(
                    q, k=fetch_k, filter={"source": "caselaw"}
                )
                for doc, _ in case_results:
                    idx = doc.metadata.get("idx", "")
                    if idx not in case_seen:
                        case_seen.add(idx)
                        case_candidates.append(doc)
            except Exception as e:
                logger.warning("multi-query retrieval pool error (caselaw): %s", e)

        primary_query = queries[0]
        study_k = (k + 1) // 2
        case_k = k - study_k

        reranked_study = rerank_with_cross_encoder(primary_query, study_candidates, top_k=study_k)
        reranked_case = rerank_with_cross_encoder(primary_query, case_candidates, top_k=case_k)

        result = list(reranked_study)
        seen = {doc.metadata.get("idx", "") for doc in result}
        for doc in reranked_case:
            if len(result) >= k:
                break
            idx = doc.metadata.get("idx", "")
            if idx not in seen:
                seen.add(idx)
                result.append(doc)

        if len(result) < k:
            all_candidates = study_candidates + case_candidates
            all_reranked = rerank_with_cross_encoder(primary_query, all_candidates, top_k=k * 2)
            for doc in all_reranked:
                if len(result) >= k:
                    break
                idx = doc.metadata.get("idx", "")
                if idx not in seen:
                    seen.add(idx)
                    result.append(doc)

        return result
    else:
        # Unified: pool all candidates across all query variants, then rerank
        all_candidates = []
        seen_idx = set(exclude_ids) if exclude_ids else set()

        for q in queries:
            retriever = get_retriever(k=fetch_k, vectorstore=vectorstore)
            for doc in retriever.invoke(q):
                idx = doc.metadata.get("idx", "")
                if idx not in seen_idx:
                    seen_idx.add(idx)
                    all_candidates.append(doc)

        return rerank_with_cross_encoder(queries[0], all_candidates, top_k=k)


def compute_confidence(query: str, docs: List[Document]) -> float:
    """Compute confidence as mean cosine similarity between query and doc embeddings."""
    if not docs:
        return 0.0

    embeddings = get_embeddings()
    query_emb = np.array(embeddings.embed_query(query))
    doc_texts = [doc.page_content for doc in docs]
    doc_embs = np.array(embeddings.embed_documents(doc_texts))

    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
    doc_norms = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-10)
    similarities = doc_norms @ query_norm

    return float(np.mean(similarities))

_memory_store_instance = None

def get_memory_store() -> Chroma:
    """Returns the Chroma QA memory collection singleton (cosine distance)."""
    global _memory_store_instance
    if _memory_store_instance is None:
        embeddings = get_embeddings()
        _memory_store_instance = Chroma(
            collection_name=QA_MEMORY_COLLECTION,
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_DIR,
            collection_metadata={"hnsw:space": "cosine"},
        )
    return _memory_store_instance


@gpu_locked
def check_memory(query: str, threshold: float = 0.92) -> Dict[str, Any]:
    """Check if a similar question has been answered before.

    Uses cosine similarity. Threshold of 0.92 requires near-exact match to
    avoid serving cached answers for substantially different questions.

    Returns {"found": bool, "answer": str, "confidence": float, "question": str}.
    """
    store = get_memory_store()
    if store._collection.count() == 0:
        return {"found": False, "answer": "", "confidence": 0.0, "question": ""}

    results = store.similarity_search_with_relevance_scores(query, k=1)
    if results:
        doc, score = results[0]
        if score >= threshold:
            return {
                "found": True,
                "answer": doc.metadata.get("answer", ""),
                "confidence": score,
                "question": doc.page_content,
            }
    return {"found": False, "answer": "", "confidence": 0.0, "question": ""}


@gpu_locked
def write_to_memory(question: str, answer: str, confidence: float) -> None:
    """Store a question-answer pair in the QA memory collection."""
    store = get_memory_store()
    doc = Document(
        page_content=question,
        metadata={
            "answer": answer,
            "confidence": str(confidence),
            "timestamp": str(time.time()),
        },
    )
    store.add_documents([doc])


if __name__ == "__main__":
    valid_passages = "barexam_qa/passages/barexam_qa_validation.csv"
    if os.path.exists(valid_passages):
        load_passages_to_chroma(valid_passages)
    else:
        print(f"Could not find {valid_passages}")
