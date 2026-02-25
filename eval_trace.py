"""Traced experiment: run a few questions through the pipeline with detailed logging.

Shows the full journey of each question:
  1. Classification decision
  2. Plan generated
  3. Query rewrite (primary + alternatives)
  4. Retrieved passages (with source, idx, and preview)
  5. Gold passage check (if applicable)
  6. Synthesized answer
  7. Verification result
  8. MC correctness check (if applicable)

Usage:
  uv run python eval_trace.py                  # Run all trace queries
  uv run python eval_trace.py 3                # Run first N queries only
  uv run python eval_trace.py --query "..."    # Run a custom query
  uv run python eval_trace.py 3 --save         # Run 3 queries, save to case_studies/
"""

import json
import os
import re
import sys
import time
import pandas as pd

os.environ.setdefault("SKIP_INJECTION_CHECK", "1")

from main import (
    build_graph, _reset_llm_call_counter, _llm_call_counter,
    skill_query_rewrite, _get_deepseek_balance
)
from llm_config import get_provider_info
from rag_utils import (
    retrieve_documents, retrieve_documents_multi_query,
    compute_confidence, get_vectorstore, get_memory_store,
)
from eval_comprehensive import _check_mc_correctness


def _load_qa_with_gold():
    """Load QA pairs that have gold passages in the current store."""
    qa = pd.read_csv("datasets/barexam_qa/qa/qa.csv")
    vs = get_vectorstore()
    count = vs._collection.count()
    passages = pd.read_csv("datasets/barexam_qa/barexam_qa_train.csv", nrows=count)
    passage_ids = set(passages["idx"].tolist())
    qa_in = qa[qa["gold_idx"].isin(passage_ids)].copy()

    def full_question(row):
        prompt = str(row["prompt"]) if pd.notna(row["prompt"]) else ""
        q = str(row["question"])
        return (prompt + " " + q).strip()

    qa_in["full_q"] = qa_in.apply(full_question, axis=1)
    return qa_in


def _get_gold_passage(gold_idx: str) -> str:
    """Retrieve the gold passage text from the CSV."""
    vs = get_vectorstore()
    count = vs._collection.count()
    passages = pd.read_csv("datasets/barexam_qa/barexam_qa_train.csv", nrows=count)
    match = passages[passages["idx"] == gold_idx]
    if len(match) > 0:
        return str(match.iloc[0]["text"])
    return ""


def trace_retrieval_only(question: str, gold_idx: str = ""):
    """Trace just the retrieval pipeline for a single question."""
    print(f"\n{'─'*80}")
    print(f"RETRIEVAL TRACE")
    print(f"{'─'*80}")
    print(f"Question: {question[:120]}...")

    # 1. Raw retrieval (no rewrite)
    print(f"\n  ── Stage 1: Raw bi-encoder + cross-encoder (no rewrite) ──")
    t0 = time.time()
    raw_docs = retrieve_documents(question, k=5)
    raw_time = time.time() - t0
    raw_ids = [doc.metadata.get("idx", "") for doc in raw_docs]
    raw_conf = compute_confidence(question, raw_docs)
    gold_in_raw = gold_idx in raw_ids if gold_idx else None

    for i, doc in enumerate(raw_docs):
        idx = doc.metadata.get("idx", "")
        src = doc.metadata.get("source", "")
        gold_marker = " ★ GOLD" if idx == gold_idx else ""
        print(f"    [{i+1}] {idx} ({src}) — {doc.page_content[:100]}...{gold_marker}")
    print(f"    Confidence: {raw_conf:.3f} | Gold found: {gold_in_raw} | Time: {raw_time:.1f}s")

    # 2. Query rewrite
    print(f"\n  ── Stage 2: Query rewrite ──")
    rewrite = skill_query_rewrite(question)
    print(f"    Primary: {rewrite['primary'][:100]}...")
    for j, alt in enumerate(rewrite.get("alternatives", [])):
        print(f"    Alt {j+1}:    {alt[:100]}...")

    # 3. Multi-query retrieval
    print(f"\n  ── Stage 3: Multi-query retrieval ──")
    all_queries = [rewrite["primary"]] + rewrite.get("alternatives", [])
    t0 = time.time()
    mq_docs = retrieve_documents_multi_query(all_queries, k=5)
    mq_time = time.time() - t0
    mq_ids = [doc.metadata.get("idx", "") for doc in mq_docs]
    mq_conf = compute_confidence(rewrite["primary"], mq_docs)
    gold_in_mq = gold_idx in mq_ids if gold_idx else None

    for i, doc in enumerate(mq_docs):
        idx = doc.metadata.get("idx", "")
        src = doc.metadata.get("source", "")
        gold_marker = " ★ GOLD" if idx == gold_idx else ""
        print(f"    [{i+1}] {idx} ({src}) — {doc.page_content[:100]}...{gold_marker}")
    print(f"    Confidence: {mq_conf:.3f} | Gold found: {gold_in_mq} | Time: {mq_time:.1f}s")

    # 4. Gold passage comparison
    if gold_idx:
        gold_text = _get_gold_passage(gold_idx)
        if gold_text:
            print(f"\n  ── Gold passage ({gold_idx}) ──")
            print(f"    {gold_text[:200]}...")

    return {
        "raw_recall": gold_in_raw,
        "mq_recall": gold_in_mq,
        "raw_conf": raw_conf,
        "mq_conf": mq_conf,
        "rewrite": rewrite,
    }


def trace_full_pipeline(question: str, gold_idx: str = "", correct_answer: str = "",
                         choices: dict = None):
    """Run the full pipeline on a single question with detailed tracing."""
    print(f"\n{'='*80}")
    print(f"FULL PIPELINE TRACE")
    print(f"{'='*80}")
    print(f"Question: {question[:200]}...")
    if correct_answer:
        correct_text = choices.get(correct_answer, "") if choices else ""
        print(f"Correct answer: {correct_answer}" + (f" — {correct_text[:80]}" if correct_text else ""))

    # Run retrieval trace first (uses raw question, no choices — tests pure retrieval)
    ret_trace = trace_retrieval_only(question, gold_idx)

    # Build the pipeline objective: append MC choices so the LLM can select among them
    objective = question
    if choices and any(choices.values()):
        choice_text = "\n".join(f"  ({k}) {v}" for k, v in sorted(choices.items()) if v)
        objective = f"{question}\n\nAnswer choices:\n{choice_text}"

    # Run full pipeline
    print(f"\n{'─'*80}")
    print(f"PIPELINE EXECUTION")
    print(f"{'─'*80}")

    _reset_llm_call_counter()
    app = build_graph()
    initial_state = {
        "global_objective": objective,
        "planning_table": [],
        "query_type": "",
        "final_cited_answer": "",
        "accumulated_context": [],
        "iteration_count": 0,
        "injection_check": {},
        "verification_result": {},
        "memory_hit": {},
        "run_metrics": {},
    }

    t0 = time.time()
    final_state = None
    try:
        for output in app.stream(initial_state):
            for node_name, node_state in output.items():
                final_state = node_state
    except Exception as e:
        print(f"\nERROR: {e}")

    elapsed = time.time() - t0
    answer = final_state.get("final_cited_answer", "") if final_state else ""

    print(f"\n{'─'*80}")
    print(f"FINAL ANSWER ({len(answer)} chars, {elapsed:.1f}s)")
    print(f"{'─'*80}")
    print(answer[:500] if answer else "(no answer)")

    # MC correctness check
    mc_result = None
    if correct_answer and choices:
        mc_result = _check_mc_correctness(answer, correct_answer, choices)
        status = "CORRECT ✓" if mc_result["correct"] else "WRONG ✗"
        print(f"\n  MC Check: {status} (method: {mc_result['method']})")
        print(f"  Details: {mc_result['details']}")

    # Extract per-step details for case study output
    steps_detail = []
    if final_state:
        for step in final_state.get("planning_table", []):
            step_info = {
                "step_id": step.step_id,
                "phase": step.phase,
                "question": step.question,
                "status": step.status,
            }
            if step.execution:
                step_info["optimized_query"] = step.execution.get("optimized_query", "")
                step_info["retrieved_doc_ids"] = step.execution.get("retrieved_doc_ids", [])
                step_info["confidence"] = step.execution.get("confidence_score", 0.0)
                # Preview of each retrieved passage
                sources = step.execution.get("sources", [])
                step_info["passage_previews"] = [s[:120] for s in sources]
                # Synthesized answer
                step_info["answer"] = step.execution.get("cited_answer", "")
            steps_detail.append(step_info)

    # Compute avg confidence across completed steps
    completed_confs = [
        s["confidence"] for s in steps_detail
        if s["status"] == "completed" and "confidence" in s
    ]
    avg_conf = sum(completed_confs) / len(completed_confs) if completed_confs else 0.0

    # Summary
    metrics = final_state.get("run_metrics", {}) if final_state else {}
    verification = final_state.get("verification_result", {}) if final_state else {}
    is_verified = verification.get("is_verified", None)
    steps_completed = metrics.get("steps_completed", 0)
    steps_failed = metrics.get("steps_failed", 0)

    print(f"\n{'─'*80}")
    print(f"TRACE SUMMARY")
    print(f"{'─'*80}")
    print(f"  Time: {elapsed:.1f}s | LLM calls: {metrics.get('total_llm_calls', '?')}")
    print(f"  Query type: {final_state.get('query_type', '?') if final_state else '?'}")
    print(f"  Steps: {steps_completed} completed, {steps_failed} failed")
    print(f"  Verified: {is_verified}")
    print(f"  Avg confidence: {avg_conf:.3f}")
    print(f"  Raw retrieval found gold: {ret_trace['raw_recall']}")
    print(f"  Multi-query found gold: {ret_trace['mq_recall']}")
    print(f"  MC correct: {mc_result['correct'] if mc_result else 'n/a'}")

    return {
        **ret_trace,
        "answer": answer,
        "elapsed": elapsed,
        "mc_result": mc_result,
        "metrics": metrics,
        "steps_completed": steps_completed,
        "steps_failed": steps_failed,
        "is_verified": is_verified,
        "avg_confidence": avg_conf,
        "steps_detail": steps_detail,
        "verification": verification,
        "query_type": final_state.get("query_type", "") if final_state else "",
        "accumulated_context": final_state.get("accumulated_context", []) if final_state else [],
    }


def select_trace_queries(n: int = 8):
    """Select a diverse set of questions for tracing."""
    qa = _load_qa_with_gold()
    queries = []

    # Pick 1 per subject (shortest = easiest)
    for subj in ["TORTS", "CONTRACTS", "CRIM. LAW", "EVIDENCE", "CONST. LAW", "REAL PROP."]:
        subj_qs = qa[qa["subject"] == subj].sort_values("full_q", key=lambda x: x.str.len())
        if len(subj_qs) > 0:
            row = subj_qs.iloc[0]
            queries.append({
                "label": f"trace_{subj.lower().replace(' ', '').replace('.', '')}",
                "question": row["full_q"],
                "gold_idx": row["gold_idx"],
                "correct_answer": row["answer"],
                "choices": {
                    "A": str(row["choice_a"]) if pd.notna(row["choice_a"]) else "",
                    "B": str(row["choice_b"]) if pd.notna(row["choice_b"]) else "",
                    "C": str(row["choice_c"]) if pd.notna(row["choice_c"]) else "",
                    "D": str(row["choice_d"]) if pd.notna(row["choice_d"]) else "",
                },
                "subject": subj,
            })

    # Add 1 multi-hop
    queries.append({
        "label": "trace_multihop",
        "question": (
            "A consumer is injured by a defective product. Under what theories can the "
            "manufacturer be held liable, and what defenses are available?"
        ),
        "gold_idx": "",
        "correct_answer": "",
        "choices": {},
        "subject": "MULTI_HOP",
    })

    # Add 1 out-of-corpus
    queries.append({
        "label": "trace_oof",
        "question": "What are the requirements for obtaining asylum in the United States?",
        "gold_idx": "",
        "correct_answer": "",
        "choices": {},
        "subject": "OUT_OF_CORPUS",
    })

    return queries[:n]


def save_case_study(query_info: dict, result: dict, output_dir: str = "case_studies"):
    """Save a single query's full trace to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    label = query_info.get("label", "unknown")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    case = {
        "label": label,
        "subject": query_info.get("subject", ""),
        "question": query_info.get("question", ""),
        "gold_idx": query_info.get("gold_idx", ""),
        "correct_answer": query_info.get("correct_answer", ""),
        "query_type": result.get("query_type", ""),
        "retrieval": {
            "raw_recall": result.get("raw_recall"),
            "mq_recall": result.get("mq_recall"),
            "raw_conf": result.get("raw_conf"),
            "mq_conf": result.get("mq_conf"),
            "rewrite": result.get("rewrite", {}),
        },
        "pipeline": {
            "steps": result.get("steps_detail", []),
            "accumulated_context": result.get("accumulated_context", []),
            "final_answer": result.get("answer", ""),
            "verification": result.get("verification", {}),
            "is_verified": result.get("is_verified"),
            "avg_confidence": result.get("avg_confidence", 0.0),
        },
        "mc_result": result.get("mc_result"),
        "metrics": {
            "elapsed_sec": round(result.get("elapsed", 0), 1),
            "total_llm_calls": result.get("metrics", {}).get("total_llm_calls", 0),
            "steps_completed": result.get("steps_completed", 0),
            "steps_failed": result.get("steps_failed", 0),
        },
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(case, f, indent=2, default=str)

    print(f"  Saved case study: {filepath}")
    return filepath


def main():
    # Set up DualLogger to tee stdout to latest_run_{provider}.txt
    provider_name = os.getenv("LLM_PROVIDER", "default").strip().lower()
    run_log_file = f"latest_run_{provider_name}.txt"
    try:
        with open(run_log_file, "w", encoding="utf-8") as f:
            f.write(f"COMMAND RUN: uv run python {' '.join(sys.argv)}\n")
            f.write("=" * 60 + "\n\n")

        class DualLogger:
            def __init__(self, filepath):
                self.terminal = sys.stdout
                self.log = open(filepath, "a", encoding="utf-8")

            def write(self, message):
                self.terminal.write(message)
                self.log.write(message)

            def flush(self):
                self.terminal.flush()
                self.log.flush()

        sys.stdout = DualLogger(run_log_file)
    except Exception as e:
        print(f"Failed to setup file logger: {e}")

    # Capture initial balance
    initial_balance = _get_deepseek_balance()
    initial_totals = {}
    if initial_balance.get("is_available"):
        for info in initial_balance.get("balance_infos", []):
            initial_totals[info.get("currency")] = float(info.get("total_balance", 0.0))

    pinfo = get_provider_info()
    print(f"Provider: {pinfo['provider']} | Model: {pinfo['model']}")
    vs = get_vectorstore()
    print(f"Corpus: {vs._collection.count()} passages")
    print(f"Embedding: {os.getenv('EMBEDDING_MODEL', 'Alibaba-NLP/gte-large-en-v1.5')}")

    # Clear QA memory cache for clean eval
    mem_store = get_memory_store()
    mem_count = mem_store._collection.count()
    if mem_count > 0:
        mem_ids = mem_store._collection.get()["ids"]
        for i in range(0, len(mem_ids), 5000):
            mem_store._collection.delete(ids=mem_ids[i:i+5000])
        print(f"Cleared QA memory cache ({mem_count} entries)")

    # Parse args
    save_mode = "--save" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--save"]

    if args and args[0] == "--query":
        query = " ".join(args[1:])
        trace_full_pipeline(query)
        return

    n = int(args[0]) if args else 8
    queries = select_trace_queries(n)

    print(f"\nTracing {len(queries)} queries...{'  (saving case studies)' if save_mode else ''}\n")

    results = []
    saved_files = []
    for i, q in enumerate(queries):
        print(f"\n{'#'*80}")
        print(f"# [{i+1}/{len(queries)}] {q['label']} ({q['subject']})")
        print(f"{'#'*80}")

        result = trace_full_pipeline(
            question=q["question"],
            gold_idx=q["gold_idx"],
            correct_answer=q["correct_answer"],
            choices=q["choices"],
        )
        result["label"] = q["label"]
        result["subject"] = q["subject"]
        results.append(result)

        if save_mode:
            path = save_case_study(q, result)
            saved_files.append(path)

    # Final summary table
    print(f"\n\n{'='*100}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*100}")
    print(f"{'Label':<25} {'Subj':<12} {'RawGold':>8} {'MQGold':>8} {'MC':>4} "
          f"{'Steps':>10} {'Vrfy':>5} {'Conf':>6} {'Time':>6} {'LLM':>4}")
    print("-" * 100)
    for r in results:
        raw_g = "Y" if r["raw_recall"] else ("N" if r["raw_recall"] is not None else ".")
        mq_g = "Y" if r["mq_recall"] else ("N" if r["mq_recall"] is not None else ".")
        mc = "Y" if r.get("mc_result", {}) and r["mc_result"]["correct"] else (
            "N" if r.get("mc_result") else "."
        )
        steps_str = f"{r.get('steps_completed', 0)}c/{r.get('steps_failed', 0)}f"
        vrfy = "Y" if r.get("is_verified") is True else (
            "N" if r.get("is_verified") is False else "?"
        )
        conf = f"{r.get('avg_confidence', 0):.3f}"
        t = f"{r['elapsed']:.0f}s"
        llm = str(r.get("metrics", {}).get("total_llm_calls", "?"))
        print(f"{r['label']:<25} {r['subject']:<12} {raw_g:>8} {mq_g:>8} {mc:>4} "
              f"{steps_str:>10} {vrfy:>5} {conf:>6} {t:>6} {llm:>4}")

    # Calculate overall accuracy
    mc_total = sum(1 for r in results if r.get("mc_result"))
    mc_correct = sum(1 for r in results if r.get("mc_result", {}) and r["mc_result"]["correct"])
    if mc_total > 0:
        accuracy = (mc_correct / mc_total) * 100
        print(f"\nOVERALL ACCURACY: {mc_correct}/{mc_total} ({accuracy:.1f}%)")

    # Calculate total cost for the eval run
    if initial_balance.get("is_available"):
        final_balance = _get_deepseek_balance()
        if final_balance.get("is_available"):
            cost_strs = []
            for fin_info in final_balance.get("balance_infos", []):
                currency = fin_info.get("currency")
                fin_tot = float(fin_info.get("total_balance", 0.0))
                init_tot = initial_totals.get(currency, 0.0)
                spent = init_tot - fin_tot
                if spent > 0:
                    cost_strs.append(f"{spent:.4f} {currency}")
                elif init_tot > 0:
                    cost_strs.append(f"< 0.01 {currency} (API precision limitation)")
            if cost_strs:
                print("\n" + "=" * 100)
                print(f"TOTAL API COST: {', '.join(cost_strs)}")
                print("=" * 100)

    if save_mode and saved_files:
        print(f"\nCase studies saved to: {os.path.dirname(saved_files[0])}/")
        for f in saved_files:
            print(f"  {os.path.basename(f)}")


if __name__ == "__main__":
    main()
