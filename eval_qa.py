"""Simple parallel QA evaluation script for the Legal RAG pipeline.

Tests exactly N randomly sampled questions from the BarExam QA dataset
using a fixed random seed for consistent benchmarks across runs.

Usage:
  uv run python eval_qa.py 20              # Evaluate 20 questions
  uv run python eval_qa.py 50 --parallel 5 # Evaluate 50 in parallel with 5 workers
"""

import os
import sys
import time
import concurrent.futures
import threading
import io
import pandas as pd

# Bypass the anti-injection prompt loop to save LLM calls/cost during eval
os.environ.setdefault("SKIP_INJECTION_CHECK", "1")

from main import build_graph, _get_deepseek_balance
from rag_utils import get_memory_store
from llm_config import get_provider_info


def _load_qa_with_gold() -> pd.DataFrame:
    """Load QA pairs whose gold passages exist in the current vector store."""
    from rag_utils import get_vectorstore
    vs = get_vectorstore()
    corpus_size = vs._collection.count()
    qa = pd.read_csv("datasets/barexam_qa/qa/qa.csv")
    passages = pd.read_csv("datasets/barexam_qa/barexam_qa_train.csv", nrows=corpus_size)
    passage_ids = set(passages["idx"].tolist())
    qa_in = qa[qa["gold_idx"].isin(passage_ids)].copy()
    
    # Build complete question text
    def _full_q(row):
        prompt = str(row["prompt"]) if pd.notna(row["prompt"]) else ""
        q = str(row["question"])
        return (prompt + " " + q).strip()
    
    qa_in["full_q"] = qa_in.apply(_full_q, axis=1)
    return qa_in


def select_qa_queries(n: int = 10):
    """Select a deterministic random sample of questions using a fixed seed."""
    qa = _load_qa_with_gold()
    queries = []

    # Sample 'n' questions deterministically to keep benchmarks fair
    sampled_qa = qa.sample(n=min(n, len(qa)), random_state=42)

    for i, row in sampled_qa.iterrows():
        subj_name = str(row["subject"]).lower().replace(" ", "").replace(".", "")
        queries.append({
            "label": f"qa_{subj_name}_{i}",
            "question": row["full_q"],
            "gold_idx": row["gold_idx"],
            "correct_answer": row["answer"],
            "choices": {
                "A": str(row["choice_a"]) if pd.notna(row["choice_a"]) else "",
                "B": str(row["choice_b"]) if pd.notna(row["choice_b"]) else "",
                "C": str(row["choice_c"]) if pd.notna(row["choice_c"]) else "",
                "D": str(row["choice_d"]) if pd.notna(row["choice_d"]) else "",
            },
            "subject": row["subject"],
        })

    return queries


def _check_mc_correctness(answer: str, correct_letter: str) -> bool:
    """Simple check if the pipeline's answer correctly selects the multiple-choice letter."""
    if not correct_letter or not answer:
        return False

    import re
    answer_lower = answer.lower()

    # Search for explicit selection of the correct letter
    letter_patterns = [
        rf'\*\*answer:\s*\({correct_letter}\)\*\*',
        rf'\banswer\s+is\s+\(?{correct_letter}\)?\b',
        rf'\bcorrect\s+answer[:\s]+\(?{correct_letter}\)?\b',
        rf'\b\({correct_letter}\)\s+is\s+correct\b',
        rf'\boption\s+\(?{correct_letter}\)?\s+is\s+correct\b',
        rf'\bselect(?:ing|ed|s)?\s+\(?{correct_letter}\)?\b',
    ]
    for pat in letter_patterns:
        if re.search(pat, answer_lower, re.IGNORECASE):
            return True

    return False


def run_single_query(app, q: dict):
    """Run one query through the LangGraph and extract correctness."""
    objective = q["question"]
    choices = q.get("choices", {})
    if choices and any(choices.values()):
        choice_text = "\n".join(f"  ({k}) {v}" for k, v in sorted(choices.items()) if v)
        objective = f"{objective}\n\nAnswer choices:\n{choice_text}"

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

    start = time.time()
    final_state = None
    error = None
    
    try:
        for output in app.stream(initial_state):
            for node_name, node_state in output.items():
                final_state = node_state
    except Exception as e:
        error = str(e)

    elapsed = time.time() - start
    fs = final_state or {}
    metrics = fs.get("run_metrics", {})
    answer = fs.get("final_cited_answer", "")
    is_correct = _check_mc_correctness(answer, q.get("correct_answer", ""))

    return {
        "label": q["label"],
        "subject": q["subject"],
        "elapsed_sec": round(elapsed, 1),
        "error": error,
        "llm_calls": metrics.get("total_llm_calls", 0),
        "is_correct": is_correct
    }


def main():
    # Parse arguments
    args = sys.argv[1:]
    
    parallel_workers = 1
    if "--parallel" in args:
        idx = args.index("--parallel")
        parallel_workers = int(args[idx + 1])
        args = args[:idx] + args[idx + 2:]

    n = int(args[0]) if len(args) > 0 else 10
    
    # Setup buffered DualLogger
    provider_name = os.getenv("LLM_PROVIDER", "default").strip().lower()
    run_log_file = f"eval_qa_{provider_name}.txt"
    try:
        with open(run_log_file, "w", encoding="utf-8") as f:
            f.write(f"COMMAND RUN: uv run python {' '.join(sys.argv)}\n")
            f.write("=" * 60 + "\n\n")

        class DualLogger:
            def __init__(self, filepath):
                self.terminal = sys.stdout
                self.log = open(filepath, "a", encoding="utf-8")
                self.local = threading.local()
                self._lock = threading.Lock()

            def _get_buffer(self):
                if not hasattr(self.local, 'buffer'):
                    self.local.buffer = io.StringIO()
                return self.local.buffer

            def write(self, message):
                if threading.current_thread() is threading.main_thread():
                    with self._lock:
                        self.terminal.write(message)
                        self.log.write(message)
                else:
                    self._get_buffer().write(message)

            def flush(self):
                with self._lock:
                    self.terminal.flush()
                    self.log.flush()

            def flush_thread_buffer(self):
                if hasattr(self.local, 'buffer'):
                    content = self.local.buffer.getvalue()
                    if content:
                        with self._lock:
                            self.terminal.write(content)
                            self.log.write(content)
                            self.terminal.flush()
                            self.log.flush()
                        self.local.buffer = io.StringIO()

        sys.stdout = DualLogger(run_log_file)
    except Exception as e:
        print(f"Failed to setup file logger: {e}")

    print(f"\n{'='*80}")
    print(f"QA EVALUATION ({n} QUERIES)")
    print(f"{'='*80}\n")
    
    # Capture initial balance
    initial_balance = _get_deepseek_balance()
    initial_totals = {}
    if initial_balance.get("is_available"):
        for info in initial_balance.get("balance_infos", []):
            initial_totals[info.get("currency")] = float(info.get("total_balance", 0.0))

    # Log provider info
    pinfo = get_provider_info()
    print(f"Provider: {pinfo['provider']} | Model: {pinfo['model']}")

    # Clear QA memory cache
    mem_store = get_memory_store()
    mem_ids = mem_store._collection.get()["ids"]
    if mem_ids:
        mem_store._collection.delete(ids=mem_ids)
        print(f"Cleared QA memory cache ({len(mem_ids)} entries).")

    queries = select_qa_queries(n)
    app = build_graph()
    
    print(f"\nEvaluating {len(queries)} questions {'in parallel (' + str(parallel_workers) + ' threads) ' if parallel_workers > 1 else ''}...\n")

    results = []
    eval_start_time = time.time()
    
    def worker_func(i, q):
        try:
            print(f"[{i+1}/{n}] Evaluating {q['label']}...")
            res = run_single_query(app, q)
            mc_tag = "CORRECT" if res["is_correct"] else ("ERROR" if res["error"] else "WRONG")
            print(f"  → Result: {mc_tag} | {res['elapsed_sec']}s | {res['llm_calls']} LLM calls")
            if res["error"]:
                print(f"  → Error: {res['error']}")
            return res
        finally:
            if hasattr(sys.stdout, 'flush_thread_buffer'):
                sys.stdout.flush_thread_buffer()

    if parallel_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [executor.submit(worker_func, i, q) for i, q in enumerate(queries)]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
    else:
        for i, q in enumerate(queries):
            results.append(worker_func(i, q))

    eval_total_time = time.time() - eval_start_time

    # Evaluate Accuracy
    correct = sum(1 for r in results if r["is_correct"])
    errors = sum(1 for r in results if r["error"])
    accuracy = correct / len(queries) * 100 if queries else 0

    # Evaluate Cost
    cost_strs = []
    if initial_balance.get("is_available"):
        final_balance = _get_deepseek_balance()
        if final_balance.get("is_available"):
            for fin_info in final_balance.get("balance_infos", []):
                currency = fin_info.get("currency")
                fin_tot = float(fin_info.get("total_balance", 0.0))
                init_tot = initial_totals.get(currency, 0.0)
                spent = init_tot - fin_tot
                if spent > 0.0001:
                    cost_strs.append(f"{spent:.4f} {currency}")
                elif init_tot > 0:
                    cost_strs.append(f"< 0.01 {currency}")

    print(f"\n\n{'='*80}")
    print("FINAL QA BENCHMARK REPORT")
    print(f"{'='*80}")
    print(f"Accuracy:           {correct}/{len(queries)} ({accuracy:.1f}%)")
    print(f"Failed to execute:  {errors}")
    print(f"Total time elapsed: {eval_total_time:.1f}s")
    if cost_strs:
        print(f"Total API cost:     {', '.join(cost_strs)}")
    
    # Write a quick breakdown at the end
    print("-" * 80)
    print(f"{'Label':<30} {'Result':<10} {'Time':>6} {'LLM':>4}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x["label"]):
        status = "✓" if r["is_correct"] else ("!" if r["error"] else "✗")
        print(f"{r['label']:<30} {status:<10} {r['elapsed_sec']:>5.1f}s {r['llm_calls']:>4}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
