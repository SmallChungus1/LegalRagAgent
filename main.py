import functools
import json
import logging
import os
import re
import sys
import time
import requests
from typing import Any, Dict, List, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from llm_config import get_llm, get_provider_info
from rag_utils import retrieve_documents_multi_query, compute_confidence, check_memory, write_to_memory

# ---------------------------------------------------------------------------
# 1. Core Data Models
# ---------------------------------------------------------------------------

class PlanStep(BaseModel):
    step_id: float
    status: Literal["pending", "completed", "failed"] = "pending"
    phase: str
    question: str
    execution: Dict[str, Any] = Field(default_factory=dict)


class AgentState(TypedDict):
    global_objective: str
    planning_table: List[PlanStep]
    query_type: str            # "simple" or "multi_hop"
    final_cited_answer: str    # aggregated output
    accumulated_context: List[Dict[str, Any]]  # step summaries for replanner
    iteration_count: int       # cycle counter for loop guard
    initial_balance: Dict[str, Any]            # initial deepseek balance snapshot
    injection_check: Dict[str, Any]            # {"is_safe": bool, "reasoning": str}
    verification_result: Dict[str, Any]        # {"is_verified": bool, "issues": [...], "reasoning": str}
    memory_hit: Dict[str, Any]                 # {"found": bool, "answer": str, "confidence": float}
    run_metrics: Dict[str, Any]                # aggregated metrics from observability node


# ---------------------------------------------------------------------------
# 2. Skill Loaders
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def load_skill_instructions(skill_name: str) -> str:
    """Loads markdown instructions from the skills/ directory (cached after first read)."""
    skill_path = os.path.join("skills", f"{skill_name}.md")
    try:
        with open(skill_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"[WARNING: Instruction file '{skill_path}' not found!]"


def _parse_json(text: str) -> Any:
    """Forgiving JSON parser that handles common LLM output issues.

    Handles: markdown fences, surrounding prose, trailing commas,
    single quotes, JS-style comments. Returns None on failure.
    """
    # Fast path
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        pass

    # Extract first JSON object or array from surrounding prose
    for pattern in [r"(\[[\s\S]*\])", r"(\{[\s\S]*\})"]:
        match = re.search(pattern, cleaned)
        if match:
            candidate = match.group(1)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                # Fix trailing commas before } or ]
                fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
                # Remove JS-style single-line comments
                fixed = re.sub(r"//.*?$", "", fixed, flags=re.MULTILINE)
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    continue

    global _parse_failure_count
    _parse_failure_count += 1
    logger.warning("JSON parse failure #%d (input length: %d chars)", _parse_failure_count, len(text))
    return None


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM call counter for observability
# ---------------------------------------------------------------------------
_llm_call_counter: Dict[str, Any] = {"count": 0, "input_chars": 0, "output_chars": 0}
_parse_failure_count: int = 0


def _reset_llm_call_counter() -> None:
    global _parse_failure_count
    _llm_call_counter["count"] = 0
    _llm_call_counter["input_chars"] = 0
    _llm_call_counter["output_chars"] = 0
    _parse_failure_count = 0


def _log_cache_metrics(response, label: str) -> None:
    """Log prompt-cache hit metrics when the provider reports them."""
    # LangChain-normalized path (OpenAI / vLLM)
    usage = getattr(response, "usage_metadata", None) or {}
    if isinstance(usage, dict):
        details = usage.get("input_token_details", {})
        cached = details.get("cache_read")
        total = usage.get("input_tokens")
        if cached is not None and total:
            pct = cached / total * 100
            logger.info("[%s] Prefix cache: %d/%d prompt tokens (%.0f%%)", label, cached, total, pct)
            return

    # Provider-specific path (DeepSeek)
    meta = getattr(response, "response_metadata", None) or {}
    token_usage = meta.get("token_usage", {})
    cached = token_usage.get("prompt_cache_hit_tokens")
    total = token_usage.get("prompt_tokens")
    if cached is not None and total:
        pct = cached / total * 100
        logger.info("[%s] Prefix cache: %d/%d prompt tokens (%.0f%%)", label, cached, total, pct)


_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2  # seconds

def _get_deepseek_balance() -> Dict[str, Any]:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()

    if "deepseek" not in provider or not api_key:
        return {"is_available": False}

    try:
        url = "https://api.deepseek.com/user/balance"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        data = res.json()
        data["is_available"] = True
        return data
    except Exception as e:
        logger.warning(f"Failed to fetch DeepSeek balance: {e}")
        return {"is_available": False}


def _extract_retry_delay(error_str: str) -> float | None:
    """Extract suggested retry delay from API error message (e.g. 'retryDelay': '17s')."""
    match = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)\s*s", error_str)
    if match:
        return float(match.group(1))
    return None


def _llm_call(system_prompt: str, user_prompt: str, label: str = "") -> str:
    """Make an LLM call with retry on transient errors.

    Retries up to _MAX_RETRIES times on rate-limit, connection, and timeout
    errors. Uses the API's suggested retry delay when available, falling
    back to exponential backoff (2s, 4s, 8s).
    """
    llm = get_llm()
    model_name = get_provider_info().get("model", "").lower()
    # Gemma models don't support system messages — merge into the user prompt
    if "gemma" in model_name:
        messages = [
            HumanMessage(content=f"[INSTRUCTIONS]\n{system_prompt}\n[/INSTRUCTIONS]\n\n{user_prompt}"),
        ]
    else:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

    last_error = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = llm.invoke(messages)
            if label:
                _log_cache_metrics(response, label)

            # Track call metrics for observability
            _llm_call_counter["count"] += 1
            _llm_call_counter["input_chars"] += len(system_prompt) + len(user_prompt)
            _llm_call_counter["output_chars"] += len(response.content)

            return response.content
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            is_retryable = (
                "429" in err_str or "rate" in err_str or "too many" in err_str
                or "connection" in err_str or "timeout" in err_str
                or isinstance(e, (ConnectionError, TimeoutError))
            )
            if is_retryable and attempt < _MAX_RETRIES - 1:
                # Use API-suggested delay if available, otherwise exponential backoff
                api_delay = _extract_retry_delay(str(e))
                delay = min(api_delay or _RETRY_BASE_DELAY * (2 ** attempt), 60)
                logger.warning("[%s] Retryable error (attempt %d/%d). Retrying in %.0fs...",
                               label, attempt + 1, _MAX_RETRIES, delay)
                time.sleep(delay)
            else:
                raise

    raise last_error  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 3. Skill Wrappers
# ---------------------------------------------------------------------------

def skill_classify_and_route(objective: str) -> Dict[str, str]:
    """Classify objective as simple or multi_hop."""
    instructions = load_skill_instructions("classify_and_route")
    user_msg = f"Objective: {objective}"
    raw = _llm_call(instructions, user_msg, label="classify")
    parsed = _parse_json(raw)
    if parsed and isinstance(parsed, dict) and "query_type" in parsed:
        return parsed
    return {"query_type": "multi_hop", "reasoning": "Fallback — could not parse classifier output"}


def skill_plan_synthesis(objective: str, query_type: str) -> List[Dict]:
    """Generate a plan as a list of step dicts."""
    instructions = load_skill_instructions("plan_synthesis")
    user_msg = f"Objective: {objective}\nQuery type: {query_type}"
    raw = _llm_call(instructions, user_msg, label="plan")
    parsed = _parse_json(raw)
    if parsed and isinstance(parsed, list):
        return parsed
    # Fallback: single step
    return [{
        "step_id": 1.0,
        "phase": "Direct Research",
        "question": objective,
    }]


def skill_query_rewrite(question: str) -> Dict[str, Any]:
    """Rewrite a question into a primary query + alternative queries for multi-query retrieval.

    Returns {"primary": str, "alternatives": List[str]}.
    Falls back to treating the raw output as primary with no alternatives.
    """
    instructions = load_skill_instructions("query_rewrite")
    raw = _llm_call(instructions, f"Question: {question}", label="query_rewrite").strip()
    parsed = _parse_json(raw)
    if parsed and isinstance(parsed, dict) and "primary" in parsed:
        return {
            "primary": parsed["primary"],
            "alternatives": parsed.get("alternatives", []),
        }
    # Fallback: treat entire output as primary (handles old plain-text format)
    return {"primary": raw, "alternatives": []}


def skill_synthesize_and_cite(question: str, evidence: List[str]) -> str:
    """Synthesize an answer from evidence with inline citations in a single pass."""
    instructions = load_skill_instructions("synthesize_and_cite")
    evidence_text = "\n\n".join(
        f"[Passage {i+1}]: {text}" for i, text in enumerate(evidence)
    )
    user_msg = f"Question: {question}\n\nEvidence:\n{evidence_text}"
    return _llm_call(instructions, user_msg, label="synthesize_and_cite").strip()


def skill_adaptive_replan(objective: str, accumulated_context: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Decide next action based on accumulated research evidence."""
    instructions = load_skill_instructions("adaptive_replan")
    context_summary = json.dumps(accumulated_context, indent=2)
    user_msg = f"Objective: {objective}\n\nAccumulated context:\n{context_summary}"
    raw = _llm_call(instructions, user_msg, label="replan")
    parsed = _parse_json(raw)
    if parsed and isinstance(parsed, dict) and "action" in parsed:
        return parsed
    # Fallback: stop planning
    return {"action": "complete", "reasoning": "Fallback — could not parse replanner output"}


def skill_detect_prompt_injection(objective: str) -> Dict[str, Any]:
    """Classify input as safe or adversarial."""
    instructions = load_skill_instructions("detect_prompt_injection")
    raw = _llm_call(instructions, f"User input: {objective}", label="injection_check")
    parsed = _parse_json(raw)
    if parsed and isinstance(parsed, dict) and "is_safe" in parsed:
        return parsed
    # Fail-open: assume safe if parser fails
    return {"is_safe": True, "reasoning": "Fallback — could not parse injection check output, assuming safe"}


def skill_select_mc_answer(objective: str, research: str) -> str:
    """Given accumulated research and an MC question, select the best answer.

    This is a final-stage call that applies completed legal research to pick
    a letter.  It runs ONCE after all research steps and verification are done,
    keeping the research pipeline itself unbiased.
    """
    system_prompt = (
        "You are a bar exam answer selector. Apply legal research to a multiple-choice question.\n\n"
        "Method:\n"
        "1. Identify the legal rule and its ELEMENTS from the research\n"
        "2. For EACH choice, check whether every required element is met by its specific facts\n"
        "3. Eliminate choices where an element is missing or a defense/exception applies\n"
        "4. Select the choice where all elements are satisfied and no defense negates liability\n\n"
        "Be precise: apply the exact legal standard from the research to each choice's facts. "
        "Do not assume facts not stated in the choice.\n\n"
        "Format your response EXACTLY as:\n"
        "**Answer: (X)**\n"
        "Reasoning: [For each choice, state which specific element is satisfied or missing]"
    )
    user_msg = f"Question:\n{objective}\n\nLegal research:\n{research}"
    return _llm_call(system_prompt, user_msg, label="mc_select").strip()


def skill_verify_answer(question: str, answer: str, evidence: List[str]) -> Dict[str, Any]:
    """Cross-check an answer against retrieved evidence."""
    instructions = load_skill_instructions("verify_answer")
    evidence_text = "\n\n".join(
        f"[Passage {i+1}]: {text}" for i, text in enumerate(evidence)
    )
    user_msg = f"Question: {question}\n\nAnswer:\n{answer}\n\nEvidence:\n{evidence_text}"
    raw = _llm_call(instructions, user_msg, label="verify")
    parsed = _parse_json(raw)
    if parsed and isinstance(parsed, dict) and "is_verified" in parsed:
        return parsed
    # Fallback: assume verified if parser fails
    return {"is_verified": True, "issues": [], "reasoning": "Fallback — could not parse verification output, assuming verified"}


# ---------------------------------------------------------------------------
# 4. Graph Nodes
# ---------------------------------------------------------------------------

def detect_injection_node(state: AgentState) -> AgentState:
    """Screen user input for adversarial prompt injection.

    Skippable via SKIP_INJECTION_CHECK=1 env var (saves 1 LLM call for eval/testing).
    """
    print("\n--- DETECT INJECTION NODE ---")
    
    if "initial_balance" not in state or not state.get("initial_balance"):
        state["initial_balance"] = _get_deepseek_balance()

    if os.getenv("SKIP_INJECTION_CHECK", "0") == "1":
        print("Injection check SKIPPED (SKIP_INJECTION_CHECK=1)")
        state["injection_check"] = {"is_safe": True, "reasoning": "Skipped via SKIP_INJECTION_CHECK"}
        return state

    objective = state["global_objective"]
    result = skill_detect_prompt_injection(objective)
    state["injection_check"] = result
    if result.get("is_safe", True):
        print(f"Input is SAFE: {result.get('reasoning', '')}")
    else:
        print(f"Input is ADVERSARIAL: {result.get('reasoning', '')}")
        state["final_cited_answer"] = (
            "Request rejected: the input was classified as adversarial. "
            "Please provide a legitimate legal research question."
        )
    return state


def classifier_node(state: AgentState) -> AgentState:
    """Classify the objective to determine routing."""
    print("\n--- CLASSIFIER NODE ---")
    objective = state["global_objective"]
    print(f"Classifying: {objective}")

    result = skill_classify_and_route(objective)
    state["query_type"] = result["query_type"]
    print(f"Classification: {result['query_type']} — {result.get('reasoning', '')}")
    return state


def planner_node(state: AgentState) -> AgentState:
    """Generate a research plan using the LLM.

    Checks QA memory first — if a high-confidence cached answer exists,
    short-circuits plan generation entirely.

    For multi_hop queries, only emits the first step — the replanner
    will adaptively generate subsequent steps based on accumulated evidence.
    """
    print("\n--- PLANNER NODE ---")

    # Check QA memory for a cached answer
    objective = state["global_objective"]
    memory_result = check_memory(objective)
    if memory_result["found"]:
        print(f"MEMORY HIT! Confidence: {memory_result['confidence']:.3f}")
        print(f"Cached question: {memory_result['question']}")
        state["memory_hit"] = memory_result
        state["final_cited_answer"] = memory_result["answer"]
        return state
    state["memory_hit"] = memory_result

    if not state.get("planning_table"):
        query_type = state.get("query_type", "multi_hop")
        # Strip MC choices — planner should research legal concepts, not analyze options
        research_objective = _strip_mc_choices(objective)
        print(f"Generating plan for: {research_objective} (type: {query_type})")

        raw_steps = skill_plan_synthesis(research_objective, query_type)

        # For multi_hop: only take the first step; replanner handles the rest
        if query_type == "multi_hop" and len(raw_steps) > 1:
            raw_steps = raw_steps[:1]
            print("(multi_hop) Truncated to first step — replanner will generate next steps adaptively.")

        steps = []
        for s in raw_steps:
            steps.append(PlanStep(
                step_id=float(s.get("step_id", len(steps) + 1)),
                phase=s.get("phase", "Research"),
                question=s.get("question", objective),
            ))

        state["planning_table"] = steps
        state["accumulated_context"] = []
        state["iteration_count"] = 0
        print(f"Generated {len(steps)} plan steps")
    else:
        print("Plan already exists.")

    _print_table(state["planning_table"])
    return state


def executor_node(state: AgentState) -> AgentState:
    """Execute the next pending plan step with real LLM calls."""
    print("\n--- EXECUTOR NODE ---")
    table = state["planning_table"]

    # Gather doc_ids from prior steps for cross-step deduplication
    prior_doc_ids = set()
    for s in table:
        if s.status in ("completed", "failed") and s.execution:
            prior_doc_ids.update(s.execution.get("retrieved_doc_ids", []))

    for step in table:
        if step.status == "pending":
            print(f"Executing step {step.step_id}: {step.question}")
            if prior_doc_ids:
                print(f"  Excluding {len(prior_doc_ids)} prior doc_ids")

            # 1. Query rewrite (returns primary + alternatives)
            rewrite_result = skill_query_rewrite(step.question)
            optimized_query = rewrite_result["primary"]
            alternatives = rewrite_result.get("alternatives", [])
            print(f"  Primary query: {optimized_query[:80]}...")
            if alternatives:
                for j, alt in enumerate(alternatives):
                    print(f"  Alt {j+1}: {alt[:80]}...")

            # 2. Multi-query retrieve (excluding docs from prior steps)
            all_queries = [optimized_query] + alternatives
            docs = retrieve_documents_multi_query(all_queries, k=5,
                                                  exclude_ids=prior_doc_ids or None)
            evidence = [doc.page_content for doc in docs]
            print(f"  Retrieved {len(evidence)} passages:")
            for j, doc in enumerate(docs):
                idx = doc.metadata.get("idx", "")
                print(f"    [{j+1}] {idx}: {doc.page_content[:80]}...")

            # 3. Synthesize and cite (single pass)
            # Per-step synthesis does pure legal research — no MC choices here.
            # MC selection happens once at the end in verify_answer_node.
            cited_answer = skill_synthesize_and_cite(step.question, evidence)
            print(f"  Synthesized ({len(cited_answer)} chars): {cited_answer[:200]}...")

            # 4. Compute confidence
            confidence = compute_confidence(optimized_query, docs)
            print(f"  Confidence score: {confidence:.3f}")

            step.execution = {
                "cited_answer": cited_answer,
                "optimized_query": optimized_query,
                "sources": evidence,
                "retrieved_doc_ids": [doc.metadata.get("idx", "") for doc in docs],
                "confidence_score": confidence,
            }
            break  # One step per node run

    _print_table(state["planning_table"])
    return state


def evaluator_node(state: AgentState) -> AgentState:
    """Evaluate executed steps and accumulate evidence for the replanner."""
    print("\n--- EVALUATOR NODE ---")
    table = state["planning_table"]

    # Default 0.70 calibrated for gte-large-en-v1.5 (observed range 0.709-0.804).
    # Puts threshold at the floor of observed scores so ~10-20% of steps fail
    # and get retried/rephrased by the replanner. Old default was 0.6 (no-op).
    threshold = float(os.getenv("EVAL_CONFIDENCE_THRESHOLD", "0.70"))

    for step in table:
        if step.status == "pending" and "confidence_score" in step.execution:
            score = step.execution["confidence_score"]
            if score >= threshold:
                print(f"Step {step.step_id} PASSED (score: {score:.3f} >= {threshold})")
                step.status = "completed"
            else:
                print(f"Step {step.step_id} FAILED (score: {score:.3f} < {threshold})")
                step.status = "failed"

            # Accumulate evidence for replanner
            answer_summary = step.execution.get("cited_answer", "")
            if len(answer_summary) > 300:
                answer_summary = answer_summary[:300] + "..."
            state["accumulated_context"].append({
                "step_id": step.step_id,
                "question": step.question,
                "answer": answer_summary,
                "confidence": score,
                "status": step.status,
            })
            state["iteration_count"] = state.get("iteration_count", 0) + 1
            print(f"Iteration count: {state['iteration_count']}")
            break

    # Aggregate final answer when all steps are done
    all_done = all(s.status in ("completed", "failed") for s in table)
    if all_done:
        aggregated = _aggregate_completed_answers(table)
        if aggregated:
            state["final_cited_answer"] = aggregated
            print("Aggregated final cited answer from completed steps.")
        elif not state.get("final_cited_answer"):
            failed_questions = [s.question for s in table if s.status == "failed"]
            state["final_cited_answer"] = (
                "Unable to produce a sufficiently grounded answer. "
                f"All {len(failed_questions)} research step(s) failed to retrieve "
                "high-confidence evidence from the corpus."
            )
            print("All steps failed — set failure message.")

    _print_table(state["planning_table"])
    return state


def replanner_node(state: AgentState) -> AgentState:
    """Adaptively plan the next research step based on accumulated evidence.

    Only fires for multi_hop queries. Decides whether to add a new step,
    retry a failed step, or mark research as complete.
    """
    print("\n--- REPLANNER NODE ---")
    # Strip MC choices — replanner should plan pure legal research
    objective = _strip_mc_choices(state["global_objective"])
    accumulated = state.get("accumulated_context", [])

    try:
        result = skill_adaptive_replan(objective, accumulated)
    except Exception as e:
        logger.error("Replanner failed after retries: %s. Falling back to complete.", e)
        print(f"Replanner error — graceful fallback to verify with existing evidence.")
        result = {"action": "complete", "reasoning": f"Fallback — replanner error: {e}"}
    completed_count = sum(1 for s in accumulated if s.get("status") == "completed")
    failed_count = sum(1 for s in accumulated if s.get("status") == "failed")
    print(f"Accumulated context: {len(accumulated)} steps ({completed_count} completed, {failed_count} failed)")

    action = result.get("action", "complete")
    print(f"Replanner action: {action} — {result.get('reasoning', '')}")

    if action in ("next_step", "retry"):
        # Generate a new step ID
        existing_ids = [s.step_id for s in state["planning_table"]]
        new_id = max(existing_ids) + 1.0 if existing_ids else 1.0

        new_step = PlanStep(
            step_id=new_id,
            phase=result.get("phase", "Adaptive Research"),
            question=result.get("question", state["global_objective"]),
        )
        state["planning_table"].append(new_step)
        print(f"Added new step {new_id}: {new_step.question}")
    else:
        # action == "complete": aggregate final answer
        print("Replanner says research is complete.")
        aggregated = _aggregate_completed_answers(state["planning_table"])
        if aggregated:
            state["final_cited_answer"] = aggregated
            print("Aggregated final cited answer from completed steps.")

    _print_table(state["planning_table"])
    return state


def verify_answer_node(state: AgentState) -> AgentState:
    """MC answer selection node (verification removed — was always passing).

    If the objective contains MC answer choices, runs a single LLM call to
    select the best letter based on accumulated research. Otherwise, passes
    through to memory writeback.
    """
    print("\n--- VERIFY / MC SELECT NODE ---")
    objective = state["global_objective"]
    answer = state.get("final_cited_answer", "")

    if not answer:
        print("No answer produced. Skipping.")
        state["verification_result"] = {"is_verified": True, "issues": [], "reasoning": "No answer produced"}
        return state

    if answer.startswith("Unable to produce") or answer.startswith("Request rejected"):
        print("Answer is a fallback/rejection message — skipping.")
        state["verification_result"] = {"is_verified": False, "issues": ["Answer is a failure message"], "reasoning": "No real answer to verify"}
        return state

    # Auto-pass verification (verifier was always passing — see R1 in pipeline_flags.md)
    state["verification_result"] = {"is_verified": True, "issues": [], "reasoning": "Auto-pass (verifier removed)"}

    # MC answer selection — single final call after all research is done
    if "Answer choices:" in objective:
        print("  MC choices detected — selecting answer...")
        mc_response = skill_select_mc_answer(objective, answer)
        state["final_cited_answer"] = answer + "\n\n---\n\n" + mc_response
        print(f"  MC selection: {mc_response}")
    else:
        print("  No MC choices — passing through.")

    return state


def memory_writeback_node(state: AgentState) -> AgentState:
    """Persist successful query-answer pairs to QA memory for future retrieval."""
    print("\n--- MEMORY WRITEBACK NODE ---")
    memory_hit = state.get("memory_hit", {})
    answer = state.get("final_cited_answer", "")

    # Skip write if answer came from cache (already in memory)
    if memory_hit.get("found", False):
        print("Answer came from memory cache — skipping write.")
        return state

    if not answer:
        print("No answer to cache — skipping write.")
        return state

    # Don't cache unverified answers — they may contain unsupported claims
    vr = state.get("verification_result", {})
    if vr and not vr.get("is_verified", True):
        print("Answer failed verification — skipping memory write.")
        return state

    # Compute average confidence across completed steps
    table = state.get("planning_table", [])
    completed_scores = [
        s.execution.get("confidence_score", 0.0)
        for s in table
        if s.status == "completed" and s.execution
    ]
    avg_confidence = sum(completed_scores) / len(completed_scores) if completed_scores else 0.0

    if avg_confidence >= 0.70:
        objective = state["global_objective"]
        # Strip MC selection from cached answer — only cache the research portion.
        # MC selection is appended after "\n\n---\n\n" and contains "**Answer:".
        # Re-running MC selection each time prevents stale letter answers.
        cache_answer = answer
        if "**Answer:" in answer:
            last_sep = answer.rfind("\n\n---\n\n")
            if last_sep != -1:
                cache_answer = answer[:last_sep]
        write_to_memory(objective, cache_answer, avg_confidence)
        print(f"Wrote to memory (avg confidence: {avg_confidence:.3f}, {len(cache_answer)} chars cached)")
    else:
        print(f"Confidence too low ({avg_confidence:.3f}) — skipping memory write.")

    return state


def observability_node(state: AgentState) -> AgentState:
    """Aggregate and print run metrics before termination."""
    print("\n--- OBSERVABILITY NODE ---")
    table = state.get("planning_table", [])
    completed = sum(1 for s in table if s.status == "completed")
    failed = sum(1 for s in table if s.status == "failed")
    pending = sum(1 for s in table if s.status == "pending")

    memory_hit = state.get("memory_hit", {})
    has_answer = bool(state.get("final_cited_answer", ""))
    injection_safe = state.get("injection_check", {}).get("is_safe", True)
    
    cost_spend = 0.0
    initial_balance = state.get("initial_balance", {})
    if initial_balance.get("is_available"):
        final_balance = _get_deepseek_balance()
        if final_balance.get("is_available"):
            init_infos = {info["currency"]: float(info["total_balance"]) for info in initial_balance.get("balance_infos", [])}
            for info in final_balance.get("balance_infos", []):
                currency = info.get("currency")
                fin_bal = float(info.get("total_balance", 0.0))
                if currency in init_infos:
                    cost_spend += max(0.0, init_infos[currency] - fin_bal)

    metrics = {
        "model": get_provider_info().get("model", "unknown"),
        "total_llm_calls": _llm_call_counter["count"],
        "input_chars": _llm_call_counter["input_chars"],
        "output_chars": _llm_call_counter["output_chars"],
        "cost_spend": f"{cost_spend:.4f} CNY" if cost_spend > 0 else ("< 0.01 CNY" if initial_balance.get("is_available") else "N/A"),
        "parse_failures": _parse_failure_count,
        "iteration_count": state.get("iteration_count", 0),
        "steps_completed": completed,
        "steps_failed": failed,
        "steps_pending": pending,
        "query_type": state.get("query_type", "") or "(not classified)",
        "memory_hit": memory_hit.get("found", False),
        "has_answer": has_answer,
        "injection_safe": injection_safe,
    }
    state["run_metrics"] = metrics

    print(f"\n{'='*50}")
    print("  RUN METRICS SUMMARY")
    print(f"{'='*50}")
    for key, val in metrics.items():
        label = key.replace("_", " ").title()
        print(f"  {label:<22} {val}")
    print(f"{'='*50}\n")

    return state


# ---------------------------------------------------------------------------
# 5. Helpers
# ---------------------------------------------------------------------------

def _aggregate_completed_answers(table: List[PlanStep]) -> str:
    """Aggregate cited answers from completed steps with step headers and scoped citations.

    Rewrites [Source N] → [Query X][Source N] so citations are unambiguous
    across steps. Adds ### Step headers for structure.
    """
    completed = [s for s in table if s.status == "completed" and s.execution]
    if not completed:
        return ""

    sections = []
    for i, step in enumerate(completed, 1):
        answer = step.execution.get("cited_answer", "")
        if not answer:
            continue
        # Rewrite [Source N] → [Query i][Source N] throughout the answer text
        tagged = re.sub(r"\[Source (\d+)\]", rf"[Query {i}][Source \1]", answer)
        sections.append(f"### Step {i}: {step.phase}\n\n{tagged}")

    return "\n\n---\n\n".join(sections)


def _strip_mc_choices(objective: str) -> str:
    """Strip 'Answer choices:' block from objective for unbiased research.

    The planner, executor, and replanner should research legal concepts without
    seeing MC options. Only the classifier (routing) and MC selector (final
    stage) need the choices.
    """
    marker = "\n\nAnswer choices:"
    idx = objective.find(marker)
    return objective[:idx].strip() if idx != -1 else objective


def _print_table(table: List[PlanStep]):
    print("\nPlanning Table:")
    for s in table:
        score = f"  ({s.execution['confidence_score']:.3f})" if "confidence_score" in s.execution else ""
        print(f"  [{s.status.upper():>9}] {s.step_id}: {s.question[:70]}{score}")
    print("-" * 60)


# ---------------------------------------------------------------------------
# 6. Routing
# ---------------------------------------------------------------------------

def route_after_injection(state: AgentState) -> Literal["classifier_node", "observability_node"]:
    """Route after injection check: safe → classifier, unsafe → observability → END."""
    if state.get("injection_check", {}).get("is_safe", True):
        return "classifier_node"
    print("Adversarial input detected. Routing to OBSERVABILITY.")
    return "observability_node"


def route_after_planner(state: AgentState) -> Literal["executor_node", "memory_writeback_node"]:
    """Route after planner: memory hit → memory_writeback, otherwise → executor."""
    if state.get("memory_hit", {}).get("found", False):
        print("MEMORY HIT! Short-circuiting to MEMORY_WRITEBACK.")
        return "memory_writeback_node"
    return "executor_node"


def route_after_evaluator(state: AgentState) -> Literal["executor_node", "replanner_node", "verify_answer_node"]:
    """3-way routing after evaluator:
    - executor_node: pending steps remain
    - replanner_node: multi_hop query, all current steps done, under iteration limit
    - verify_answer_node: simple query done, iteration limit exceeded, or hard step cap
    """
    table = state.get("planning_table", [])
    iteration_count = state.get("iteration_count", 0)

    if iteration_count > 4:
        print("Iteration limit hit (>4). Routing to VERIFY_ANSWER.")
        return "verify_answer_node"

    has_pending = any(step.status == "pending" for step in table)
    if has_pending:
        print("Routing back to EXECUTOR (pending steps)...")
        return "executor_node"

    # All current steps are done
    query_type = state.get("query_type", "simple")
    if query_type == "multi_hop":
        # Hard cap: 3+ completed steps is enough evidence.
        completed_count = sum(1 for s in table if s.status == "completed")
        if completed_count >= 3:
            print(f"Hard step cap ({completed_count} completed). Routing to VERIFY_ANSWER.")
            return "verify_answer_node"

        # Stagnation check: if 3+ consecutive steps all failed with similar
        # scores, the topic isn't in the corpus — skip replanner.
        recent_failed = [
            s.execution.get("confidence_score", 0.0)
            for s in table
            if s.status == "failed" and s.execution
        ]
        if len(recent_failed) >= 3:
            last_three = recent_failed[-3:]
            score_range = max(last_three) - min(last_three)
            if score_range < 0.1 and max(last_three) < 0.35:
                print(f"Stagnation detected: last 3 failures scored {last_three} "
                      f"(range {score_range:.3f}). Topic likely not in corpus.")
                print("Routing to VERIFY_ANSWER (skipping futile replanning).")
                return "verify_answer_node"

        print("All current steps done (multi_hop). Routing to REPLANNER...")
        return "replanner_node"

    print("All steps done (simple). Routing to VERIFY_ANSWER.")
    return "verify_answer_node"


def route_after_replanner(state: AgentState) -> Literal["executor_node", "verify_answer_node"]:
    """2-way routing after replanner:
    - executor_node: new pending steps were added
    - verify_answer_node: replanner said complete, or no pending steps
    """
    table = state.get("planning_table", [])
    has_pending = any(step.status == "pending" for step in table)
    if has_pending:
        print("Routing to EXECUTOR (replanner added new step)...")
        return "executor_node"

    print("Replanner complete. Routing to VERIFY_ANSWER.")
    return "verify_answer_node"


# ---------------------------------------------------------------------------
# 7. Graph Topology
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    workflow = StateGraph(AgentState)

    workflow.add_node("detect_injection_node", detect_injection_node)
    workflow.add_node("classifier_node", classifier_node)
    workflow.add_node("planner_node", planner_node)
    workflow.add_node("executor_node", executor_node)
    workflow.add_node("evaluator_node", evaluator_node)
    workflow.add_node("replanner_node", replanner_node)
    workflow.add_node("verify_answer_node", verify_answer_node)
    workflow.add_node("memory_writeback_node", memory_writeback_node)
    workflow.add_node("observability_node", observability_node)

    workflow.set_entry_point("detect_injection_node")

    workflow.add_conditional_edges(
        "detect_injection_node",
        route_after_injection,
    )
    workflow.add_edge("classifier_node", "planner_node")
    workflow.add_conditional_edges(
        "planner_node",
        route_after_planner,
    )
    workflow.add_edge("executor_node", "evaluator_node")
    workflow.add_conditional_edges(
        "evaluator_node",
        route_after_evaluator,
    )
    workflow.add_conditional_edges(
        "replanner_node",
        route_after_replanner,
    )
    workflow.add_edge("verify_answer_node", "memory_writeback_node")
    workflow.add_edge("memory_writeback_node", "observability_node")
    workflow.add_edge("observability_node", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# 8. Demo Queries
# ---------------------------------------------------------------------------

DEMO_QUERIES = {
    "simple": {
        "objective": "What are the elements of a negligence claim?",
    },
    "multi_hop": {
        "objective": (
            "A police officer pulls over a driver for a broken taillight, smells marijuana, "
            "and searches the car without a warrant, finding illegal drugs. The driver invokes "
            "the Fifth Amendment at trial. What are the driver's constitutional rights and what "
            "legal standards apply to the search and the testimony?"
        ),
    },
    "medium": {
        "objective": (
            "A plaintiff seeks a preliminary injunction to prevent a competitor from using a "
            "similar trademark. What legal standard must the court apply, and what factors "
            "are considered?"
        ),
    },
}


if __name__ == "__main__":
    # Select demo query via CLI arg: python main.py [simple|multi_hop|medium]
    query_key = sys.argv[1] if len(sys.argv) > 1 else "simple"
    if query_key not in DEMO_QUERIES:
        print(f"Unknown query key '{query_key}'. Choose from: {', '.join(DEMO_QUERIES)}")
        sys.exit(1)

    demo = DEMO_QUERIES[query_key]
    print(f"\n{'='*60}")
    print(f"Demo mode: {query_key}")
    print(f"Objective: {demo['objective']}")
    print(f"{'='*60}")

    app = build_graph()

    initial_state = {
        "global_objective": demo["objective"],
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

    _reset_llm_call_counter()
    print("\nStarting Legal RAG Agent...")
    final_state = None
    try:
        for output in app.stream(initial_state):
            # Capture the last state from the stream
            for node_name, node_state in output.items():
                final_state = node_state
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()

    # Print final answer
    print(f"\n{'='*60}")
    print("FINAL CITED ANSWER")
    print(f"{'='*60}")
    if final_state and final_state.get("final_cited_answer"):
        print(final_state["final_cited_answer"])
    else:
        print("(No final answer produced — check step outputs above)")
