"""Evaluation runner for ContextEngine.

Runs the agents on the benchmark dataset, scores every answer on 7 dimensions and
saves the results for evaluation/report.py.

Agents:
    baseline          LangGraph ReAct loop, execute_sql only (control)
    contexthub_react  same loop + same prompt + the 6 context tools (clean ablation)
    contexthub        deterministic context pipeline (LLM only formats the answer)

Usage:
    python -m evaluation.runner [--agent all|both|baseline|contexthub|contexthub_react]
                                [--questions N] [--runs N]
"""

import json
import logging
import math
import re
import statistics
import sys
import time
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).parent
DATASET_PATH = EVAL_DIR / "dataset.json"
RESULTS_DIR = EVAL_DIR / "results"

DIMENSIONS = [
    "execution_accuracy",
    "metric_selection",
    "source_selection",
    "sql_correctness",
    "groundedness",
    "trust_awareness",
    "tool_success",
]

AGENT_CHOICES = {
    "all": ["baseline", "contexthub_react", "contexthub"],
    "both": ["baseline", "contexthub"],
    "baseline": ["baseline"],
    "contexthub": ["contexthub"],
    "contexthub_react": ["contexthub_react"],
}

WARNING_WORDS = ["deprecated", "experimental", "draft", "stale", "not reliable", "unreliable", "do not use",
                 "don't use", "not approved", "not recommended", "should not", "shouldn't", "caution",
                 "not sufficient", "isn't sufficient", "is not sufficient", "not enough"]

SOURCE_WORDS = ["table", "column", "orders", "customers", "payments", "products",
                "net_amount", "total_amount", "metric", "definition"]

Executor = Callable[[str], dict]


def make_agent(name: str):
    from app.agent.registry import get_agent
    return get_agent(name)


def load_dataset(limit: int | None = None) -> list[dict]:
    """Load the evaluation dataset."""
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    questions = data["questions"]
    if limit:
        questions = questions[:limit]
    return questions


# --------------------------------------------------------------------------- helpers

def _executed_queries(response: dict) -> list[str]:
    return [str(c.get("args", {}).get("query", "")) for c in response.get("tool_calls", [])
            if c.get("tool") == "execute_sql"]


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float, Decimal)) and not isinstance(v, bool)


def _split_values(rows: list[dict]) -> tuple[list[float | None], list[str]]:
    numbers: list[float | None] = []
    strings: list[str] = []
    for row in rows:
        for v in row.values():
            if v is None:
                numbers.append(None)
            elif _is_number(v):
                numbers.append(float(v))
            else:
                strings.append(str(v).lower())
    return numbers, strings


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=0.005, abs_tol=0.01)


def results_match(gold_rows: list[dict], pred_rows: list[dict]) -> bool:
    """Does the predicted result contain the gold result?

    Order-insensitive and column-name-insensitive. Every gold number must be matched
    by a distinct predicted number (0.5% tolerance; a percentage may be given as a
    fraction or vice versa) and every gold label (category, month, ...) must be a
    prefix of some predicted label. Extra predicted columns are allowed.
    """
    g_nums, g_strs = _split_values(gold_rows)
    p_nums, p_strs = _split_values(pred_rows)

    for g in g_strs:
        if not any(p.startswith(g) for p in p_strs):
            return False

    remaining = list(p_nums)
    for g in g_nums:
        for i, p in enumerate(remaining):
            if g is None or p is None:
                ok = g is None and p is None
            else:
                ok = _close(g, p) or _close(g, p * 100) or _close(g * 100, p)
            if ok:
                remaining.pop(i)
                break
        else:
            return False
    return True


_SCALE = {"thousand": 1e3, "k": 1e3, "lakh": 1e5, "lakhs": 1e5, "million": 1e6, "mn": 1e6,
          "crore": 1e7, "crores": 1e7, "cr": 1e7, "billion": 1e9, "bn": 1e9}


def _numbers_in_text(text: str) -> list[float]:
    values = []
    for m in re.finditer(r"(\d[\d,]*(?:\.\d+)?)\s*(thousand|k|lakhs?|million|mn|crores?|cr|billion|bn)?\b",
                         text.lower()):
        try:
            v = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        values.append(v * _SCALE.get(m.group(2) or "", 1))
    return values


# --------------------------------------------------------------------------- scorers
# Each scorer returns a value in [0, 1], or None when the dimension does not apply
# to the question. Averages are taken over applicable questions only.

def score_execution_accuracy(response: dict, expected: dict, executor: Executor | None) -> float | None:
    """1.0 if any SQL the agent executed returns the gold result, else 0.0."""
    gold_sql = expected.get("gold_sql")
    if not gold_sql or executor is None:
        return None
    gold = executor(gold_sql)
    if "rows" not in gold:
        raise RuntimeError(f"Gold SQL for {expected.get('id')} failed: {gold.get('error')}")
    for query in _executed_queries(response):
        pred = executor(query)
        if "rows" in pred and results_match(gold["rows"], pred["rows"]):
            return 1.0
    return 0.0


def score_metric_selection(response: dict, expected: dict) -> float | None:
    """Did the agent choose the right business metric?

    1.0 if the answer names the expected metric, or the executed SQL uses the metric's
    column on its table with all parts of its filter; 0.5 if the right column is used
    or named; 0.0 otherwise. Judged from the answer and SQL only, so an agent without
    context tools can score full marks by writing the right SQL.
    """
    expected_metric = expected.get("expected_metric")
    if not expected_metric:
        return None

    answer = response.get("answer", "").lower()
    if expected_metric.lower() in answer or expected_metric.lower().replace("_", " ") in answer:
        return 1.0

    column = (expected.get("expected_source_column") or "").lower()
    table = (expected.get("expected_source_table") or "").lower().replace("business.", "")
    filter_parts = [p.strip().lower() for p in re.split(r"\s+AND\s+", expected.get("expected_filter") or "",
                                                         flags=re.IGNORECASE) if p.strip()]
    for query in (q.lower() for q in _executed_queries(response)):
        uses_column = column and column in query and table in query
        if uses_column and all(p in query or _fuzzy_filter_match(p, query) for p in filter_parts):
            return 1.0

    all_text = answer + " " + " ".join(_executed_queries(response)).lower()
    if column and column in all_text:
        return 0.5
    return 0.0


def score_source_selection(response: dict, expected: dict) -> float | None:
    """1.0 for correct table+column in answer/SQL, 0.5 for correct table only."""
    expected_table = expected.get("expected_source_table", "")
    expected_column = expected.get("expected_source_column", "")
    if not expected_table:
        return None

    all_text = (response.get("answer", "") + " " + " ".join(_executed_queries(response))).lower()
    table_name = expected_table.replace("business.", "").lower()
    has_table = table_name in all_text
    has_column = expected_column.lower() in all_text if expected_column else True

    if has_table and has_column:
        return 1.0
    if has_table:
        return 0.5
    return 0.0


def score_sql_correctness(response: dict, expected: dict) -> float | None:
    """Fraction of the expected filter's parts present in the executed SQL."""
    expected_filter = expected.get("expected_filter")
    if not expected_filter:
        return None

    sql_queries = [q.lower() for q in _executed_queries(response)]
    if not sql_queries:
        return 0.0

    filter_parts = [p for p in re.split(r"\s+AND\s+", expected_filter, flags=re.IGNORECASE) if p.strip()]
    matched = 0
    for part in filter_parts:
        part_lower = part.lower().strip()
        if any(part_lower in q or _fuzzy_filter_match(part_lower, q) for q in sql_queries):
            matched += 1
    return matched / len(filter_parts)


def _strip_qualifiers(sql: str) -> str:
    """'customers.segment' / 'c.segment' -> 'segment', and drop whitespace."""
    sql = re.sub(r"\b[a-z_]+\.(?=[a-z_])", "", sql.lower())
    return re.sub(r"\s+", "", sql)


def _fuzzy_filter_match(expected_part: str, query: str) -> bool:
    """Fuzzy matching for filter conditions (whitespace, table aliases, small variations)."""
    compact_query = _strip_qualifiers(query)
    compact_part = _strip_qualifiers(expected_part)
    if compact_part in compact_query:
        return True
    variations = {
        "is_active = true": ["is_active", "is_active='t'", "is_active is true"],
        "last_order_date >= current_date - interval '90 days'": ["interval '90 days'", "interval '90 day'", "- 90"],
    }
    for alt in variations.get(expected_part, []):
        if alt in query:
            return True
    return False


def score_groundedness(response: dict, expected: dict) -> float | None:
    """Is the answer grounded in data the agent actually retrieved?

    Checks: (1) if SQL returned numbers, the answer states one of them (allowing
    thousand/lakh/crore/million formatting); (2) the answer names a source (table,
    column, metric); (3) tools were used. Trust vocabulary is deliberately not
    rewarded here - see trust_awareness.
    """
    answer = response.get("answer", "")
    tool_calls = response.get("tool_calls", [])
    if not answer:
        return 0.0

    checks: list[float] = []

    result_numbers = []
    for call in tool_calls:
        if call.get("tool") == "execute_sql":
            # Values of result columns (skipping bookkeeping fields like row_count / elapsed_ms)
            result_numbers += [float(n) for n in re.findall(
                r'"(?!row_count|elapsed_ms)[^"]+":\s*"?(-?\d+(?:\.\d+)?)', call.get("result_preview", ""))]
    if result_numbers:
        answer_numbers = _numbers_in_text(answer)
        hit = any(_close(a, r) or math.isclose(a, r, rel_tol=0.01) for a in answer_numbers for r in result_numbers)
        checks.append(1.0 if hit else 0.0)

    checks.append(1.0 if any(w in answer.lower() for w in SOURCE_WORDS) else 0.0)
    checks.append(1.0 if tool_calls else 0.0)
    return sum(checks) / len(checks)


def score_trust_awareness(response: dict, expected: dict) -> float | None:
    """For questions about deprecated/draft/experimental/unreliable assets: does the answer warn?"""
    if not expected.get("expects_warning"):
        return None
    answer = response.get("answer", "").lower()
    return 1.0 if any(w in answer for w in WARNING_WORDS) else 0.0


def score_tool_success(response: dict, expected: dict) -> float | None:
    """Ratio of tool calls that returned no error (None if no tools were called)."""
    tool_calls = response.get("tool_calls", [])
    if not tool_calls:
        return None
    successful = sum(1 for c in tool_calls if '"error"' not in c.get("result_preview", "").lower())
    return successful / len(tool_calls)


def evaluate_response(response: dict, expected: dict, executor: Executor | None = None) -> dict[str, float | None]:
    """Evaluate a single agent response against expected results."""
    return {
        "execution_accuracy": score_execution_accuracy(response, expected, executor),
        "metric_selection": score_metric_selection(response, expected),
        "source_selection": score_source_selection(response, expected),
        "sql_correctness": score_sql_correctness(response, expected),
        "groundedness": score_groundedness(response, expected),
        "trust_awareness": score_trust_awareness(response, expected),
        "tool_success": score_tool_success(response, expected),
    }


def average_scores(scores: list[dict]) -> dict[str, float | None]:
    """Mean per dimension over the questions where it applies."""
    totals = {}
    for dim in DIMENSIONS:
        values = [s[dim] for s in scores if s.get(dim) is not None]
        totals[dim] = round(sum(values) / len(values), 4) if values else None
    return totals


def overall(totals: dict[str, float | None]) -> float:
    values = [v for v in totals.values() if v is not None]
    return sum(values) / len(values) if values else 0.0


# --------------------------------------------------------------------------- running

def run_agent_once(agent, questions: list[dict], executor: Executor, verbose: bool, delay: float) -> dict[str, Any]:
    responses, scores = [], []
    total_latency, total_tool_calls, errors = 0.0, 0, 0

    for i, q in enumerate(questions):
        if verbose:
            print(f"  [{i+1}/{len(questions)}] {q['question'][:60]}...", end=" ", flush=True)
        try:
            response = agent.ask(q["question"])
            question_scores = evaluate_response(response, q, executor)
        except Exception as e:
            logger.error(f"Error evaluating question {q['id']}: {e}")
            response = {"answer": f"Error: {e}", "tool_calls": [], "latency_seconds": 0, "success": False}
            question_scores = {dim: 0.0 for dim in DIMENSIONS}

        responses.append({
            "question_id": q["id"],
            "question": q["question"],
            "answer": response.get("answer", "")[:500],
            "tool_calls": response.get("tool_calls", []),
            "latency": response.get("latency_seconds", 0),
            "success": response.get("success", False),
        })
        scores.append({"question_id": q["id"], **question_scores})
        total_latency += response.get("latency_seconds", 0)
        total_tool_calls += len(response.get("tool_calls", []))
        if not response.get("success"):
            errors += 1

        if verbose:
            avg = overall(question_scores)
            status = "PASS" if avg > 0.6 else "PART" if avg > 0.3 else "FAIL"
            exec_score = question_scores["execution_accuracy"]
            exec_str = "-" if exec_score is None else f"{exec_score:.0f}"
            print(f"{status} (avg {avg:.2f}, exec {exec_str}, {response.get('latency_seconds', 0):.1f}s)")

        time.sleep(delay)  # Rate limiting for free-tier APIs

    n = max(len(questions), 1)
    return {
        "responses": responses,
        "scores": scores,
        "totals": average_scores(scores),
        "avg_latency": round(total_latency / n, 2),
        "total_tool_calls": total_tool_calls,
        "errors": errors,
    }


def aggregate_runs(runs: list[dict]) -> dict[str, Any]:
    """Mean and standard deviation across repeated runs."""
    mean, std = {}, {}
    for dim in DIMENSIONS:
        values = [r["totals"][dim] for r in runs if r["totals"][dim] is not None]
        mean[dim] = round(statistics.mean(values), 4) if values else None
        std[dim] = round(statistics.stdev(values), 4) if len(values) > 1 else 0.0
    return {
        # First run's details are kept at the top level for the per-question report
        "responses": runs[0]["responses"],
        "scores": runs[0]["scores"],
        "totals": mean,
        "totals_std": std,
        "avg_latency": round(statistics.mean(r["avg_latency"] for r in runs), 2),
        "total_tool_calls": round(statistics.mean(r["total_tool_calls"] for r in runs), 1),
        "errors": sum(r["errors"] for r in runs),
        "runs": [{"totals": r["totals"], "avg_latency": r["avg_latency"], "errors": r["errors"]} for r in runs],
    }


def run_evaluation(
    agent_type: str = "all",
    limit: int | None = None,
    verbose: bool = True,
    runs: int = 1,
    delay: float = 2.0,
) -> dict[str, Any]:
    """Run the full evaluation pipeline and save the results as JSON."""
    from app.config import get_settings
    from app.mcp.tools import ContextTools

    questions = load_dataset(limit)
    executor = ContextTools().execute_sql
    settings = get_settings()

    if verbose:
        print(f"\n{'='*64}")
        print(f"  ContextEngine Evaluation - {len(questions)} questions x {runs} run(s)")
        print(f"  LLM provider: {settings.llm_provider.value}")
        print(f"{'='*64}\n")

    results = {
        "timestamp": datetime.now().isoformat(),
        "num_questions": len(questions),
        "num_runs": runs,
        "llm_provider": settings.llm_provider.value,
        "dimensions": DIMENSIONS,
        "agents": {},
    }

    for agent_name in AGENT_CHOICES[agent_type]:
        agent = make_agent(agent_name)
        agent_runs = []
        for run in range(runs):
            if verbose:
                print(f"\n--- {agent_name.upper()} (run {run+1}/{runs}) ---\n")
            agent_runs.append(run_agent_once(agent, questions, executor, verbose, delay))
        results["agents"][agent_name] = aggregate_runs(agent_runs)

    if verbose and len(results["agents"]) > 1:
        print_comparison(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"eval_{timestamp}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    if verbose:
        print(f"\nResults saved to: {results_path}")
    return results


def _pct(value: float | None, std: float | None = None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}" + (f" +/-{std:.1%}" if std else "")


def print_comparison(results: dict) -> None:
    """Print a formatted comparison table (ASCII only, safe on Windows consoles)."""
    agents = results["agents"]
    names = list(agents)
    width = 18
    print(f"\n{'='*(26 + width * len(names))}")
    print("  EVALUATION RESULTS")
    print(f"{'='*(26 + width * len(names))}\n")
    print(f"{'Metric':<26}" + "".join(f"{n:>{width}}" for n in names))
    print("-" * (26 + width * len(names)))
    for dim in DIMENSIONS:
        row = f"{dim:<26}"
        for n in names:
            row += f"{_pct(agents[n]['totals'][dim], agents[n].get('totals_std', {}).get(dim)):>{width}}"
        print(row)
    print("-" * (26 + width * len(names)))
    print(f"{'Overall average':<26}" + "".join(f"{_pct(overall(agents[n]['totals'])):>{width}}" for n in names))
    print(f"{'Avg latency (s)':<26}" + "".join(f"{agents[n]['avg_latency']:>{width}.1f}" for n in names))
    print(f"{'Tool calls (per run)':<26}" + "".join(f"{agents[n]['total_tool_calls']:>{width}}" for n in names))
    print(f"{'Errors':<26}" + "".join(f"{agents[n]['errors']:>{width}}" for n in names))
    print()


def main():
    """CLI entry point for evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Run ContextEngine evaluation")
    parser.add_argument("--agent", choices=list(AGENT_CHOICES), default="all")
    parser.add_argument("--questions", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--runs", type=int, default=1, help="Repeat each agent N times (reports mean +/- std)")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between questions (rate limits)")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    run_evaluation(agent_type=args.agent, limit=args.questions, verbose=not args.quiet,
                   runs=args.runs, delay=args.delay)


if __name__ == "__main__":
    main()
