"""Generate a markdown evaluation report from results.

Usage:
    python -m evaluation.report [results_file]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.runner import DIMENSIONS, overall  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"

from app.agent.registry import AGENT_LABELS as LABELS  # noqa: E402

DIMENSION_NOTES = {
    "execution_accuracy": "Agent's SQL returns the gold result (numbers checked)",
    "metric_selection": "Right business metric named or computed",
    "source_selection": "Right table and column",
    "sql_correctness": "Required filters present in SQL",
    "groundedness": "Answer states retrieved numbers and names its source",
    "trust_awareness": "Warns on deprecated/draft/experimental/unreliable assets",
    "tool_success": "Tool calls without errors",
}


def find_latest_results() -> Path | None:
    """Find the most recent evaluation results file."""
    if not RESULTS_DIR.exists():
        return None
    files = sorted(RESULTS_DIR.glob("eval_*.json"), reverse=True)
    return files[0] if files else None


def _pct(value, std=None) -> str:
    if value is None:
        return "n/a"
    return f"{value*100:.1f}%" + (f" +/- {std*100:.1f}" if std else "")


def _delta(value, base) -> str:
    if value is None or base is None:
        return ""
    d = (value - base) * 100
    return f" ({'+' if d >= 0 else ''}{d:.1f})"


def generate_report(results_path: Path) -> str:
    """Generate a markdown report from evaluation results."""
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    agents = results.get("agents", {})
    names = list(agents)
    base = agents.get("baseline")
    lines = [
        "# ContextEngine Evaluation Report\n",
        f"**Date:** {results.get('timestamp', 'Unknown')}  ",
        f"**LLM provider:** {results.get('llm_provider', 'unknown')}  ",
        f"**Questions:** {results.get('num_questions', 0)} | **Runs per agent:** {results.get('num_runs', 1)}\n",
        "Scores are averaged over the questions where each dimension applies. "
        "With several runs, values are mean +/- standard deviation across runs. "
        "Numbers in parentheses are percentage points relative to the baseline.\n",
        "## Results\n",
        "| Dimension | " + " | ".join(LABELS.get(n, n) for n in names) + " |",
        "|---|" + "---|" * len(names),
    ]

    for dim in DIMENSIONS:
        cells = []
        for n in names:
            value = agents[n]["totals"].get(dim)
            cell = _pct(value, agents[n].get("totals_std", {}).get(dim))
            if base is not None and n != "baseline":
                cell += _delta(value, base["totals"].get(dim))
            cells.append(cell)
        lines.append(f"| {dim} | " + " | ".join(cells) + " |")

    overall_cells = []
    for n in names:
        cell = _pct(overall(agents[n]["totals"]))
        if base is not None and n != "baseline":
            cell += _delta(overall(agents[n]["totals"]), overall(base["totals"]))
        overall_cells.append(f"**{cell}**")
    lines.append("| **Overall** | " + " | ".join(overall_cells) + " |")

    lines += ["", "| Dimension | What it measures |", "|---|---|"]
    lines += [f"| {d} | {DIMENSION_NOTES[d]} |" for d in DIMENSIONS]

    lines += ["\n## Performance\n", "| | " + " | ".join(LABELS.get(n, n) for n in names) + " |",
              "|---|" + "---|" * len(names)]
    lines.append("| Avg latency (s) | " + " | ".join(f"{agents[n]['avg_latency']:.1f}" for n in names) + " |")
    lines.append("| Tool calls per run | " + " | ".join(str(agents[n]["total_tool_calls"]) for n in names) + " |")
    lines.append("| Errors | " + " | ".join(str(agents[n]["errors"]) for n in names) + " |")

    # Per-question breakdown (first run)
    lines += ["\n## Per-question scores (first run)\n",
              "Cell = execution accuracy / average over applicable dimensions.\n",
              "| ID | Question | " + " | ".join(LABELS.get(n, n) for n in names) + " |",
              "|---|---|" + "---|" * len(names)]
    first = agents[names[0]] if names else {"responses": []}
    for i, resp in enumerate(first.get("responses", [])):
        cells = []
        for n in names:
            score = agents[n]["scores"][i]
            values = {k: v for k, v in score.items() if k != "question_id"}
            exec_score = values.get("execution_accuracy")
            exec_str = "-" if exec_score is None else ("✓" if exec_score == 1 else "✗")
            cells.append(f"{exec_str} / {overall(values)*100:.0f}%")
        lines.append(f"| {resp['question_id']} | {resp['question'][:55]} | " + " | ".join(cells) + " |")

    # Findings
    lines.append("\n## Key findings\n")
    if base is not None:
        for n in names:
            if n == "baseline":
                continue
            label = LABELS.get(n, n)
            for dim in ("execution_accuracy", "metric_selection", "trust_awareness"):
                v, b = agents[n]["totals"].get(dim), base["totals"].get(dim)
                if v is not None and b is not None and v != b:
                    word = "higher" if v > b else "lower"
                    lines.append(f"- {label}: {dim} is {abs(v - b)*100:.1f} points {word} than baseline "
                                 f"({_pct(v)} vs {_pct(b)}).")
            if agents[n]["avg_latency"] > base["avg_latency"] * 1.5:
                lines.append(f"- {label} is slower ({agents[n]['avg_latency']:.1f}s vs {base['avg_latency']:.1f}s): "
                             f"context lookups cost extra tool calls.")
        if "contexthub_react" in agents:
            lines.append("\nThe baseline vs ContextEngine (ReAct) comparison is the clean ablation: same LLM, same loop, "
                         "same schema prompt; only the context tools differ.")
    lines.append("\n**Caveats:** 25 questions on one synthetic schema; heuristic scorers (except execution accuracy) "
                 "check for keywords; results depend on the LLM used.")
    return "\n".join(lines)


def main():
    """CLI entry point."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # Windows consoles can't print every character

    results_path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_latest_results()
    if not results_path or not results_path.exists():
        print("No evaluation results found. Run evaluation first:")
        print("  python -m evaluation.runner")
        sys.exit(1)

    report = generate_report(results_path)
    print(report)

    report_path = RESULTS_DIR / "report.md"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
