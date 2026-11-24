"""ContextEngine CLI - Interactive interface for demo and testing.

Usage:
    python cli.py                          # Interactive mode (ContextEngine pipeline agent)
    python cli.py --agent contexthub_react # LLM-driven ContextEngine agent
    python cli.py --baseline               # Use baseline agent (same as --agent baseline)
    python cli.py --compare                # Run every question on all three agents
    python cli.py --question "..."         # Single question mode
"""

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.agent.registry import AGENT_DESCRIPTIONS, AGENT_LABELS, get_agent

console = Console()

COLORS = {"contexthub": "green", "contexthub_react": "magenta", "baseline": "blue"}
COMPARE_ORDER = ["baseline", "contexthub_react", "contexthub"]


def format_tool_calls(tool_calls: list[dict]) -> str:
    """Format tool calls as a readable trace."""
    if not tool_calls:
        return "  (no tool calls)"

    lines = []
    for i, call in enumerate(tool_calls, 1):
        tool = call.get("tool", "unknown")
        args = call.get("args", {})
        args_str = ", ".join(f'{k}="{v}"' for k, v in args.items())
        lines.append(f"  {i}. {tool}({args_str})")
    return "\n".join(lines)


def display_response(response: dict, show_trace: bool = True) -> None:
    """Display an agent response with formatting."""
    agent = response.get("agent", "unknown")
    color = COLORS.get(agent, "white")
    console.print(f"\n[bold {color}]--- {AGENT_LABELS.get(agent, agent)} ---[/bold {color}]")

    tool_calls = response.get("tool_calls", [])
    if show_trace and tool_calls:
        console.print(f"\n[dim]Tool Calls ({len(tool_calls)}):[/dim]")
        # markup=False: SQL like "[...]" must not be parsed as Rich markup
        console.print(format_tool_calls(tool_calls), style="dim", markup=False, highlight=False)

    console.print("\n[bold]Answer:[/bold]")
    console.print(Panel(response.get("answer", "No answer generated"), border_style=color))

    sources = response.get("sources", [])
    if sources:
        console.print("[dim]Sources:[/dim]")
        for src in sources:
            if src.get("type") == "definition":
                console.print(f"  [dim]- {src['metric']} ({src['status']}) -> {src['source']}[/dim]")
            elif src.get("type") == "trust_signal":
                console.print(f"  [dim]- {src['asset']}: {src['level']}[/dim]")

    console.print(f"\n[dim]Latency: {response.get('latency_seconds', 0):.1f}s | "
                  f"Tools used: {len(tool_calls)}[/dim]")


def display_comparison(responses: list[dict]) -> None:
    """Display a side-by-side comparison of several agents' responses."""
    console.print("\n[bold]--- COMPARISON ---[/bold]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Aspect", style="dim")
    for r in responses:
        agent = r.get("agent", "?")
        table.add_column(AGENT_LABELS.get(agent, agent), style=COLORS.get(agent, "white"))

    table.add_row("Tool Calls", *[str(len(r.get("tool_calls", []))) for r in responses])
    table.add_row("Latency", *[f"{r.get('latency_seconds', 0):.1f}s" for r in responses])
    table.add_row("Sources Cited", *[str(len(r.get("sources", []))) for r in responses])
    table.add_row("Succeeded", *["yes" if r.get("success") else "no" for r in responses])
    console.print(table)


def run_question(question: str, agent_names: list[str], show_trace: bool) -> None:
    responses = []
    for name in agent_names:
        with console.status(f"[bold]{AGENT_LABELS[name]} thinking..."):
            response = get_agent(name).ask(question)
        display_response(response, show_trace=show_trace)
        responses.append(response)
    if len(responses) > 1:
        display_comparison(responses)


@click.command()
@click.option("--agent", "agent_name", type=click.Choice(list(AGENT_DESCRIPTIONS)), default="contexthub",
              help="Agent to use")
@click.option("--baseline", is_flag=True, help="Shortcut for --agent baseline")
@click.option("--compare", is_flag=True, help="Run all three agents and compare")
@click.option("--question", "-q", type=str, help="Single question (non-interactive)")
@click.option("--no-trace", is_flag=True, help="Hide tool call traces")
def main(agent_name: str, baseline: bool, compare: bool, question: str, no_trace: bool):
    """ContextEngine CLI - AI Context Layer for Enterprise Data."""

    console.print(Panel.fit(
        "[bold]ContextEngine[/bold] - AI Context Layer for Enterprise Data\n"
        "[dim]Structured business context makes AI agents more reliable.[/dim]",
        border_style="green",
    ))

    if baseline:
        agent_name = "baseline"
    agent_names = COMPARE_ORDER if compare else [agent_name]
    show_trace = not no_trace

    label = "all agents" if compare else AGENT_LABELS[agent_name]
    console.print(f"\n[dim]Agent: {label} | Type 'quit' to exit | 'trace' toggles traces | 'help'[/dim]\n")

    if question:
        console.print(f"\n[bold]Question:[/bold] {question}")
        run_question(question, agent_names, show_trace)
        return

    while True:
        try:
            user_input = console.input("\n[bold green]>[/bold green] ").strip()

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                console.print("[dim]Goodbye![/dim]")
                break
            if user_input.lower() == "trace":
                show_trace = not show_trace
                console.print(f"[dim]Traces {'enabled' if show_trace else 'disabled'}[/dim]")
                continue
            if user_input.lower() == "help":
                console.print("[dim]Commands: quit, trace, help[/dim]")
                console.print("[dim]Examples:[/dim]")
                console.print("[dim]  What was our revenue in July 2026?[/dim]")
                console.print("[dim]  Which metric should I use for revenue?[/dim]")
                console.print("[dim]  How many active customers do we have?[/dim]")
                console.print("[dim]  What was revenue by product category in July 2026?[/dim]")
                continue

            run_question(user_input, agent_names, show_trace)

        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()
