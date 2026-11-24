"""One place that knows every agent, used by the API, CLI and evaluation."""

from functools import lru_cache

AGENT_DESCRIPTIONS = {
    "contexthub": "ContextEngine pipeline: deterministic context lookups, LLM only formats the answer",
    "contexthub_react": "ContextEngine ReAct: LLM decides how to use all 7 context tools",
    "baseline": "Baseline: LLM with schema + execute_sql only (control group)",
}

AGENT_LABELS = {
    "contexthub": "ContextEngine (pipeline)",
    "contexthub_react": "ContextEngine (ReAct)",
    "baseline": "Baseline",
}


@lru_cache(maxsize=None)
def get_agent(name: str):
    """Return a shared agent instance (agents are stateless between questions)."""
    if name == "baseline":
        from app.agent.baseline import BaselineAgent
        return BaselineAgent()
    if name == "contexthub":
        from app.agent.contexthub import ContextEngineAgent
        return ContextEngineAgent()
    if name == "contexthub_react":
        from app.agent.contexthub_react import ContextEngineReActAgent
        return ContextEngineReActAgent()
    raise ValueError(f"Unknown agent: {name}. Use one of: {', '.join(AGENT_DESCRIPTIONS)}")
