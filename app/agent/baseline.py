"""Baseline agent - has database access but NO business context layer.

This agent can only see the raw schema (in its system prompt) and execute SQL queries.
It represents the "naive" approach where an LLM is given database access without
understanding business definitions, trust signals, or metric semantics.

Used as the control group in evaluation. It runs the same ReAct graph as the
ContextEngine ReAct agent (app/agent/react.py); only the tools and prompt differ.
"""

from app.agent.prompts import BASELINE_SYSTEM_PROMPT
from app.agent.react import AgentState, ReActAgent  # noqa: F401  (AgentState re-exported)
from app.agent.tools import SQL_ONLY_TOOLS

# Baseline only gets SQL execution - no context tools
BASELINE_TOOLS = SQL_ONLY_TOOLS


class BaselineAgent(ReActAgent):
    """Baseline agent with only schema knowledge and SQL execution."""

    def __init__(self):
        super().__init__(name="baseline", system_prompt=BASELINE_SYSTEM_PROMPT, tools=BASELINE_TOOLS)
