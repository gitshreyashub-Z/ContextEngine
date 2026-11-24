"""ContextEngine ReAct agent - the LLM decides how to use the context tools.

Same LangGraph loop, same LLM and same schema prompt as the baseline; the only
differences are the six extra context tools and instructions on *how* to use them.
The prompt never states the answers (e.g. which column is revenue) - the agent has
to look them up. Comparing this agent with the baseline isolates the effect of the
context layer.
"""

from app.agent.prompts import CONTEXTENGINE_REACT_SYSTEM_PROMPT
from app.agent.react import ReActAgent
from app.agent.tools import ALL_CONTEXT_TOOLS

CONTEXTHUB_TOOLS = ALL_CONTEXT_TOOLS


class ContextEngineReActAgent(ReActAgent):
    """LLM tool-calling agent with full access to the context layer."""

    def __init__(self):
        super().__init__(name="contexthub_react", system_prompt=CONTEXTENGINE_REACT_SYSTEM_PROMPT,
                         tools=CONTEXTHUB_TOOLS)
