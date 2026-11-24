"""Shared LangGraph ReAct loop used by the LLM-driven agents.

Both the baseline and the ContextEngine ReAct agent are built from this one graph, so in
the evaluation the *only* difference between them is the system prompt's context
instructions and the list of tools. That makes the comparison a clean ablation.

    agent ──(tool calls?)──► tools ──► agent ──► ... ──► END
"""

import json
import logging
import time
from typing import Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph

from app.agent.llm import get_llm

logger = logging.getLogger(__name__)

# Max graph steps (each LLM turn and each tool turn is one step).
RECURSION_LIMIT = 20


class AgentState(TypedDict):
    """State for the LangGraph agent."""
    messages: list
    tool_calls_log: list[dict[str, Any]]
    start_time: float


def build_react_graph(tools: list, label: str = "agent"):
    """Build a ReAct graph for the given LangChain tools."""
    tools_by_name = {t.name: t for t in tools}

    def agent_node(state: AgentState) -> dict:
        """The LLM reasoning node - decides whether to call a tool or respond."""
        llm = get_llm(temperature=0.0).bind_tools(tools)
        response = llm.invoke(state["messages"])
        # Some providers (Gemini) return content=[] for pure tool-call messages
        if response.content == [] or response.content is None:
            response.content = ""
        return {"messages": state["messages"] + [response]}

    def tool_node(state: AgentState) -> dict:
        """Execute every tool call in the last message and log it."""
        last_message = state["messages"][-1]
        tool_messages = []
        tool_logs = list(state.get("tool_calls_log", []))

        for call in last_message.tool_calls:
            name, args = call["name"], call["args"]
            logger.info(f"[{label}] Tool call: {name}({args})")
            if name in tools_by_name:
                try:
                    result_str = tools_by_name[name].invoke(args)
                except Exception as e:  # bad arguments from the LLM, etc.
                    result_str = json.dumps({"error": f"Tool {name} failed: {e}"})
            else:
                result_str = json.dumps({"error": f"Unknown tool: {name}. Available: {sorted(tools_by_name)}"})

            tool_messages.append(ToolMessage(content=result_str, tool_call_id=call["id"]))
            tool_logs.append({"tool": name, "args": args, "result_preview": result_str[:500], "result": result_str})

        return {"messages": state["messages"] + tool_messages, "tool_calls_log": tool_logs}

    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")
    return workflow.compile()


def extract_sources(tool_calls_log: list[dict]) -> list[dict[str, str]]:
    """Derive cited sources (definitions, trust signals) from tool results."""
    sources = []
    for call in tool_calls_log:
        try:
            result = json.loads(call.get("result", "{}"))
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(result, dict) or "error" in result:
            continue
        if call["tool"] == "get_definition":
            sources.append({
                "type": "definition",
                "metric": result.get("metric_name", ""),
                "status": result.get("status", ""),
                "source": f"{result.get('source_table')}.{result.get('source_column')}",
            })
        elif call["tool"] == "get_trust_signal":
            sources.append({"type": "trust_signal", "asset": result.get("asset_name", ""), "level": result.get("trust_level", "")})
    return sources


class ReActAgent:
    """An LLM tool-calling agent: system prompt + tools + the shared ReAct graph."""

    def __init__(self, name: str, system_prompt: str, tools: list):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools
        self.graph = build_react_graph(tools, label=name)

    def ask(self, question: str) -> dict[str, Any]:
        start_time = time.time()
        initial_state: AgentState = {
            "messages": [SystemMessage(content=self.system_prompt), HumanMessage(content=question)],
            "tool_calls_log": [],
            "start_time": start_time,
        }

        try:
            final_state = self.graph.invoke(initial_state, config={"recursion_limit": RECURSION_LIMIT})
            final_answer = ""
            for msg in reversed(final_state["messages"]):
                if isinstance(msg, AIMessage) and not msg.tool_calls:
                    final_answer = msg.content if isinstance(msg.content, str) else str(msg.content)
                    break
            log = final_state.get("tool_calls_log", [])
            return self._response(question, final_answer, log, start_time, success=True)

        except GraphRecursionError:
            logger.error(f"[{self.name}] Hit recursion limit ({RECURSION_LIMIT} steps)")
            return self._response(question, f"Error: agent did not finish within {RECURSION_LIMIT} steps",
                                  [], start_time, success=False, error="recursion_limit")
        except Exception as e:
            logger.error(f"[{self.name}] Error: {e}")
            return self._response(question, f"Error: {str(e)}", [], start_time, success=False, error=str(e))

    def _response(self, question: str, answer: str, log: list, start_time: float,
                  success: bool, error: str | None = None) -> dict[str, Any]:
        public_log = [{k: v for k, v in call.items() if k != "result"} for call in log]
        return {
            "agent": self.name,
            "question": question,
            "answer": answer,
            "tool_calls": public_log,
            "sources": extract_sources(log),
            "latency_seconds": round(time.time() - start_time, 2),
            "num_tool_calls": len(log),
            "success": success,
            "error": error,
        }
