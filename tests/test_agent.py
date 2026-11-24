"""Tests for the AI agents.

These tests verify agent construction and interface.
Full integration tests require an LLM API key.

Run with: pytest tests/test_agent.py -v
"""

from datetime import date
from unittest.mock import patch

import pytest

from app.agent.prompts import BASELINE_SYSTEM_PROMPT


class TestPrompts:
    """Test that prompts are well-structured."""

    def test_baseline_prompt_has_schema(self):
        """Baseline prompt should include table schemas."""
        assert "business.customers" in BASELINE_SYSTEM_PROMPT
        assert "business.orders" in BASELINE_SYSTEM_PROMPT
        assert "business.products" in BASELINE_SYSTEM_PROMPT
        assert "business.payments" in BASELINE_SYSTEM_PROMPT

    def test_baseline_prompt_has_no_context_instructions(self):
        """Baseline prompt should NOT mention business definitions or trust signals."""
        assert "certified" not in BASELINE_SYSTEM_PROMPT.lower()
        assert "trust signal" not in BASELINE_SYSTEM_PROMPT.lower()
        assert "business definition" not in BASELINE_SYSTEM_PROMPT.lower()

    def test_baseline_prompt_restricts_to_select(self):
        """Baseline prompt should mention the SELECT-only restriction."""
        assert "SELECT" in BASELINE_SYSTEM_PROMPT

    def test_baseline_prompt_does_not_leak_answers(self):
        """The control group must not be told which column is revenue."""
        assert "net_revenue" not in BASELINE_SYSTEM_PROMPT
        assert "90 days" not in BASELINE_SYSTEM_PROMPT


class TestAgentInterface:
    """Test that agents have the correct interface."""

    def test_baseline_agent_has_ask_method(self):
        """BaselineAgent should have an ask() method."""
        from app.agent.baseline import BaselineAgent
        agent = BaselineAgent()
        assert hasattr(agent, "ask")
        assert agent.name == "baseline"

    def test_contexthub_agent_has_ask_method(self):
        """ContextEngineAgent should have an ask() method."""
        from app.agent.contexthub import ContextEngineAgent
        agent = ContextEngineAgent()
        assert hasattr(agent, "ask")
        assert agent.name == "contexthub"

    def test_baseline_has_limited_tools(self):
        """Baseline agent should only have execute_sql tool."""
        from app.agent.baseline import BASELINE_TOOLS
        tool_names = [t.name for t in BASELINE_TOOLS]
        assert "execute_sql" in tool_names
        assert len(tool_names) == 1  # Only SQL execution

    def test_contexthub_react_has_all_tools(self):
        """The ReAct ContextEngine agent gets all 7 tools."""
        from app.agent.contexthub_react import CONTEXTHUB_TOOLS, ContextEngineReActAgent
        names = {t.name for t in CONTEXTHUB_TOOLS}
        assert names == {"search_assets", "get_definition", "get_trust_signal", "get_schema",
                         "get_lineage", "execute_sql", "list_metrics"}
        assert ContextEngineReActAgent().name == "contexthub_react"

    def test_react_prompts_differ_only_by_context_instructions(self):
        """Fair ablation: the context prompt is the baseline prompt plus tool instructions."""
        from app.agent.prompts import CONTEXTENGINE_REACT_SYSTEM_PROMPT
        assert CONTEXTENGINE_REACT_SYSTEM_PROMPT.startswith(BASELINE_SYSTEM_PROMPT)

    def test_context_prompt_does_not_leak_answers(self):
        """Answers must come from the tools, not the prompt."""
        from app.agent.prompts import CONTEXTENGINE_REACT_SYSTEM_PROMPT as p
        for leak in ("net_revenue", "gross_revenue", "90 days", "status = 'completed'", "net_amount for"):
            assert leak not in p


class FakeLLM:
    """Scripted stand-in for a chat model: returns the given AIMessages in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        return self.responses.pop(0)


def _ai_tool_call(name, args, call_id="call_1"):
    from langchain_core.messages import AIMessage
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


class TestReActLoop:
    """The shared LangGraph loop, driven by a fake LLM (no API key needed)."""

    def test_keeps_conversation_history(self):
        """Regression: nodes used to replace `messages`, so after a tool call the LLM saw only
        the tool result - no system prompt, no question."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
        from app.agent import react, tools
        from app.agent.baseline import BaselineAgent

        fake = FakeLLM([_ai_tool_call("execute_sql", {"query": "SELECT 1"}), AIMessage(content="The answer is 1.")])
        with patch.object(react, "get_llm", return_value=fake), \
             patch.object(tools._tools, "execute_sql", return_value={"rows": [{"n": 1}], "query": "SELECT 1"}):
            response = BaselineAgent().ask("What is one?")

        assert response["success"] is True
        assert response["answer"] == "The answer is 1."
        assert [c["tool"] for c in response["tool_calls"]] == ["execute_sql"]
        second_call = fake.calls[1]
        assert [type(m) for m in second_call] == [SystemMessage, HumanMessage, AIMessage, ToolMessage]

    def test_unknown_tool_returns_error_to_llm(self):
        from langchain_core.messages import AIMessage
        from app.agent import react
        from app.agent.baseline import BaselineAgent

        fake = FakeLLM([_ai_tool_call("get_definition", {"metric_name": "x"}), AIMessage(content="ok")])
        with patch.object(react, "get_llm", return_value=fake):
            response = BaselineAgent().ask("q")
        assert "Unknown tool" in response["tool_calls"][0]["result_preview"]

    def test_recursion_limit_is_reported(self):
        from app.agent import react
        from app.agent.baseline import BaselineAgent

        endless = FakeLLM([_ai_tool_call("execute_sql", {"query": "SELECT 1"}, f"c{i}") for i in range(50)])
        with patch.object(react, "get_llm", return_value=endless), \
             patch.object(react.logger, "error"):
            from app.agent import tools
            with patch.object(tools._tools, "execute_sql", return_value={"rows": []}):
                response = BaselineAgent().ask("q")
        assert response["success"] is False
        assert response["error"] == "recursion_limit"

    def test_sources_extracted_from_context_tools(self):
        from langchain_core.messages import AIMessage
        from app.agent import react, tools
        from app.agent.contexthub_react import ContextEngineReActAgent

        fake = FakeLLM([
            _ai_tool_call("get_definition", {"metric_name": "net_revenue"}),
            _ai_tool_call("get_trust_signal", {"asset_name": "net_revenue"}, "call_2"),
            AIMessage(content="done"),
        ])
        with patch.object(react, "get_llm", return_value=fake), \
             patch.object(tools._tools, "get_definition", return_value=NET_REVENUE), \
             patch.object(tools._tools, "get_trust_signal",
                          return_value={"asset_name": "net_revenue", "trust_level": "certified"}):
            response = ContextEngineReActAgent().ask("revenue?")
        assert {"type": "trust_signal", "asset": "net_revenue", "level": "certified"} in response["sources"]
        assert response["sources"][0]["source"] == "business.orders.net_amount"



NET_REVENUE = {
    "metric_name": "net_revenue", "source_table": "business.orders", "source_column": "net_amount",
    "filter_condition": "status = 'completed'", "calculation": "SUM(orders.net_amount) WHERE orders.status = 'completed'",
    "status": "certified",
}
ACTIVE_CUSTOMERS = {
    "metric_name": "active_customer_count", "source_table": "business.customers", "source_column": "id",
    "filter_condition": "is_active = true AND last_order_date >= CURRENT_DATE - INTERVAL '90 days'",
    "calculation": "COUNT(DISTINCT customers.id) ...", "status": "certified",
}
REFUND_RATE = {
    "metric_name": "refund_rate", "source_table": "business.orders", "source_column": "status",
    "filter_condition": "status IN ('completed', 'refunded')", "calculation": "COUNT(...) / COUNT(...) * 100",
    "status": "certified",
}
AOV = {
    "metric_name": "average_order_value", "source_table": "business.orders", "source_column": "net_amount",
    "filter_condition": "status = 'completed'", "calculation": "AVG(orders.net_amount) WHERE ...",
    "status": "certified",
}


class TestDateFilter:
    """ContextEngine date parsing (pure logic, no DB/LLM needed)."""

    @pytest.fixture
    def agent(self):
        from app.agent.contexthub import ContextEngineAgent
        return ContextEngineAgent()

    def test_month_and_year(self, agent):
        f = agent._extract_date_filter("What was our total revenue in July 2026?")
        assert f == "order_date >= '2026-07-01' AND order_date < '2026-08-01'"

    def test_december_rolls_over_year(self, agent):
        f = agent._extract_date_filter("Revenue in December 2025")
        assert f == "order_date >= '2025-12-01' AND order_date < '2026-01-01'"

    def test_last_month(self, agent):
        f = agent._extract_date_filter("What was our revenue last month?", today=date(2026, 8, 10))
        assert f == "order_date >= '2026-07-01' AND order_date < '2026-08-01'"

    def test_last_month_in_january(self, agent):
        f = agent._extract_date_filter("revenue last month", today=date(2026, 1, 15))
        assert f == "order_date >= '2025-12-01' AND order_date < '2026-01-01'"

    def test_month_without_year_uses_most_recent(self, agent):
        f = agent._extract_date_filter("revenue from enterprise customers in July", today=date(2026, 8, 10))
        assert f.startswith("order_date >= '2026-07-01'")
        f = agent._extract_date_filter("revenue in September", today=date(2026, 8, 10))
        assert f.startswith("order_date >= '2025-09-01'")

    def test_year_only(self, agent):
        f = agent._extract_date_filter("How many orders in 2025?")
        assert f == "order_date >= '2025-01-01' AND order_date < '2026-01-01'"

    def test_may_as_verb_is_not_a_month(self, agent):
        assert agent._extract_date_filter("May I see the revenue?") == ""

    def test_custom_date_column(self, agent):
        f = agent._extract_date_filter("in 2025", column="paid_at")
        assert f.startswith("paid_at >= '2025-01-01'")


class TestBuildSQL:
    """ContextEngine SQL generation from definitions."""

    @pytest.fixture
    def agent(self):
        from app.agent.contexthub import ContextEngineAgent
        return ContextEngineAgent()

    def test_sum_metric_with_date(self, agent):
        sql = agent._build_sql("What was our total revenue in July 2026?", NET_REVENUE)
        assert sql.startswith("SELECT SUM(net_amount)")
        assert "status = 'completed'" in sql
        assert "order_date >= '2026-07-01'" in sql

    def test_count_metric(self, agent):
        sql = agent._build_sql("How many active customers do we have?", ACTIVE_CUSTOMERS)
        assert "COUNT(DISTINCT id)" in sql
        assert "INTERVAL '90 days'" in sql

    def test_no_order_date_filter_on_customers_table(self, agent):
        """customers has no order_date column; a date filter would make the SQL fail."""
        sql = agent._build_sql("How many active customers in 2025?", ACTIVE_CUSTOMERS)
        assert "2025" not in sql
        assert " order_date" not in sql  # last_order_date (from the definition) is fine

    def test_rate_metric_is_a_percentage(self, agent):
        """Previously produced COUNT(DISTINCT status), which always returned 2."""
        sql = agent._build_sql("What is our refund rate?", REFUND_RATE)
        assert "COUNT(DISTINCT" not in sql
        assert "FILTER (WHERE status = 'refunded')" in sql
        assert "100.0" in sql

    def test_average_metric(self, agent):
        sql = agent._build_sql("What is our average order value?", AOV)
        assert "AVG(net_amount)" in sql

    def test_group_by_category(self, agent):
        sql = agent._build_sql("What was revenue by product category in July 2026?", NET_REVENUE)
        assert "JOIN business.products p ON p.id = o.product_id" in sql
        assert "GROUP BY p.category" in sql
        assert "order_date >= '2026-07-01'" in sql

    def test_top_n_products(self, agent):
        sql = agent._build_sql("What are our top 5 products by revenue?", NET_REVENUE)
        assert "GROUP BY p.name" in sql
        assert sql.endswith("ORDER BY net_revenue DESC LIMIT 5")

    def test_segment_filter(self, agent):
        sql = agent._build_sql("What was the revenue from enterprise customers in July 2026?", NET_REVENUE)
        assert "JOIN business.customers c ON c.id = o.customer_id" in sql
        assert "c.segment = 'enterprise'" in sql

    def test_group_by_segment(self, agent):
        sql = agent._build_sql("What customer segments generate the most revenue?", NET_REVENUE)
        assert "GROUP BY c.segment" in sql

    def test_segment_on_customers_table_needs_no_join(self, agent):
        sql = agent._build_sql("How many active customers per segment?", ACTIVE_CUSTOMERS)
        assert "JOIN" not in sql
        assert "GROUP BY segment" in sql

    def test_multi_month_comparison(self, agent):
        sql = agent._build_sql("Compare revenue between June and July 2026", NET_REVENUE)
        assert "order_date >= '2026-06-01'" in sql and "order_date >= '2026-07-01'" in sql
        assert "GROUP BY TO_CHAR(order_date, 'YYYY-MM')" in sql
        assert "ORDER BY month" in sql

    def test_synonym_sales_is_revenue(self, agent):
        assert agent._extract_topic("What were sales last month?") == "revenue"
        sql = agent._build_sql("sales by category", NET_REVENUE)
        assert "GROUP BY p.category" in sql

    def test_total_amount_column_is_not_a_data_question(self, agent):
        assert agent._needs_sql("What does the total_amount column in orders represent?") is False
        assert agent._needs_sql("What was our total revenue?") is True

    def test_generated_sql_passes_validator(self, agent):
        from app.mcp.tools import validate_sql
        for q, d in [("revenue in July 2026", NET_REVENUE), ("how many active customers", ACTIVE_CUSTOMERS),
                     ("refund rate", REFUND_RATE), ("average order value", AOV)]:
            assert validate_sql(agent._build_sql(q, d))[0] is True


class TestMetricPicking:
    """Metric selection uses explicit mentions and trust signals (DB calls mocked)."""

    METRICS = [{"metric_name": n} for n in [
        "net_revenue", "gross_revenue", "recognized_revenue", "active_customer_count",
        "total_customer_count", "average_order_value", "order_completion_rate", "refund_rate",
        "monthly_recurring_revenue",
    ]]
    TRUST = {"net_revenue": "certified", "gross_revenue": "deprecated", "recognized_revenue": "experimental",
             "active_customer_count": "certified", "total_customer_count": "certified"}

    @pytest.fixture
    def agent(self):
        from app.agent import contexthub
        with patch.object(contexthub._tools, "list_metrics", return_value={"metrics": self.METRICS}), \
             patch.object(contexthub._tools, "get_trust_signal",
                          side_effect=lambda n: {"trust_level": self.TRUST[n]} if n in self.TRUST else {"error": "x"}):
            yield contexthub.ContextEngineAgent()

    @staticmethod
    def _search(*names):
        return {"results": [{"name": n, "type": "metric", "description": ""} for n in names]}

    def test_prefers_certified_over_deprecated(self, agent):
        # Deprecated listed first: must not be picked just because of ordering
        found = self._search("gross_revenue", "recognized_revenue", "net_revenue")
        assert agent._pick_best_metric("What was our revenue last month?", found) == "net_revenue"

    def test_explicit_mention_wins(self, agent):
        found = self._search("net_revenue", "gross_revenue")
        assert agent._pick_best_metric("What is our gross revenue?", found) == "gross_revenue"

    def test_alias(self, agent):
        assert agent._pick_best_metric("Can I use MRR for reporting?", self._search()) == "monthly_recurring_revenue"

    def test_word_overlap_breaks_ties(self, agent):
        found = self._search("active_customer_count", "total_customer_count")
        assert agent._pick_best_metric("How many active customers do we have?", found) == "active_customer_count"
        assert agent._pick_best_metric("What is the total number of customers?", found) == "total_customer_count"

    def test_rejects_unrelated_metric(self, agent):
        """'cancelled orders' must not be answered with net revenue just because it was found."""
        found = self._search("net_revenue", "active_customer_count")
        assert agent._pick_best_metric("How many orders were cancelled?", found) is None

    def test_explicit_rate_metric(self, agent):
        found = self._search("gross_revenue")  # 'refund' search only finds gross_revenue's description
        assert agent._pick_best_metric("What is our refund rate?", found) == "refund_rate"


class TestEvaluationScoring:
    """Test evaluation scoring functions."""

    def test_metric_selection_scoring(self):
        """Test that metric selection scoring works correctly."""
        from evaluation.runner import score_metric_selection

        # Correct metric mentioned in answer
        response = {"answer": "Using net_revenue metric which is certified.", "tool_calls": []}
        expected = {"expected_metric": "net_revenue"}
        assert score_metric_selection(response, expected) == 1.0

        # Wrong metric
        response = {"answer": "I used gross_revenue for this calculation.", "tool_calls": []}
        expected = {"expected_metric": "net_revenue"}
        assert score_metric_selection(response, expected) == 0.0

    def test_sql_correctness_scoring(self):
        """Test SQL correctness scoring."""
        from evaluation.runner import score_sql_correctness

        # Correct filter applied
        response = {
            "answer": "",
            "tool_calls": [
                {"tool": "execute_sql", "args": {"query": "SELECT SUM(net_amount) FROM business.orders WHERE status = 'completed'"}, "result_preview": '{"rows": []}'}
            ],
        }
        expected = {"expected_filter": "status = 'completed'"}
        score = score_sql_correctness(response, expected)
        assert score == 1.0

    def test_groundedness_scoring(self):
        """Test groundedness scoring."""
        from evaluation.runner import score_groundedness

        # Well-grounded answer
        response = {
            "answer": "Net revenue in July was ₹48,21,320. I used the certified net_revenue metric from the orders table.",
            "tool_calls": [{"tool": "get_definition", "args": {}}],
        }
        score = score_groundedness(response, {})
        assert score > 0.5

        # Ungrounded answer
        response = {"answer": "I think maybe it's around some amount.", "tool_calls": []}
        score = score_groundedness(response, {})
        assert score < 0.5


class TestFairScoring:
    """The v2 scorers: execution accuracy and removal of context-only bonuses."""

    def test_results_match_order_and_column_insensitive(self):
        from evaluation.runner import results_match
        gold = [{"category": "AI/ML", "sum": 10.0}, {"category": "Data", "sum": 5.0}]
        pred = [{"cat": "Data", "revenue": 5.0}, {"cat": "AI/ML", "revenue": 10.0, "orders": 3}]
        assert results_match(gold, pred) is True

    def test_results_match_rejects_wrong_number(self):
        from evaluation.runner import results_match
        assert results_match([{"v": 291657723.14}], [{"v": 432045530.44}]) is False

    def test_results_match_percentage_or_fraction(self):
        from evaluation.runner import results_match
        assert results_match([{"rate": 7.557}], [{"rate": 0.07557}]) is True
        assert results_match([{"rate": 7.557}], [{"rate": 7.56}]) is True

    def test_results_match_month_label_prefix(self):
        from evaluation.runner import results_match
        assert results_match([{"m": "2026-06", "v": 1.0}], [{"month": "2026-06-01 00:00:00", "v": 1.0}]) is True

    def test_execution_accuracy_uses_executor(self):
        from evaluation.runner import score_execution_accuracy
        results = {"GOLD": {"rows": [{"v": 100}]}, "RIGHT": {"rows": [{"x": 100}]}, "WRONG": {"rows": [{"x": 150}]}}
        expected = {"gold_sql": "GOLD"}

        def response(query):
            return {"tool_calls": [{"tool": "execute_sql", "args": {"query": query}}]}

        assert score_execution_accuracy(response("RIGHT"), expected, results.__getitem__) == 1.0
        assert score_execution_accuracy(response("WRONG"), expected, results.__getitem__) == 0.0
        assert score_execution_accuracy(response("RIGHT"), {"gold_sql": None}, results.__getitem__) is None

    def test_metric_selection_credits_correct_sql_without_metric_name(self):
        """A baseline that writes the right SQL gets full credit, even though it
        has never heard of 'net_revenue'."""
        from evaluation.runner import score_metric_selection
        expected = {"expected_metric": "net_revenue", "expected_source_table": "business.orders",
                    "expected_source_column": "net_amount", "expected_filter": "status = 'completed'"}
        response = {"answer": "Revenue was 30,552,981.27", "tool_calls": [{"tool": "execute_sql", "args": {
            "query": "SELECT SUM(net_amount) FROM business.orders WHERE status='completed'"}}]}
        assert score_metric_selection(response, expected) == 1.0

    def test_no_bonus_for_just_calling_get_definition(self):
        from evaluation.runner import score_metric_selection
        expected = {"expected_metric": "net_revenue", "expected_source_column": "net_amount"}
        response = {"answer": "It is about 43 crore.", "tool_calls": [
            {"tool": "get_definition", "args": {"metric_name": "net_revenue"}}]}
        assert score_metric_selection(response, expected) == 0.0

    def test_groundedness_does_not_reward_trust_words(self):
        from evaluation.runner import score_groundedness
        base = {"answer": "Revenue was 100 from the orders table.", "tool_calls": [
            {"tool": "execute_sql", "args": {}, "result_preview": '{"rows": [{"v": "100"}], "row_count": 1}'}]}
        with_trust = {**base, "answer": base["answer"] + " This metric is certified."}
        assert score_groundedness(base, {}) == score_groundedness(with_trust, {}) == 1.0

    def test_groundedness_checks_number_matches_result(self):
        from evaluation.runner import score_groundedness
        response = {"answer": "Revenue was 999 from the orders table.", "tool_calls": [
            {"tool": "execute_sql", "args": {}, "result_preview": '{"rows": [{"v": "100"}], "row_count": 1}'}]}
        assert score_groundedness(response, {}) < 1.0

    def test_groundedness_understands_crore(self):
        from evaluation.runner import score_groundedness
        response = {"answer": "Net revenue was 3.06 crore (orders table).", "tool_calls": [
            {"tool": "execute_sql", "args": {}, "result_preview": '{"rows": [{"v": "30552981.27"}]}'}]}
        assert score_groundedness(response, {}) == 1.0

    def test_trust_awareness_only_applies_when_expected(self):
        from evaluation.runner import score_trust_awareness
        assert score_trust_awareness({"answer": "x"}, {"expects_warning": False}) is None
        assert score_trust_awareness({"answer": "gross_revenue is deprecated"}, {"expects_warning": True}) == 1.0
        assert score_trust_awareness({"answer": "It is 43 crore"}, {"expects_warning": True}) == 0.0

    def test_sql_correctness_ignores_table_aliases(self):
        from evaluation.runner import score_sql_correctness
        expected = {"expected_filter": "status = 'completed' AND customers.segment = 'enterprise'"}
        response = {"tool_calls": [{"tool": "execute_sql", "args": {
            "query": "SELECT SUM(o.net_amount) FROM business.orders o JOIN business.customers c ON c.id = o.customer_id "
                     "WHERE o.status = 'completed' AND c.segment = 'enterprise'"}}]}
        assert score_sql_correctness(response, expected) == 1.0

    def test_dataset_is_consistent(self):
        from evaluation.runner import load_dataset
        questions = load_dataset()
        assert len(questions) == 25
        assert len({q["id"] for q in questions}) == 25
        assert sum(1 for q in questions if q["gold_sql"]) >= 15
        for q in questions:
            assert {"gold_sql", "expects_warning", "expected_metric"} <= q.keys()
