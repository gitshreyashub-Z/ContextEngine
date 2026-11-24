"""Tests for the FastAPI app.

/health, /agents and request validation need nothing else; /metrics, /schema and
/ask with the pipeline agent need the seeded database (no LLM: the pipeline falls
back to a plain-text answer when no LLM is configured).
"""

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json()["status"] == "healthy"


def test_agents_lists_all_three():
    assert set(client.get("/agents").json()["agents"]) == {"baseline", "contexthub", "contexthub_react"}


def test_unknown_agent_is_400():
    response = client.post("/ask", json={"question": "x", "agent": "nope"})
    assert response.status_code == 400
    assert "contexthub_react" in response.json()["detail"]


def test_missing_question_is_422():
    assert client.post("/ask", json={"agent": "contexthub"}).status_code == 422


@pytest.mark.db
def test_metrics():
    assert client.get("/metrics").json()["count"] >= 9


@pytest.mark.db
def test_schema_found_and_not_found():
    assert client.get("/schema/orders").status_code == 200
    assert client.get("/schema/nope").status_code == 404


@pytest.mark.db
@pytest.mark.parametrize("question", ["How many active customers do we have?", "What is our refund rate?"])
def test_ask_pipeline_returns_sources(question):
    body = client.post("/ask", json={"question": question, "agent": "contexthub"}).json()
    assert body["success"] is True
    assert any(s.get("type") == "trust_signal" for s in body["sources"])
    assert any(c["tool"] == "execute_sql" for c in body["tool_calls"])


@pytest.mark.db
def test_concurrent_requests():
    """Endpoints run in a threadpool; the context service must be safe to use concurrently."""
    questions = ["What is our average order value?", "What is our refund rate?",
                 "How many active customers do we have?", "What is our order completion rate?"] * 3
    with ThreadPoolExecutor(max_workers=6) as pool:
        bodies = list(pool.map(
            lambda q: client.post("/ask", json={"question": q, "agent": "contexthub"}).json(), questions))
    assert all(b["success"] for b in bodies)
