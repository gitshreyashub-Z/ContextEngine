# ContextEngine

**A structured context layer that makes AI agents more reliable over enterprise data.**

ContextEngine is an engineering experiment testing whether giving an AI agent structured business context — definitions, trust signals, metadata, and lineage — improves its ability to correctly answer questions about enterprise databases, compared to a baseline agent with only schema access.

---

## Problem

Enterprise databases contain data, but not business meaning. A schema tells an LLM that `orders.total_amount` and `orders.net_amount` exist. It does not tell the LLM:

- Which one the company considers "revenue"
- Which is certified vs deprecated
- Who owns the metric
- What filters must be applied (e.g., only completed orders)
- Why one should be preferred over another

**Without this context, AI agents over enterprise data fail silently** — they produce confident answers using wrong metrics, deprecated columns, or incorrect filters. The failure mode is not a crash; it's a plausible-sounding wrong answer.

How big is the error? On the seeded dataset (numbers vary slightly with the seed date):

| Question | Naive SQL | Certified definition |
|---|---|---|
| Revenue in July 2026 | `SUM(total_amount)` → **₹4.11 Cr** | `SUM(net_amount) WHERE status='completed'` → **₹3.06 Cr** (naive is 35% too high) |
| Active customers | `WHERE is_active` → **805** | `is_active AND last order within 90 days` → **274** |

---

## Hypothesis

> Does giving an AI agent a structured context layer containing business definitions, metadata, trust signals, and lineage make it measurably more reliable at selecting correct metrics, generating correct SQL, and producing grounded answers?

---

## Architecture

```
                              User question
                                    │
          ┌─────────────────────────┼──────────────────────────┐
          ▼                         ▼                          ▼
   ┌─────────────┐        ┌──────────────────┐       ┌──────────────────┐
   │  Baseline   │        │ ContextEngine    │       │ ContextEngine    │
   │ (ReAct loop)│        │ ReAct (same loop)│       │ pipeline         │
   │ execute_sql │        │ all 7 tools      │       │ fixed tool order │
   └──────┬──────┘        └────────┬─────────┘       └────────┬─────────┘
          │                        │                          │
          │              ┌─────────▼──────────────────────────▼───┐
          │              │  Context tools (ContextTools)          │ ◄── also served over
          │              │  search_assets · get_definition ·      │     MCP (stdio) to any
          │              │  get_trust_signal · get_schema ·       │     MCP client
          │              │  get_lineage · list_metrics            │
          │              └─────────┬──────────────────────────────┘
          │                        │
          ▼                        ▼
   ┌────────────────────────────────────────────┐
   │ execute_sql: validator → read-only DB role │
   └──────────────────────┬─────────────────────┘
                          ▼
        PostgreSQL 16:  business.*  (data)   context.*  (definitions, trust,
                                                         schema info, lineage)
```

**Key principle:** the context agents do not blindly query the database. They first consult the context layer to understand business meaning, then compute the metric from its certified definition.

---

## The three agents

| Agent | How it works | Role in the experiment |
|---|---|---|
| `baseline` | LangGraph ReAct loop. The LLM sees table/column names in its system prompt and writes SQL with an `execute_sql` tool. | Control group |
| `contexthub_react` | **Same** LangGraph loop, **same** LLM, **same** schema prompt — plus the six context tools and instructions on *how* to use them. The prompt never states the answers (no "use net_revenue", no 90-day rule). | Clean ablation: only context access differs from baseline |
| `contexthub` | Deterministic pipeline: Python picks the metric (trust-ranked) and builds SQL from its definition; the LLM is called once to phrase the answer. | Upper bound for "context + reliable control" |

### ContextEngine pipeline workflow

```
1. User: "What was revenue by product category in July 2026?"
2. search_assets("revenue")              topic from keywords + synonyms (sales/turnover → revenue)
3. Pick metric                           explicitly named metric first, otherwise rank candidates by
   → net_revenue                         trust signal (certified > experimental > stale > deprecated);
                                         reject candidates unrelated to the question
4. get_definition("net_revenue")         SUM(orders.net_amount) WHERE status = 'completed'
5. get_trust_signal("net_revenue")       certified, approved by Finance Team / CFO
6. (get_schema / get_lineage / list_metrics when the question is about columns,
    relationships or which metrics are deprecated)
7. Build SQL from the definition + what the question asks for, then execute_sql:
     SELECT p.category, SUM(net_amount) FROM business.orders o
     JOIN business.products p ON p.id = o.product_id
     WHERE status = 'completed' AND order_date >= '2026-07-01' AND order_date < '2026-08-01'
     GROUP BY p.category ORDER BY net_revenue DESC
8. LLM phrases a short answer citing metric, source and certification status
   (without an LLM configured, a plain-text answer is returned instead)
```

The SQL builder supports date ranges ("July 2026", "last month", "in 2025"), multi-month comparisons ("June and July 2026"), monthly trends, customer-segment filters and group-bys, product-category group-bys and top-N products.

**Why keep a pipeline?** With small free-tier models (7–8B parameters) an LLM-driven agent often skips the definition lookup or stops before running SQL. The pipeline makes the context layer's effect reliable and unit-testable; the ReAct variant shows how much of that an LLM achieves on its own.

---

## Context Layer

The context layer is stored as structured PostgreSQL tables (not a giant prompt or vector store):

| Component | Purpose | Example |
|-----------|---------|---------|
| **Metadata** | Asset ownership, type, description | `net_revenue` owned by Finance Team |
| **Business Definitions** | Exact metric calculations with filters | `net_revenue = SUM(net_amount) WHERE status='completed'` |
| **Trust Signals** | Certification status | `net_revenue: certified`, `gross_revenue: deprecated` |
| **Schema Info** | Enriched column descriptions with business context | `total_amount: WARNING - not revenue` |
| **Lineage** | Relationships between assets | `orders.net_amount → net_revenue metric` |

### Why Structured Storage?

Each piece of context is a queryable record with typed fields — not embedded in a prompt or retrieved via semantic search. This means:

- The agent can look up exactly what it needs (no irrelevant context flooding)
- Trust signals are first-class data, not footnotes
- Context is maintainable: update one definition, and the agent immediately uses the new version

---

## MCP Server

The same tools are exposed via the [Model Context Protocol](https://modelcontextprotocol.io/), so any MCP client (Claude Desktop, IDEs, other agent frameworks) can use the context layer:

```bash
python -m app.mcp.server     # stdio transport
```

| Tool | Purpose |
|------|---------|
| `search_assets(query)` | Find relevant tables, columns, or metrics |
| `get_definition(metric_name)` | Get certified business definition with calculation |
| `get_trust_signal(asset_name)` | Check if asset is certified/deprecated/experimental |
| `get_schema(table_name)` | Get columns with business context descriptions |
| `get_lineage(asset_name)` | Understand upstream/downstream relationships |
| `execute_sql(query)` | Run read-only SQL (SELECT only, validated) |
| `list_metrics()` | List all available metric definitions |

The agents call the shared `ContextTools` implementation in-process (lower latency, simpler deployment); the MCP server wraps the same class, and `tests/test_mcp_server.py` drives it end to end with the official MCP client over stdio.

---

## Evaluation

All three agents answer the same 25 questions (`evaluation/dataset.json`). 15 numeric questions have **gold SQL**; 5 questions require a **warning** (deprecated, draft, experimental or unreliable asset).

### Scoring dimensions

| Dimension | What it measures | Applies to |
|-----------|-----------------|---|
| **execution_accuracy** | The agent's SQL, re-executed, returns the gold result (order/column-insensitive, 0.5% tolerance) | 15 questions with gold SQL |
| **metric_selection** | Answer names the expected metric, or the SQL computes it (right column + filter) | questions with an expected metric |
| **source_selection** | Right table and column | questions with an expected source |
| **sql_correctness** | Required filter conditions present in the SQL | questions with an expected filter |
| **groundedness** | Answer states a number the SQL actually returned, names a source, used tools | all |
| **trust_awareness** | Answer warns about the deprecated/draft/experimental/unreliable asset | 5 questions |
| **tool_success** | Tool calls without errors | answers that used tools |

Dimensions that don't apply to a question are excluded from its averages. Scorers are judged from the answer and the SQL only, so the baseline can earn full marks by writing the right SQL — no points are given just for calling a context tool.

### Running the evaluation

```bash
python -m evaluation.runner                         # all three agents, 25 questions
python -m evaluation.runner --agent both            # baseline + pipeline
python -m evaluation.runner --agent contexthub_react --questions 5
python -m evaluation.runner --runs 3                # repeat; report mean +/- std
python -m evaluation.report                         # markdown report of the latest run
```

### Results

Measured so far (seeded database, **no LLM configured**, so the pipeline used its plain-text answer):

| Dimension | ContextEngine (pipeline) |
|---|---|
| execution_accuracy | 86.7% (13/15) |
| metric_selection | 100% |
| sql_correctness | 80.8% |
| trust_awareness | 100% |
| overall | 95.0% |

The two execution misses are "How many orders were cancelled?" and "What payment methods do customers use most?": no certified metric covers them, so the pipeline says so instead of guessing. The baseline and ReAct agents need an LLM and have **not been run yet** — results depend on the model, so run the evaluation with your provider and commit the report rather than quoting expected numbers.

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **One ReAct graph for baseline and ContextEngine ReAct** | Same LLM, loop and schema prompt; only the tools differ → the comparison isolates the context layer |
| **Deterministic pipeline as a third agent** | Small free models are unreliable at multi-step tool use; the pipeline shows the effect of context with reliable control and is unit-testable |
| **PostgreSQL for context** | Structured, queryable context. Not a vector store — metrics have known fields, not fuzzy semantics |
| **In-process tools + MCP server** | Agents call the shared `ContextTools` class directly (simplicity, latency); the MCP server exposes the same tools to any MCP client |
| **Free-tier LLMs** | Gemini (free tier), Groq (free tier), Ollama (local). No paid API required |
| **Read-only SQL** | SQL validation + read-only DB role + statement timeout. Agents cannot mutate data |
| **Single PostgreSQL instance** | Business data and context in one DB (different schemas). Minimizes infrastructure |
| **Execution accuracy as the primary score** | Keyword checks can be gamed; comparing query results against gold SQL cannot |
| **No vector search** | Context is structured with known fields. Exact lookup > semantic search for definitions |

---

## Running Locally

### Prerequisites

- Python 3.11+
- Docker (for PostgreSQL)
- For the LLM agents: a Gemini API key (free), a Groq API key (free), or Ollama installed

### Setup

```bash
# 1. Start PostgreSQL
docker compose up -d

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your LLM provider / API key (see below)

# 4. Seed database (business data + context layer)
python seed.py
# Seeding skips if data exists. To re-seed from scratch (needed after model changes):
#   docker compose down -v && docker compose up -d && python seed.py

# 5. Run the CLI
python cli.py
```

### LLM Configuration

Choose one:

**Google Gemini:**
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-key-from-https://aistudio.google.com/apikey
```

**Groq:**
```env
LLM_PROVIDER=groq
GROQ_API_KEY=your-key-from-https://console.groq.com/keys
```

**Ollama (fully local, no API key):**
```bash
ollama pull qwen2.5:7b
```
```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:7b
```

**OpenRouter:** `LLM_PROVIDER=openrouter` and `OPENROUTER_API_KEY=...`

### Usage

```bash
# Interactive CLI (pipeline agent)
python cli.py
python cli.py --agent contexthub_react
python cli.py --baseline

# Compare all three agents side-by-side
python cli.py --compare -q "What was our revenue in July 2026?"

# API server  (docs at http://localhost:8000/docs)
uvicorn app.main:app --reload
# POST /ask {"question": "...", "agent": "contexthub" | "contexthub_react" | "baseline"}
# GET /agents, /metrics, /schema/{table}, /health

# Web UI (needs the API running)
python frontend/serve.py      # http://localhost:3000

# MCP server
python -m app.mcp.server

# Tests (database tests are skipped automatically if Postgres isn't running)
pytest tests/ -v
```

---

## Project Structure

```
contexthub/
├── app/
│   ├── config.py              # Pydantic settings (LLM provider, DB URL)
│   ├── main.py                # FastAPI app (/ask, /agents, /metrics, /schema, /health)
│   ├── agent/
│   │   ├── react.py           # Shared LangGraph ReAct loop
│   │   ├── baseline.py        # Baseline: ReAct + execute_sql only
│   │   ├── contexthub_react.py# ContextEngine ReAct: ReAct + all 7 tools
│   │   ├── contexthub.py      # ContextEngine pipeline (deterministic)
│   │   ├── tools.py           # LangChain wrappers around ContextTools
│   │   ├── registry.py        # Builds agents by name (API, CLI, evaluation)
│   │   ├── llm.py             # LLM factory (Gemini/Groq/Ollama/OpenRouter)
│   │   └── prompts.py         # System prompts
│   ├── context/
│   │   ├── models.py          # SQLAlchemy: assets, definitions, trust, schema info, lineage
│   │   ├── seed.py            # Seeds business definitions and trust signals
│   │   └── service.py         # Context retrieval logic
│   ├── db/
│   │   ├── connection.py      # Cached engines, read-only role, statement timeout
│   │   ├── models.py          # Business data: customers, orders, products, payments
│   │   └── seed.py            # Generates ~11K records of realistic data
│   └── mcp/
│       ├── server.py          # MCP server (stdio transport)
│       └── tools.py           # Tool implementations + SQL validation
├── evaluation/
│   ├── dataset.json           # 25 questions: expected metric/source/filter, gold SQL, warnings
│   ├── runner.py              # Runs agents, scores 7 dimensions, repeated runs
│   └── report.py              # Markdown comparison report
├── tests/                     # pytest; `db`-marked tests need the seeded database
│   ├── test_agent.py          # Pipeline logic, ReAct loop (fake LLM), scorers
│   ├── test_api.py            # FastAPI endpoints, concurrent requests
│   ├── test_context.py        # Context service
│   ├── test_mcp.py            # SQL safety + tools
│   └── test_mcp_server.py     # MCP server over stdio with the official client
├── frontend/                  # Single-page UI (ask, compare, metrics, schema)
├── cli.py                     # Rich interactive CLI with tool traces
├── seed.py                    # Unified seed script
├── docker-compose.yml         # PostgreSQL 16
├── requirements.txt           # Dependencies (major versions bounded)
└── .env.example               # Configuration template
```

---

## Security

- **Read-only SQL execution**: only SELECT/WITH queries pass validation, and they run as a database role that only has SELECT privileges.
- **Validator**: rejects DDL/DML keywords, stacked statements, comments, `SELECT ... INTO`, `COPY` and dangerous functions (`pg_sleep`, `pg_read_file`, ...). The database role is the real guarantee; the validator gives early, readable errors.
- **Resource limits**: 10-second statement timeout; at most 100 rows returned.
- **No secrets in code**: API keys are loaded from environment variables (`.env` is git-ignored).
- **CORS**: credentials disabled; restrict `allow_origins` before deploying beyond localhost.

---

## Limitations

- **Small benchmark**: 25 questions on 4 tables of synthetic data. Real enterprise environments have hundreds of tables with more complex ambiguities.
- **LLM-dependent results**: baseline and ReAct scores depend on the model; small free models are noisy (use `--runs`).
- **Pipeline coverage**: the SQL builder handles the patterns listed above; arbitrary questions (e.g. counts by status, payment methods) fall outside its templates — that's what the ReAct agent is for.
- **Keyword topic matching in the pipeline**: a fixed synonym list; unseen phrasings may not match a metric.
- **Heuristic secondary scores**: metric/source/filter/groundedness/trust scores look for keywords; execution accuracy is the robust measure.
- **No multi-turn conversation**: each question is independent.
- **Context is manually curated**: in production it would come from a data catalog or dbt.
- **No versioning**: context definitions don't have version history or approval workflows.

---

## Future Work

- **Data catalog integration**: replace the manual context layer with live data from a data catalog API
- **Automated context ingestion**: parse dbt schema files or catalog exports into the context layer
- **Hybrid retrieval**: semantic search over definitions for synonyms, filtered by trust level
- **Multi-turn context**: LangGraph checkpointer for follow-up questions
- **Context freshness alerts**: warn when context was last validated >30 days ago
- **Model sweep**: run the evaluation across several LLMs and compare

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Agent framework | LangGraph | Explicit state machine for the ReAct loop; shared by both LLM agents |
| LLM | Gemini / Groq / Ollama / OpenRouter | Free-tier or local; swap with one env var |
| Tool protocol | MCP (Model Context Protocol) | Standardized agent-tool interface |
| Database | PostgreSQL 16 | Single DB for data + context |
| ORM | SQLAlchemy 2.x | Typed models, parameterized context queries |
| API | FastAPI | Pydantic validation, auto-docs |
| CLI | Rich + Click | Readable tool traces for demos |
| Testing | pytest | Unit tests (fake LLM) + DB integration tests |
| Config | pydantic-settings | Validated env config |

---

*Built during a hackathon as an engineering experiment in AI agent reliability.*
