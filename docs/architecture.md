# Architecture

## System Overview

ContextEngine is composed of 5 layers:

```
┌───────────────────────────────────────────────────────────────────────┐
│                           Interface Layer                             │
│     CLI (Rich) │ FastAPI │ Web UI │ Evaluation Runner │ MCP clients   │
├───────────────────────────────────────────────────────────────────────┤
│                             Agent Layer                               │
│  BaselineAgent ─┐                                                     │
│                 ├─ shared LangGraph ReAct loop (app/agent/react.py)   │
│  ContextEngine  ┘                                                     │
│  ReActAgent                                                           │
│  ContextEngineAgent (deterministic pipeline)                          │
│  registry.get_agent(name) builds and caches all three                 │
├───────────────────────────────────────────────────────────────────────┤
│                          Tool Layer                                   │
│  ContextTools (app/mcp/tools.py) ◄─ LangChain tools (app/agent/tools) │
│                                  ◄─ MCP server (app/mcp/server.py)    │
│  search_assets │ get_definition │ get_trust_signal │ get_schema       │
│  get_lineage │ execute_sql │ list_metrics                             │
├───────────────────────────────────────────────────────────────────────┤
│                           Context Layer                               │
│  ContextService → Assets, Definitions, Trust, Schema, Lineage         │
│  (one short-lived read-only session per lookup)                       │
├───────────────────────────────────────────────────────────────────────┤
│                            Data Layer                                 │
│  PostgreSQL: business.* (data) + context.* (metadata)                 │
│  cached engines; agent SQL runs as a read-only role with a timeout    │
└───────────────────────────────────────────────────────────────────────┘
```

## Agents

| Agent | Control | Tools | Prompt |
|---|---|---|---|
| `baseline` | LLM (ReAct loop) | `execute_sql` | `BASELINE_SYSTEM_PROMPT`: table/column list, rules |
| `contexthub_react` | LLM (same ReAct loop) | all 7 | baseline prompt **+** how to use the context tools (no answers) |
| `contexthub` | Python pipeline | called in fixed order | LLM only formats the final answer |

`baseline` vs `contexthub_react` is the clean ablation: same graph, same LLM, same schema
prompt — only context access differs. `contexthub` shows the effect of context with
reliable, deterministic control.

## Data Flow

### ReAct agents (baseline and ContextEngine ReAct)

```
User Question
    │
    ▼
[Entry] ─── messages = [SystemMessage(prompt), HumanMessage(question)]
    │
    ▼
[agent node] ─── llm.bind_tools(tools).invoke(messages) → AIMessage appended
    │
    ├── has tool_calls ──▶ [tools node] ─── run each tool → ToolMessage appended,
    │                            │           call logged; unknown tool / exception
    │                            │           → JSON error the LLM can read
    │                            └──▶ [agent node] (loop)
    │
    └── no tool_calls ──▶ [END] ─── final answer = last AIMessage without tool calls

Guard: recursion_limit = 20 steps → reported as a failed run.
Sources (definitions, trust signals) are extracted from the logged tool results.
```

Each node returns the full message list (`state["messages"] + new`), because
`AgentState.messages` has no reducer: returning only the new messages would replace
the conversation and the LLM would lose the question and system prompt.

### ContextEngine pipeline

```
User Question
    │
    ├─ 1. topic = keyword/synonym match      ("sales" → revenue)
    ├─ 2. search_assets(topic)
    ├─ 3. pick metric
    │      explicit mention / alias (gross revenue, MRR, AOV)?  → use it
    │      else rank metric candidates by trust (certified > experimental > stale > deprecated),
    │      tie-break by word overlap; reject candidates sharing no words with the question
    ├─ 4. get_definition(metric)  → source table/column, filter, owner, notes
    ├─ 5. get_trust_signal(metric)
    ├─ 6. supporting context when asked for:
    │      columns/tables → get_schema │ relationships → get_lineage │ governance → list_metrics
    ├─ 7. data question and metric not draft?
    │      build SQL from the definition + question:
    │        date range / last month / multi-month (GROUP BY month) / trend
    │        segment filter or GROUP BY segment (JOIN customers)
    │        GROUP BY category or top-N products (JOIN products)
    │        aggregate: SUM │ COUNT │ AVG │ rate (100 * COUNT FILTER / COUNT)
    │      → execute_sql
    └─ 8. LLM formats a 3–5 sentence answer from the collected context
           (no LLM / LLM error → plain-text answer with result, status, warnings, notes)
```

### API request

```
POST /ask {"question", "agent"}
   → AskRequest validation (422 on bad body), unknown agent → 400
   → registry.get_agent(agent).ask(question)        (sync endpoint → FastAPI threadpool)
   → AskResponse {answer, sources, tool_calls (previews truncated), latency, success}
```

## Database Schema

### Business Schema (business.*)

Enum columns store the lowercase **values** (`'completed'`), via `pg_enum()` in
`app/db/connection.py`, so SQL written from the definitions works.

```sql
business.customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    email VARCHAR(255) UNIQUE,
    company VARCHAR(255),
    segment ENUM(enterprise, mid_market, smb, startup),
    is_active BOOLEAN,
    created_at TIMESTAMP,
    last_order_date DATE
)

business.orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER → customers.id,
    product_id INTEGER → products.id,
    order_date DATE,                -- 2024-01-01 .. seed date
    status ENUM(pending, confirmed, completed, cancelled, refunded),
    total_amount NUMERIC(12,2),     -- Raw price (NOT revenue)
    discount_amount NUMERIC(12,2),
    tax_amount NUMERIC(12,2),
    refund_amount NUMERIC(12,2),
    net_amount NUMERIC(12,2)        -- CERTIFIED revenue column
)

business.products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    category VARCHAR(100),
    base_price NUMERIC(12,2),
    is_active BOOLEAN,
    created_at TIMESTAMP
)

business.payments (
    id SERIAL PRIMARY KEY,
    order_id INTEGER → orders.id,
    amount NUMERIC(12,2),
    method ENUM(credit_card, debit_card, upi, net_banking, wallet),
    status ENUM(pending, completed, failed, refunded),
    paid_at TIMESTAMP,
    created_at TIMESTAMP
)
```

### Context Schema (context.*)

```sql
context.assets (id, name, asset_type, description, owner, schema_name, table_name, column_name)
context.business_definitions (id, metric_name, display_name, definition, calculation, source_table, source_column, filter_condition, owner, status, examples, notes)
context.trust_signals (id, asset_name, trust_level, reason, certified_by, last_validated)
context.schema_info (id, table_name, column_name, data_type, is_nullable, is_primary_key, description, business_context)
context.lineage (id, source_asset, target_asset, relationship_type, description)
```

## Security Model

```
┌─────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
│   Agent     │────▶│  SQL Validator       │────▶│ Read-only role         │
│ (generates  │     │ • starts SELECT/WITH │     │ contexthub_readonly    │
│  SQL)       │     │ • no DDL/DML words   │     │ • SELECT only          │
│             │     │ • no ';' + more SQL  │     │ • statement_timeout 10s│
│             │     │ • no -- or /* */     │     └────────────────────────┘
│             │     │ • no INTO/COPY/      │               │
│             │     │   pg_sleep/pg_read_* │               ▼
│             │     │ • LIMIT 100 appended │      at most 100 rows fetched
└─────────────┘     └──────────────────────┘
                           │
                           └── rejected → JSON error returned to the agent
```

The validator is a first line of defence that gives readable errors; the database role
is the guarantee.

## LLM Provider Architecture

```
┌────────────────────────────────────────┐
│           get_llm() Factory            │
├────────────────────────────────────────┤
│  LLM_PROVIDER=gemini     → ChatGoogleGenerativeAI   (GEMINI_API_KEY)  │
│  LLM_PROVIDER=groq       → ChatGroq                 (GROQ_API_KEY)    │
│  LLM_PROVIDER=ollama     → ChatOllama               (local, no key)   │
│  LLM_PROVIDER=openrouter → ChatOpenAI (OpenRouter)  (OPENROUTER_API_KEY) │
└────────────────────────────────────────┘
```

## Evaluation Pipeline

```
evaluation/dataset.json (25 questions; 15 with gold_sql; 5 expect a warning)
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│ evaluation/runner.py                                     │
│ for each agent (baseline, contexthub_react, contexthub): │
│   for each run (--runs N):                               │
│     for each question:                                   │
│       response = agent.ask(question)                     │
│       scores = 7 dimensions (None where not applicable)  │
│         execution_accuracy: re-run agent SQL + gold SQL, │
│                             compare result sets          │
│       sleep(--delay)   (free-tier rate limits)           │
│   mean +/- std across runs                               │
└───────────────┬──────────────────────────────────────────┘
                ▼
evaluation/results/eval_YYYYMMDD_HHMMSS.json   (git-ignored)
                ▼
evaluation/report.py → results/report.md (table per agent, deltas vs baseline,
                                           per-question scores, findings)
```

## Why These Choices?

| Component | Alternative Considered | Why Current Choice |
|-----------|----------------------|-------------------|
| LangGraph | Custom while loop | Explicit state machine; one graph shared by both LLM agents |
| Shared ReAct graph | Separate agent implementations | Fair ablation: only tools/instructions differ |
| Deterministic pipeline | LLM-only agent | Reliable with small models; unit-testable |
| PostgreSQL context | Vector store | Context is structured with known fields; exact lookup > fuzzy search |
| MCP tools | Direct function calls only | Standardized protocol; any MCP client can reuse the context layer |
| Execution accuracy | Keyword scoring only | Checks the actual numbers; can't be gamed by wording |
| Single DB | Separate context DB | Simplicity; one docker compose service |
| Free LLMs | Paid frontier models | Accessibility; no cost barrier to run |
| 25 questions | 100+ questions | Manageable eval time with free-tier rate limits |
