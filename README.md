# Mortgage Underwriting Agents

A six-agent LangGraph workflow that underwrites a residential mortgage application against a
14-page policy manual, and a live web UI that shows the file moving desk to desk as it happens.

**[Live demo](#)** · **[Architecture](docs/architecture.md)**

> Built on a Johns Hopkins Agentic AI course template, then rebuilt as a deployable service.
> The interesting work was not wiring the agents — it was finding out that the original
> silently produced wrong answers, and building the tests that prove it doesn't any more.

---

## What it does

An application arrives as JSON. Before any model is called, the service redacts the PII,
computes every ratio in Python, and runs a deterministic policy gate. Then four specialist
agents analyse it concurrently — credit, income, assets, collateral — each retrieving only
the policy sections relevant to its own remit. A QA critic reads all four looking for
contradictions. A senior underwriter agent returns a typed decision, a risk score, a list
of conditions, and an audit-ready credit memo.

Every figure the agents reason about is settled before they see it. Their job is judgement,
never arithmetic.

```
                        ┌─ credit ─┐
 intake ─ redact PII ───┼─ income ─┼─── QA critic ─── underwriter ─── decision
 compute metrics        ├─ assets ─┤                                  + risk score
 policy gate            └─collateral┘                                 + conditions
                         (concurrent)                                 + credit memo
```

## Why it isn't just another LangGraph demo

The template it grew from ran end to end and printed a success message for every cell.
It was also wrong in five ways that no test would have caught, because there were no tests.
Each is now a named regression test.

| Defect | Effect | Fix |
|---|---|---|
| `sum(debts.values())` summed the `total_monthly_debt` roll-up alongside the line items | Every applicant's DTI roughly doubled — 86% instead of 56% — pushing approvable files toward denial | `monthly_debt_total()` excludes roll-up keys · [test](tests/test_calculators.py) |
| Bias detector used substring matching, and `"age"` is inside `"mortgage"` | Fired on 100% of cases, so every file escalated to a human and the signal meant nothing | Word-boundary matching with an allowlist for `age of the property` · [test](tests/test_compliance.py) |
| Decision parsed out of prose: `if "APPROVED" in content and "CONDITIONAL" not in content` | A memo containing "the application was not denied" classified as DENIED | Pydantic `with_structured_output` — the parsing step no longer exists · [decision.py](app/agents/decision.py) |
| PII redaction was a blocklist and forgot `email` | The fixture's email is `sarah.johnson@email.com` — the name it had just redacted | Allowlist that fails closed, plus a deep copy so the source record isn't mutated |
| Large-deposit threshold used only 25% of income | Policy 3.3 says "$1,000 **or** 25%, whichever is less" — deposits between $1,000 and $3,125 were invisible | `min(1000, income * 0.25)` |
| Reserves measured gross liquidity | Overstated every borrower's cushion by the entire down payment — 58 months where the real figure was 23 | Post-closing liquidity per policy 3.1 |
| Collateral agent quoted LTV on appraised value, decision agent on the lesser of price/appraisal | Two agents citing different LTVs for the same file | One `collateral_basis` in `metrics`, consumed by both |

### The one that mattered most

Writing the end-to-end tests surfaced something the unit tests couldn't: **once DTI was computed
correctly, all three fixtures breached the 50% policy ceiling and every case denied.**

The fixtures' `dti_ratio` field recorded debts *excluding* the proposed housing payment, and the
`expected_decision` labels had been assigned against that number. So the ground truth and the
policy manual disagreed — and the ground truth was wrong. Fixing the engine made the test set
fail, which is the correct and uncomfortable outcome.

The obligations in [`data/mortgage_test_cases.json`](data/mortgage_test_cases.json) were rebuilt
so that line items, roll-ups and stated ratios all agree, and the intended narrative — strong /
borderline / weak — survives a correct back-end calculation. A test now asserts that the stated
and computed ratios match, so the two can never drift apart again.

The lesson worth carrying: **an agent system fails silently.** Nothing crashes, every log line is
green, and the answer is confidently wrong. Worse, the fixtures you measure against are the one
thing nobody thinks to check. The tests are the product.

## Design decisions

**Arithmetic in code, judgement in the model.** Every ratio is computed by a tool and injected
into the prompt marked `PRE-CALCULATED — AUTHORITATIVE`. The model never gets the opportunity
to do maths it is bad at. Thresholds live in [`calculators.py`](app/tools/calculators.py) with
the policy section cited inline, so a rule change is a diff.

**Hard rules are gates, not prompts.** A credit score below 620 or repairs above the escrow cap
is a bright line. [`policy_gate()`](app/compliance.py) decides those in Python before an LLM is
consulted, and a violation forces a denial regardless of what the model concluded. Bright lines
should not be arguable.

**Four specialists, one spec.** The original carried four near-identical 80-line functions
differing only in retrieval query, tools, framework and output key. That is data, so
[`specialists.py`](app/agents/specialists.py) declares four `SpecialistSpec` values and one
runner executes them. A fifth desk — title, flood, HOA — is a spec, not a module.

**Parallel by default, and the reducers are why.** The four specialists share no data, so
serialising them buys only latency. That is safe *only* because the accumulating state fields
carry `operator.add` reducers; with overwrite semantics three of four concurrent writes would
be lost. [`test_graph.py`](tests/test_graph.py) asserts the annotations so nobody removes one
by accident. A `sequential` topology is retained for tracing.

**Provider-agnostic.** `LLM_PROVIDER=openai|anthropic` switches the chat model without touching
agent code. Embeddings stay on OpenAI — Anthropic doesn't serve them.

## Running it

```bash
git clone https://github.com/<you>/mortgage-underwriting-agents
cd mortgage-underwriting-agents
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # add your OPENAI_API_KEY
uvicorn app.main:app --reload
```

Open http://localhost:8000.

```bash
pytest -q          # 67 tests, zero API calls, ~0.5s
```

The suite makes no network calls at all. The end-to-end tests in
[`test_workflow.py`](tests/test_workflow.py) run the **real** graph — real routing, real merge
semantics, real policy gate, real redaction — against a stubbed model and retriever. That keeps
it free and fast, and it separates orchestration bugs from prompt bugs. When something
misbehaves you want to know which of the two it was before you start reading prompts.

## Deploying to Railway

1. Push to GitHub.
2. Railway → **New Project** → **Deploy from GitHub repo**.
3. Add variables: `OPENAI_API_KEY`, and optionally `RATE_LIMIT_PER_HOUR`, `DAILY_RUN_CAP`.
4. `railway.json` supplies the start command and points the health check at `/api/health`.

The policy index is built once at container start (~30s, ~45 embedding calls), not per request.
Expect a slow first boot and fast requests after.

**A public demo spends real money.** One run is roughly six model calls. `RATE_LIMIT_PER_HOUR`
(default 5, per IP) and `DAILY_RUN_CAP` (default 200) bound it. Set a hard spend limit in your
provider dashboard as well — application-level limits protect against traffic, not against bugs.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Status, provider, model, runs used today |
| `GET` | `/api/cases` | Case summaries for the picker (no PII) |
| `POST` | `/api/run/{case_id}` | Runs the workflow, streams SSE (`start`, `node`, `complete`, `error`) |

`POST` accepts optional overrides — `credit_score`, `monthly_income`, `loan_amount` — so a
visitor can push a file across a policy boundary and watch the decision change.

## Layout

```
app/
├── config.py         env-driven settings, validated at start-up
├── state.py          the shared "folder"; reducers decide merge semantics
├── llm.py            provider abstraction
├── compliance.py     PII allowlist, Fair Lending detector, hard policy gate
├── tools/
│   ├── calculators.py  policy thresholds + deterministic maths
│   └── metrics.py      one canonical numeric picture per application
├── policies/store.py RAG over the policy PDF
├── agents/
│   ├── base.py         SpecialistSpec + the single runner
│   ├── specialists.py  the four desks, as data
│   ├── critic.py       cross-checks the four memos
│   └── decision.py     typed decision via structured output
├── graph.py          wiring, both topologies
└── main.py           FastAPI + SSE
```

## Known limitations

- Rate limiting is in-process; a multi-instance deployment needs Redis.
- The vector index is in-memory and rebuilds on every cold start.
- Three fixtures prove the pipeline runs. They cannot measure fairness or accuracy — that
  needs hundreds of historical decisions with known loan performance.
- Human-in-the-loop escalation is flagged but not enforced; a production build would use
  `interrupt_before` with a durable checkpointer so the graph genuinely pauses.

## Licence

MIT. The policy manual and test fixtures are synthetic educational material, not real
underwriting guidance or real applicant data.
