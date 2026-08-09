"""FastAPI service.

Streams the workflow to the browser over Server-Sent Events so the UI can show
each desk lighting up as it finishes, rather than spinning for a minute and then
dumping a verdict.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.applications import CustomApplication, build_case, estimate_monthly_payment
from app.config import get_settings
from app.graph import build_graph, initial_state
from app.policies.store import get_policy_store
from app.state import FIXTURE_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

_graph = None
_cases: dict[str, dict[str, Any]] = {}


# --- Spend protection -----------------------------------------------------
# A run is roughly six model calls. Without a ceiling, one link on social media
# is an unbounded invoice. In-process counters are adequate for a single-instance
# demo; a multi-instance deployment would move these to Redis.
_hits: dict[str, deque[float]] = defaultdict(deque)
_daily = {"date": date.today(), "count": 0}


def _check_quota(client_ip: str) -> None:
    s = get_settings()

    if _daily["date"] != date.today():
        _daily.update(date=date.today(), count=0)
    if _daily["count"] >= s.daily_run_cap:
        raise HTTPException(429, "Daily demo limit reached. Try again tomorrow.")

    now = time.time()
    window = _hits[client_ip]
    while window and now - window[0] > 3600:
        window.popleft()
    if len(window) >= s.rate_limit_per_hour:
        raise HTTPException(429, f"Rate limit: {s.rate_limit_per_hour} runs per hour.")

    window.append(now)
    _daily["count"] += 1


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph, _cases
    settings = get_settings()

    problems = settings.validate()
    if problems:
        for p in problems:
            log.error("Configuration problem: %s", p)

    _cases = {
        c["case_id"]: c
        for c in json.loads(settings.test_cases.read_text(encoding="utf-8"))["test_cases"]
    }
    log.info("Loaded %d test cases", len(_cases))

    if not problems:
        # Build the index at start-up, not per request.
        await asyncio.to_thread(get_policy_store)
        _graph = build_graph(mode="parallel")
        log.info("Workflow ready (provider=%s model=%s)", settings.provider, settings.resolved_model)

    yield


app = FastAPI(title="Mortgage Underwriting Agents", version="1.0.0", lifespan=lifespan)


class Overrides(BaseModel):
    """Optional tweaks so a visitor can push a case across a policy boundary."""

    credit_score: int | None = Field(None, ge=300, le=850)
    monthly_income: float | None = Field(None, ge=0, le=1_000_000)
    loan_amount: float | None = Field(None, ge=0, le=10_000_000)


def _apply(case: dict[str, Any], ov: Overrides) -> dict[str, Any]:
    import copy

    c = copy.deepcopy(case)
    if ov.credit_score is not None:
        c["credit_score"] = ov.credit_score
    if ov.monthly_income is not None:
        c.setdefault("employment", {})["monthly_income"] = ov.monthly_income
    if ov.loan_amount is not None:
        c.setdefault("loan", {})["amount"] = ov.loan_amount
    return c


@app.get("/api/health")
async def health() -> dict[str, Any]:
    s = get_settings()
    return {
        "status": "ok" if _graph is not None else "degraded",
        "provider": s.provider,
        "model": s.resolved_model,
        "cases": len(_cases),
        "runs_today": _daily["count"],
        "daily_cap": s.daily_run_cap,
        "problems": s.validate(),
    }


@app.get("/api/cases")
async def list_cases() -> list[dict[str, Any]]:
    """Summaries for the case picker. No PII leaves this endpoint."""
    out = []
    for c in _cases.values():
        emp = c.get("employment", {})
        out.append(
            {
                "case_id": c["case_id"],
                "label": c.get("name", c["case_id"]),
                "credit_score": c.get("credit_score"),
                "monthly_income": emp.get("monthly_income"),
                "employment_type": emp.get("type"),
                "loan_amount": c.get("loan", {}).get("amount"),
                "property_type": c.get("property", {}).get("type"),
                "expected_decision": c.get("expected_decision"),
            }
        )
    return out


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"


async def _run(case: dict[str, Any], thread_id: str) -> AsyncIterator[str]:
    started = time.time()
    yield _sse("start", {"case_id": case["case_id"], "expected": case.get("expected_decision")})

    config = {"configurable": {"thread_id": thread_id}}
    final: dict[str, Any] = {}

    try:
        async for chunk in _graph.astream(initial_state(case), config):
            for node, update in chunk.items():
                if node == "__start__" or not isinstance(update, dict):
                    continue
                final.update({k: v for k, v in update.items() if v is not None})
                yield _sse(
                    "node",
                    {
                        "node": node,
                        "elapsed": round(time.time() - started, 2),
                        "metrics": update.get("metrics"),
                        "analysis": next(
                            (
                                update[k]
                                for k in (
                                    "credit_analysis",
                                    "income_analysis",
                                    "asset_analysis",
                                    "collateral_analysis",
                                    "critic_review",
                                    "decision_memo",
                                )
                                if update.get(k)
                            ),
                            None,
                        ),
                        "policy_violations": update.get("policy_violations"),
                        "bias_flags": update.get("bias_flags"),
                        "reasoning": update.get("reasoning_chain"),
                    },
                )
    except Exception as exc:  # surfaced to the UI rather than dying silently
        log.exception("Run failed")
        yield _sse("error", {"message": str(exc)[:300]})
        return

    decision = final.get("final_decision")
    expected = case.get("expected_decision")
    yield _sse(
        "complete",
        {
            "decision": decision,
            # An invented applicant has no expected outcome, so there is nothing
            # to match against — the UI hides the badge when this is null.
            "matches_expected": (
                None if expected is None else FIXTURE_LABELS.get(decision) == expected
            ),
            "expected": expected,
            "risk_score": final.get("risk_score"),
            "conditions": final.get("conditions", []),
            "credit_memo": final.get("decision_memo"),
            "human_review_required": final.get("human_review_required"),
            "human_review_reasons": final.get("human_review_reasons", []),
            "policy_violations": final.get("policy_violations", []),
            "bias_flags": final.get("bias_flags", []),
            "metrics": final.get("metrics", {}),
            "elapsed": round(time.time() - started, 2),
        },
    )


@app.post("/api/run/{case_id}")
async def run_case(case_id: str, request: Request, overrides: Overrides | None = None):
    """Execute the workflow and stream progress as SSE."""
    if _graph is None:
        raise HTTPException(503, "Service not configured — check /api/health")
    if case_id not in _cases:
        raise HTTPException(404, f"Unknown case '{case_id}'")

    _check_quota(request.client.host if request.client else "unknown")

    case = _apply(_cases[case_id], overrides or Overrides())
    thread_id = f"{case_id}-{int(time.time() * 1000)}"

    return StreamingResponse(
        _run(case, thread_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/estimate-payment")
async def estimate_payment(payload: CustomApplication) -> dict[str, Any]:
    """Derived monthly PITI, so the form can show it before a run is spent."""
    return {
        "monthly_payment": estimate_monthly_payment(payload.loan_amount, payload.property_value),
        "assumed_rate": "6.5% / 30yr + 1.5% escrow",
    }


@app.post("/api/run-custom")
async def run_custom(payload: CustomApplication, request: Request):
    """Run the workflow against an applicant the visitor invented."""
    if _graph is None:
        raise HTTPException(503, "Service not configured — check /api/health")

    _check_quota(request.client.host if request.client else "unknown")

    case_id = f"CUSTOM-{int(time.time() * 1000)}"
    case = build_case(payload, case_id)

    return StreamingResponse(
        _run(case, case_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")
