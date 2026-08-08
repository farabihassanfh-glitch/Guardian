"""Workflow wiring.

LangGraph does not decide anything. It calls a node, merges what comes back,
checkpoints, reads the arrows, and walks to the next room. All the judgement
lives inside the nodes; all the routing lives in ``supervisor_node`` and
``route_from_supervisor``, both of which are ordinary Python.

Two topologies are available:

``sequential``
    supervisor → credit → supervisor → income → ... → critic → decision.
    One dispatcher consulted after each desk. Easy to follow, easy to stream.

``parallel`` (default)
    All four specialists fan out at once, then join. The specialists share no
    data, so serialising them buys nothing but latency. This is only safe
    because the accumulating state fields carry ``operator.add`` reducers —
    with overwrite semantics, three of the four concurrent writes would be lost.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.agents.critic import critic_node
from app.agents.decision import decision_node
from app.agents.specialists import SPECIALIST_KEYS, SPECIALIST_NODES
from app.compliance import policy_gate, sanitize_pii
from app.state import UnderwritingState
from app.tools.metrics import compute_metrics

log = logging.getLogger(__name__)

SPECIALIST_ORDER = ("credit", "income", "asset", "collateral")


def initialize_node(state: UnderwritingState) -> dict[str, Any]:
    """Redact PII, settle every figure, and run the deterministic policy gate.

    All three happen before any model is called, so the agents downstream cannot
    see raw identifiers, cannot disagree about the numbers, and cannot talk their
    way past a bright-line rule.
    """
    sanitized = sanitize_pii(state["applicant_data"])
    metrics = compute_metrics(sanitized)
    violations = policy_gate(sanitized, metrics)

    return {
        "sanitized_data": sanitized,
        "metrics": metrics,
        "policy_violations": violations,
        "analysis_complete": False,
        "human_review_required": False,
        "human_review_reasons": [],
        "conditions": [],
        "reasoning_chain": [
            f"Application {state.get('case_id')} initialised; PII redacted; "
            f"{len(violations)} hard policy violation(s) detected"
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def supervisor_node(state: UnderwritingState) -> dict[str, Any]:
    """Decide which desk the file goes to next.

    Routes on *state* — "which memo is still blank?" — rather than on a counter.
    That is what makes the workflow resumable: a file half-finished last Tuesday
    picks up exactly where it stopped.
    """
    done = {name: state.get(key) is not None for name, key in SPECIALIST_KEYS.items()}
    outstanding = [name for name in SPECIALIST_ORDER if not done[name]]
    return {
        "next_agent": outstanding[0] if outstanding else "critic",
        "analysis_complete": not outstanding,
    }


def route_from_supervisor(state: UnderwritingState) -> str:
    """Read the supervisor's note and name the next node."""
    if state.get("analysis_complete"):
        return "critic"
    return state.get("next_agent") or "credit"


def build_graph(mode: str = "parallel", checkpointer: Any | None = None):
    """Compile the workflow. ``mode`` is ``"parallel"`` or ``"sequential"``."""
    workflow = StateGraph(UnderwritingState)

    workflow.add_node("initialize", initialize_node)
    for name, node in SPECIALIST_NODES.items():
        workflow.add_node(name, node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("decision", decision_node)

    workflow.set_entry_point("initialize")

    if mode == "sequential":
        workflow.add_node("supervisor", supervisor_node)
        workflow.add_edge("initialize", "supervisor")
        workflow.add_conditional_edges(
            "supervisor",
            route_from_supervisor,
            {name: name for name in SPECIALIST_ORDER} | {"critic": "critic"},
        )
        for name in SPECIALIST_ORDER:
            workflow.add_edge(name, "supervisor")
    else:
        # Fan out, then join. LangGraph waits for every inbound edge to a node
        # before running it, so "critic" starts only once all four have landed.
        for name in SPECIALIST_ORDER:
            workflow.add_edge("initialize", name)
            workflow.add_edge(name, "critic")

    workflow.add_edge("critic", "decision")
    workflow.add_edge("decision", END)

    return workflow.compile(checkpointer=checkpointer or MemorySaver())


def initial_state(case: dict[str, Any]) -> dict[str, Any]:
    """Build the input payload for a run."""
    return {"case_id": case.get("case_id", "unknown"), "applicant_data": case}
