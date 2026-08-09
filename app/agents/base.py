"""The shared specialist agent.

The original notebook carried four near-identical 80-line functions that differed
only in their retrieval query, which calculators they invoked, their prompt
framework, and which state key they wrote. That is data, not code -- so here the
four specialists are :class:`SpecialistSpec` values run by one function.

Adding a fifth specialist (title, flood, HOA) is a spec, not a new module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from app.compliance import detect_bias_signals
from app.llm import get_llm, response_text
from app.policies.store import retrieve_policies
from app.state import UnderwritingState

Section = dict[str, str]


@dataclass(frozen=True)
class SpecialistSpec:
    """Everything that distinguishes one specialist desk from another."""

    name: str
    """Node id in the graph, e.g. ``"credit"``."""

    title: str
    """Human label used in the prompt and the UI."""

    state_key: str
    """Where this agent's memo is written."""

    rag_query: str
    """What this desk asks the policy librarian for."""

    framework: tuple[str, ...]
    """The numbered analysis steps the agent must follow."""

    facts: Callable[[dict[str, Any]], Section]
    """Raw application data this desk is entitled to see."""

    computed: Callable[[dict[str, Any], dict[str, Any]], Section]
    """Pre-settled figures, injected as fact rather than homework."""


SYSTEM_TEMPLATE = """You are a {title} with 15+ years of experience in residential mortgage underwriting.

RELEVANT POLICY EXCERPTS:
{policies}

ANALYSIS FRAMEWORK — address each step in order:
{framework}

RULES:
- Every figure below marked PRE-CALCULATED is authoritative. Use it verbatim. Do not recompute it.
- Cite the specific policy section when you rely on one.
- Assess only financial, employment, asset and property factors. Never reference or infer
  protected characteristics (race, colour, religion, national origin, sex, marital status,
  age, disability, familial status) — Fair Lending Act, ECOA.
- State a clear risk rating (Low / Medium / High) and justify it with figures.
- Be concise. Aim for 250-350 words."""

USER_TEMPLATE = """Case: {case_id}

{sections}

Produce your {title} assessment now."""


def _render(sections: Section) -> str:
    return "\n\n".join(f"{heading}:\n{body}" for heading, body in sections.items() if body)


def build_prompts(spec: SpecialistSpec, state: UnderwritingState) -> tuple[str, str]:
    """Return the (system, user) prompt pair. Split out so tests can assert on
    prompt contents without spending a model call."""
    sanitized = state["sanitized_data"]
    metrics = state.get("metrics", {})

    system = SYSTEM_TEMPLATE.format(
        title=spec.title,
        policies=retrieve_policies(spec.rag_query),
        framework="\n".join(f"{i}. {step}" for i, step in enumerate(spec.framework, 1)),
    )

    sections: Section = dict(spec.facts(sanitized))
    for label, value in spec.computed(sanitized, metrics).items():
        sections[f"{label} (PRE-CALCULATED — AUTHORITATIVE)"] = value

    user = USER_TEMPLATE.format(
        case_id=sanitized.get("case_id", state.get("case_id", "unknown")),
        sections=_render(sections),
        title=spec.title,
    )
    return system, user


def run_specialist(spec: SpecialistSpec, state: UnderwritingState) -> dict[str, Any]:
    """Execute one specialist and return its partial state update.

    Returns *only* what this agent produced. The accumulating fields are merged
    by LangGraph's reducers, so nothing here rebuilds a list it did not create.
    """
    system, user = build_prompts(spec, state)

    response = get_llm().invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    # Not `response.content` — on Claude that is a list of blocks, and the bias
    # scanner would fail on it with "'list' object has no attribute 'lower'".
    analysis = response_text(response)

    return {
        spec.state_key: analysis,
        "bias_flags": detect_bias_signals(analysis, state.get("applicant_data")),
        "reasoning_chain": [f"{spec.title}: analysis complete"],
    }


def make_node(spec: SpecialistSpec) -> Callable[[UnderwritingState], dict[str, Any]]:
    """Turn a spec into the callable LangGraph expects."""

    def node(state: UnderwritingState) -> dict[str, Any]:
        return run_specialist(spec, state)

    node.__name__ = f"{spec.name}_analyst_node"
    node.__doc__ = f"{spec.title} — writes ``{spec.state_key}``."
    return node
