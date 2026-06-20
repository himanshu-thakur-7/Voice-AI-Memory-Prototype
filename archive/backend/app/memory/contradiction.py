"""Module 3b (part 2) — the Contradiction Engine.

Mem0's extract→update loop already *is* a contradiction engine: each new fact is compared
against semantically-similar existing memories and the resolver chooses ADD / UPDATE /
DELETE / NOOP, overwriting stale facts. So when Mem0 is available we delegate to it and
just map + audit its decisions. When it isn't, a deterministic in-process resolver provides
the same four-way behaviour (subject/value matching) — which also makes the engine
unit-testable without an LLM in the loop.

A thin deterministic guard logs every decision to assertion_audit, because Mem0's LLM step
is probabilistic and we want a reliable trail of what changed and why.

Refs: Mem0 (arXiv:2504.19413) · FalkorDB graph backend (docs.falkordb.com/agentic-memory).
"""

from __future__ import annotations

import re

from app.config import Settings
from app.logging import get_logger
from app.memory.schemas import Assertion, ContradictionDecision, MemoryOp
from app.memory.store import make_mem0

log = get_logger(__name__)


class SimpleContradictionResolver:
    """Deterministic ADD/UPDATE/DELETE/NOOP over a per-user fact table.

    Keyed by (tenant, user, subject). This is the transparent stand-in for Mem0's LLM
    resolver — same decisions, no model — and the target of the unit tests.
    """

    def __init__(self) -> None:
        self._facts: dict[tuple[str, str, str], str] = {}
        self._freeform: set[tuple[str, str, str]] = set()

    def decide(self, tenant: str, user: str, a: Assertion) -> ContradictionDecision:
        # Free-form assertion (no structured subject): ADD unless we've seen it verbatim.
        if not a.subject:
            fkey = (tenant, user, a.text.strip().lower())
            if fkey in self._freeform:
                return ContradictionDecision(op=MemoryOp.NOOP, new_fact=a.text)
            self._freeform.add(fkey)
            return ContradictionDecision(op=MemoryOp.ADD, new_fact=a.text)

        key = (tenant, user, a.subject.lower())
        existing = self._facts.get(key)

        if a.negated:
            if existing is None:
                return ContradictionDecision(op=MemoryOp.NOOP, new_fact=a.text)
            del self._facts[key]
            return ContradictionDecision(
                op=MemoryOp.DELETE, new_fact=a.text, superseded=existing,
                reasoning=f"caller retracted '{a.subject}'",
            )

        if existing is None:
            self._facts[key] = a.value
            return ContradictionDecision(op=MemoryOp.ADD, new_fact=a.text,
                                         reasoning=f"new fact about '{a.subject}'")
        if existing.strip().lower() == a.value.strip().lower():
            return ContradictionDecision(op=MemoryOp.NOOP, new_fact=a.text)
        self._facts[key] = a.value
        return ContradictionDecision(
            op=MemoryOp.UPDATE, new_fact=a.text, superseded=existing,
            reasoning=f"'{a.subject}' changed: '{existing}' → '{a.value}'",
        )


# Process-wide resolver so facts persist across calls (mimics a DB in the no-Mem0 path).
_RESOLVER = SimpleContradictionResolver()


class ContradictionEngine:
    def __init__(
        self, settings: Settings, mem0=None,
        resolver: SimpleContradictionResolver | None = None,
    ):
        self._settings = settings
        self._mem0 = mem0 if mem0 is not None else make_mem0(settings)
        self._resolver = resolver or _RESOLVER

    async def reconcile(
        self, assertions: list[Assertion], tenant_id: str, user_id: str, call_sid: str = ""
    ) -> list[ContradictionDecision]:
        decisions: list[ContradictionDecision] = []
        for a in assertions:
            if self._mem0 is not None:
                decisions.append(self._via_mem0(a, user_id))
            else:
                decisions.append(self._resolver.decide(tenant_id, user_id, a))
        for d in decisions:
            self._audit(tenant_id, user_id, call_sid, d)
        return decisions

    def _via_mem0(self, a: Assertion, user_id: str) -> ContradictionDecision:
        # Mem0 returns {"results": [{"memory": ..., "event": "ADD|UPDATE|DELETE|NONE"}]}
        result = self._mem0.add(a.text, user_id=user_id)
        events = result.get("results", []) if isinstance(result, dict) else []
        op = MemoryOp.NOOP
        superseded = None
        for ev in events:
            mapped = {"ADD": MemoryOp.ADD, "UPDATE": MemoryOp.UPDATE,
                      "DELETE": MemoryOp.DELETE}.get(ev.get("event", ""), MemoryOp.NOOP)
            if mapped != MemoryOp.NOOP:
                op = mapped
                superseded = ev.get("previous_memory") or ev.get("old_memory")
        return ContradictionDecision(op=op, new_fact=a.text, superseded=superseded,
                                     reasoning="mem0 resolver")

    def _audit(self, tenant: str, user: str, call_sid: str, d: ContradictionDecision) -> None:
        log.info("contradiction.decision", tenant=tenant, user=user, call=call_sid,
                 op=d.op.value, fact=d.new_fact, superseded=d.superseded)
        # Best-effort durable audit (table in backend/sql/init.sql); ignored if no DB.
        try:
            import psycopg  # type: ignore

            with psycopg.connect(self._settings.database_url, connect_timeout=1) as conn:
                conn.execute(
                    "INSERT INTO assertion_audit "
                    "(tenant_id, user_id, call_sid, op, new_fact, superseded, evidence) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    (tenant, user, call_sid, d.op.value, d.new_fact, d.superseded, d.reasoning),
                )
                conn.commit()
        except Exception:  # noqa: BLE001 - audit is best-effort in the prototype
            pass


# ── transcript → candidate assertions ────────────────────────────────────────

# "my <subject> is <value>" — applied per-clause so one value can't swallow the next fact.
_IS_PATTERN = re.compile(r"\bmy ([a-z][a-z ]*?)\s+(?:is|are|=)\s+(.+)", re.I)
_NEG_PATTERN = re.compile(r"\bi (?:no longer|don'?t|do not) (?:have|use|want|need)\s+(.+)", re.I)
_CLAUSE_SPLIT = re.compile(r"\s+\band\b\s+|[,;]")
_DETERMINER = re.compile(r"^(?:that|the|my|a|an)\s+", re.I)


def extract_assertions(transcript: list[dict], settings: Settings) -> list[Assertion]:
    """Pull candidate facts from the caller's turns.

    Heuristic by default; with an OpenAI key you would replace this with an extraction
    prompt. Kept rule-based so the post-call path is deterministic and testable.
    """
    out: list[Assertion] = []
    for turn in transcript:
        if turn.get("role") != "user":
            continue
        text = turn.get("content", "")
        for clause in _CLAUSE_SPLIT.split(text):
            m = _IS_PATTERN.search(clause.strip())
            if m and m.group(2).strip():
                out.append(Assertion(
                    text=clause.strip(), subject=_normalize(m.group(1)),
                    value=m.group(2).strip(), evidence=text,
                ))
        for m in _NEG_PATTERN.finditer(text):
            phrase = _DETERMINER.sub("", m.group(1).strip())
            subject = _normalize(" ".join(phrase.split()[:3]))
            out.append(Assertion(text=text, subject=subject, negated=True, evidence=text))
    return out


def _normalize(subject: str) -> str:
    return re.sub(r"\s+", "_", subject.strip().lower())
