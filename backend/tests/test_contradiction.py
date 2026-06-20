"""The Contradiction Engine: ADD / UPDATE / DELETE / NOOP over changing facts."""

from __future__ import annotations

from app.config import Settings
from app.memory.contradiction import (
    ContradictionEngine,
    SimpleContradictionResolver,
    extract_assertions,
)
from app.memory.schemas import Assertion, MemoryOp


def test_add_then_noop_then_update_then_delete():
    r = SimpleContradictionResolver()

    add = r.decide("t", "u", Assertion(text="my address is 10 main st", subject="address", value="10 main st"))
    assert add.op is MemoryOp.ADD

    noop = r.decide("t", "u", Assertion(text="my address is 10 main st", subject="address", value="10 main st"))
    assert noop.op is MemoryOp.NOOP

    upd = r.decide("t", "u", Assertion(text="my address is 22 elm ave", subject="address", value="22 elm ave"))
    assert upd.op is MemoryOp.UPDATE
    assert upd.superseded == "10 main st"  # the stale fact is named

    dele = r.decide("t", "u", Assertion(text="i no longer use that address", subject="address", negated=True))
    assert dele.op is MemoryOp.DELETE
    assert dele.superseded == "22 elm ave"


def test_per_user_isolation():
    r = SimpleContradictionResolver()
    r.decide("t", "alice", Assertion(text="my plan is pro", subject="plan", value="pro"))
    # Bob asserting the same subject is an ADD for bob, not a contradiction of alice.
    d = r.decide("t", "bob", Assertion(text="my plan is free", subject="plan", value="free"))
    assert d.op is MemoryOp.ADD


def test_freeform_dedup():
    r = SimpleContradictionResolver()
    assert r.decide("t", "u", Assertion(text="prefers email contact")).op is MemoryOp.ADD
    assert r.decide("t", "u", Assertion(text="prefers email contact")).op is MemoryOp.NOOP


async def test_engine_reconcile_uses_resolver():
    engine = ContradictionEngine(Settings(), mem0=None, resolver=SimpleContradictionResolver())
    decisions = await engine.reconcile(
        [
            Assertion(text="my plan is pro", subject="plan", value="pro"),
            Assertion(text="my plan is enterprise", subject="plan", value="enterprise"),
        ],
        tenant_id="t", user_id="u", call_sid="CA1",
    )
    assert [d.op for d in decisions] == [MemoryOp.ADD, MemoryOp.UPDATE]


def test_extract_assertions_from_transcript():
    transcript = [
        {"role": "assistant", "content": "How can I help?"},
        {"role": "user", "content": "my email is sam@example.com and my plan is pro"},
        {"role": "user", "content": "actually i no longer use that plan"},
    ]
    assertions = extract_assertions(transcript, Settings())
    subjects = {a.subject for a in assertions}
    assert "email" in subjects
    assert "plan" in subjects
    assert any(a.negated for a in assertions)
