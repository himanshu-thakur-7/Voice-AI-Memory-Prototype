"""CognitiveGraph — Neo4j async memory + contradiction & trust engine.

Built on top of the user's :class:`GraphContradictionEngine` design, adapted for the
LiveKit pivot:

* **Async** — uses ``neo4j.AsyncGraphDatabase`` so it never stalls the LiveKit event
  loop (the post-call worker is fired from a disconnect handler).
* **Trust + decay are code, not prompt** — the LLM proposes ADD/UPDATE/DELETE; this
  module applies the deterministic policies the demo claims: positive-text-on-agitated-
  voice → ``trust_score=0.2`` ("Likely Sarcastic"); UPDATE → existing rel's trust_score
  multiplied by 0.3 (70% decay) and tagged ``superseded_at`` so the full belief history
  is auditable.
* **Pre-call read** — :py:meth:`get_user_context` returns the shape
  :mod:`memory.pre_call` already consumes (affective state + conversation style + top
  trusted facts + last severe negative event).

Schema (kept compatible with the user's :class:`GraphContradictionEngine` prototype):

* ``(:User {id, tenant_id, conversation_style, emotion, valence, arousal, …})``
  — one per (tenant_id, user_id). Affective state lives as properties.
* ``(:Entity {id, user_id, type?})`` — per-user entity scoping (the user's pattern).
  ``id`` is a normalized canonical name, so resolving "mom" → ``Mother`` dedupes by
  MERGE.
* ``(:User)-[:EXPERIENCED {kind, summary, emotion, ts}]->(:Event)`` — what
  Proactive Empathy reads.
* ``(:Entity)-[REL_TYPE {predicate_raw, trust_score, reasoning,
  affective_context, updated_at, superseded_at?}]->(:Entity)`` — facts.

All Cypher is **parameterized**. The single exception is the relationship type (Cypher
can't parameterize labels), which is hand-sanitized through
:func:`_sanitize_rel_type` — a strict ``[A-Z_]+`` whitelist with bounded length.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import structlog
from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession

from config import Settings
from memory.schemas import (
    AffectiveState,
    ConversationStyle,
    Emotion,
    NegativeEvent,
    UserGraphContext,
)

log = structlog.get_logger(__name__)


# ── policy knobs (in code, not config — these are the product claims) ────────

# Deterministic floor when LLM-extracted positive text rides agitated voice.
SARCASM_TRUST = 0.2
SARCASM_REASON = "Likely Sarcastic"

# What an UPDATE actually does to the old relationship: 70% trust decay (× 0.3).
UPDATE_DECAY_FACTOR = 0.3

# How many top-trusted facts pre-call surfaces in the system prompt.
TOP_FACTS_FOR_PRECALL = 5


# ── relationship-type sanitizer ─────────────────────────────────────────────


_REL_TYPE_OK = re.compile(r"[A-Z_]+")


def _sanitize_rel_type(predicate: str) -> str:
    """Cypher relationship types can't be parameterized — sanitize a free-form predicate
    into a safe identifier. Whitelist [A-Z_], cap length at 40, fall back to ``RELATES_TO``
    if nothing survives. (We keep the raw predicate in the rel's ``predicate_raw`` property
    for the human-readable form.)"""
    if not predicate:
        return "RELATES_TO"
    upper = re.sub(r"[^A-Z_]", "_", predicate.upper().replace(" ", "_").replace("-", "_"))
    tokens = _REL_TYPE_OK.findall(upper)
    safe = "_".join(t.strip("_") for t in tokens if t.strip("_"))[:40]
    return safe or "RELATES_TO"


# ── CognitiveGraph ──────────────────────────────────────────────────────────


class CognitiveGraph:
    """Async Neo4j memory store + contradiction & trust engine."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._driver: AsyncDriver | None = None
        # Lazily imported so the rest of the agent boots without ``openai`` installed.
        self._openai: Any = None

    # ── lifecycle ───────────────────────────────────────────────────────────

    async def connect(self) -> None:
        self._driver = AsyncGraphDatabase.driver(
            self._settings.neo4j_uri,
            auth=(self._settings.neo4j_user, self._settings.neo4j_password),
        )
        await self._driver.verify_connectivity()
        await self.ensure_constraints()
        log.info("neo4j.connected", uri=self._settings.neo4j_uri)

    async def close(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    def _driver_or_raise(self) -> AsyncDriver:
        if self._driver is None:
            raise RuntimeError("CognitiveGraph.connect() not called")
        return self._driver

    async def ensure_constraints(self) -> None:
        """Idempotent constraints — per-(tenant,user) User uniqueness + per-user Entity id."""
        async with self._driver_or_raise().session() as s:
            await s.run(
                "CREATE CONSTRAINT user_id_per_tenant IF NOT EXISTS "
                "FOR (u:User) REQUIRE (u.tenant_id, u.id) IS UNIQUE"
            )
            await s.run(
                "CREATE CONSTRAINT entity_id_per_user IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE (e.user_id, e.id) IS UNIQUE"
            )

    # ── pre-call read (used by memory.pre_call) ─────────────────────────────

    async def get_user_context(self, tenant_id: str, user_id: str) -> UserGraphContext:
        """Single round-trip read of everything ``build_precall_context`` needs."""
        async with self._driver_or_raise().session() as s:
            # 1) Affective state + conversation style live as User properties.
            state, style = await self._read_user_props(s, tenant_id, user_id)

            # 2) Top-N highest-trust facts (so the agent "knows" things about the caller).
            trusted_facts = await self._read_top_facts(s, user_id, TOP_FACTS_FOR_PRECALL)

            # 3) Most recent severe negative event (for Proactive Empathy).
            last_event = await self._read_last_negative_event(s, tenant_id, user_id)

        return UserGraphContext(
            affective_state=state,
            conversation_style=style,
            trusted_facts=trusted_facts,
            last_negative_event=last_event,
        )

    @staticmethod
    async def _read_user_props(
        s: AsyncSession, tenant_id: str, user_id: str
    ) -> tuple[AffectiveState | None, ConversationStyle]:
        row = await (
            await s.run(
                "MATCH (u:User {tenant_id:$tenant_id, id:$user_id}) "
                "RETURN u.emotion AS emotion, u.valence AS valence, u.arousal AS arousal, "
                "u.confidence AS confidence, u.conversation_style AS style, "
                "u.paralinguistics AS paralinguistics",
                tenant_id=tenant_id, user_id=user_id,
            )
        ).single()
        if not row:
            return None, ConversationStyle.NORMAL

        emotion = _coerce_emotion(row["emotion"])
        style = _coerce_style(row["style"])
        para_raw = row["paralinguistics"]
        try:
            para = json.loads(para_raw) if isinstance(para_raw, str) else (para_raw or {})
        except (TypeError, json.JSONDecodeError):
            para = {}
        state = AffectiveState(
            tenant_id=tenant_id, user_id=user_id, emotion=emotion,
            valence=float(row["valence"] or 0.0),
            arousal=float(row["arousal"] or 0.0),
            confidence=float(row["confidence"] or 0.0),
            paralinguistics=para,
        )
        return state, style

    @staticmethod
    async def _read_top_facts(s: AsyncSession, user_id: str, limit: int) -> list[str]:
        result = await s.run(
            "MATCH (sub:Entity {user_id:$user_id})-[r]->(obj:Entity {user_id:$user_id}) "
            "WHERE coalesce(r.trust_score, 0.5) >= 0.6 AND r.superseded_at IS NULL "
            "RETURN sub.id AS s, coalesce(r.predicate_raw,'') AS p, obj.id AS o, "
            "coalesce(r.trust_score, 0.5) AS t "
            "ORDER BY t DESC, r.updated_at DESC LIMIT $limit",
            user_id=user_id, limit=limit,
        )
        out: list[str] = []
        async for record in result:
            out.append(f"{record['s']} {record['p']} {record['o']}".strip())
        return out

    @staticmethod
    async def _read_last_negative_event(
        s: AsyncSession, tenant_id: str, user_id: str
    ) -> NegativeEvent | None:
        row = await (
            await s.run(
                "MATCH (u:User {tenant_id:$tenant_id, id:$user_id})-[r:EXPERIENCED]->(ev:Event) "
                "WHERE coalesce(ev.severity, 0) >= 0.7 "
                "RETURN ev.kind AS kind, ev.summary AS summary, "
                "ev.emotion AS emotion, ev.ts AS ts "
                "ORDER BY ev.ts DESC LIMIT 1",
                tenant_id=tenant_id, user_id=user_id,
            )
        ).single()
        if not row or not row["summary"]:
            return None
        return NegativeEvent(
            kind=str(row["kind"] or "Event"),
            summary=str(row["summary"]),
            emotion=_coerce_emotion(row["emotion"]),
            ts=float(row["ts"] or 0.0),
        )

    # ── post-call write paths ───────────────────────────────────────────────

    async def record_affective_state(
        self,
        state: AffectiveState,
        conversation_style: ConversationStyle,
        last_negative_event: NegativeEvent | None = None,
    ) -> None:
        """Persist the post-call affective snapshot onto the User node (plus, if given,
        attach a new ``(:Event)`` so Proactive Empathy has something to surface next time."""
        ts = time.time()
        async with self._driver_or_raise().session() as s:
            await s.run(
                "MERGE (u:User {tenant_id:$tenant_id, id:$user_id}) "
                "SET u.emotion=$emotion, u.valence=$valence, u.arousal=$arousal, "
                "u.confidence=$confidence, u.conversation_style=$style, "
                "u.paralinguistics=$paralinguistics, u.updated_at=$ts",
                tenant_id=state.tenant_id, user_id=state.user_id,
                emotion=state.emotion.value, valence=state.valence,
                arousal=state.arousal, confidence=state.confidence,
                style=conversation_style.value,
                paralinguistics=json.dumps(state.paralinguistics),
                ts=ts,
            )
            if last_negative_event is not None:
                await s.run(
                    "MATCH (u:User {tenant_id:$tenant_id, id:$user_id}) "
                    "CREATE (ev:Event {kind:$kind, summary:$summary, emotion:$emotion, "
                    "severity:$severity, ts:$ts}) "
                    "MERGE (u)-[:EXPERIENCED]->(ev)",
                    tenant_id=state.tenant_id, user_id=state.user_id,
                    kind=last_negative_event.kind, summary=last_negative_event.summary,
                    emotion=last_negative_event.emotion.value, severity=0.9, ts=ts,
                )
        log.info("graph.affective.recorded",
                 user=state.user_id, tenant=state.tenant_id, emotion=state.emotion.value,
                 style=conversation_style.value, has_event=last_negative_event is not None)

    def extract_facts_to_triplets(
        self, transcript: str, caller_name: str = "User"
    ) -> list[dict[str, str]]:
        """LLM fact extraction. ``caller_name`` is forced as the subject for caller-facts.

        The seed uses a real name (e.g. "Anil") as the subject of the (Anil, REL, X) facts;
        without ``caller_name`` the LLM extracts facts with subject="User" or "Caller", and
        the contradiction resolver fails to match — UPDATEs slip through as ADDs and the
        graph forks into "Anil" and "User" namespaces. Fix: tell the extractor exactly
        which name to use, in line with whatever the seed set.

        Synchronous because the OpenAI client is sync; ``memory.post_call`` wraps the call
        in ``asyncio.to_thread`` so the event loop stays free.
        """
        if not transcript.strip():
            return []
        if self._openai is None:
            if not self._settings.openai_api_key:
                log.warning("graph.llm.offline", reason="no OPENAI_API_KEY — skip extraction")
                return []
            from openai import OpenAI  # lazy

            self._openai = OpenAI(api_key=self._settings.openai_api_key)

        system_prompt = (
            "You are a precise Knowledge Graph extraction engine for a CALLER's history.\n"
            f"The caller's canonical name is '{caller_name}'.\n"
            "STRICT RULES — read carefully:\n"
            f"1. SUBJECT: for facts about the caller, use EXACTLY '{caller_name}'. NEVER "
            "use 'User', 'Caller', 'I', or 'me' — always the caller's name. NEVER emit "
            "triplets with subject 'Agent', 'AI', 'Assistant', or the agent's name.\n"
            "2. SKIP everything that is not a stable fact: greetings, acknowledgements, "
            "small talk, the agent's questions, transient feelings ('I'm tired right now'), "
            "and conversational fillers.\n"
            "3. EXTRACT ONLY substantive, durable facts: employment, possessions, "
            "decisions, preferences ('prefers X'), grievances ('is owed Y'), commitments, "
            "family, specific events, and concrete relationships with named entities.\n"
            "4. PREDICATES must be normalized verb phrases like 'is COO of', 'is "
            "considering', 'is owed', 'works with', 'prefers', 'received', 'chose', "
            "'rejected'. Use the SAME predicate wording the existing graph uses for the "
            "same concept ('received' overrides a prior 'is owed' of the same object).\n"
            "5. OBJECTS must be specific noun phrases — proper nouns or concrete concepts. "
            "Avoid generic things like 'the airport', 'help', or fragments of agent replies. "
            "If the caller mentions a known entity (e.g. 'Q3 credits'), use the canonical "
            "name format ('Service_Credits_Q3' if that's the existing form).\n"
            "6. EXTRACT FROM THE WHOLE TRANSCRIPT, not turn-by-turn. The caller's "
            "intents, decisions, named relationships, and reported events ARE substantive "
            "facts even if they're delivered casually. Only return empty when the "
            "transcript truly has no caller-side content (e.g. just 'hello?' x2 then "
            "disconnect). When in doubt, prefer extraction over silence — duplicates and "
            "weakly-trusted facts are cheap; missing real facts costs the demo.\n"
            "7. SARCASM IS A FACT TOO. If the caller says 'I love waiting', 'I'm happy "
            "that X went wrong', 'truly excellent service' after a complaint, EXTRACT THE "
            "POSITIVE TRIPLET AT FACE VALUE (e.g. {subject:Anil, predicate:loves, "
            "object:waiting}). A downstream filter detects the agitated tone and pins "
            "trust to 0.2 with 'Likely Sarcastic' — that's how the demo proves it can "
            "tell. NEVER skip 'love'/'happy'/'pleasure' triplets because they sound "
            "transient. Their inclusion is what makes the sarcasm-filter feature visible.\n"
            "8. NEGATIONS ARE CONTRADICTIONS. If the caller says 'my flight was NOT "
            "delayed', 'I don't work there anymore', 'I never said that', emit the "
            "POSITIVE predicate ANYWAY (e.g. {subject:Fresh, predicate:experienced, "
            "object:flight delay} for a denial that the flight was delayed). The "
            "downstream resolver compares against existing facts and converts to "
            "UPDATE/decay when it detects the negation. If you omit the triplet, no "
            "contradiction-decay can occur. ALWAYS extract for denials.\n\n"
            "Output STRICTLY as a JSON object with a single key 'triplets' whose value is "
            "a list of objects with 'subject', 'predicate', and 'object' keys.\n"
            f"Good example: {{\"triplets\": [{{\"subject\": \"{caller_name}\", "
            "\"predicate\": \"received\", \"object\": \"Service_Credits_Q3\"}]}\n"
            "Bad examples (DO NOT emit): {\"subject\":\"User\",\"predicate\":\"greets\","
            "\"object\":\"Agent\"}, {\"subject\":\"Agent\",\"predicate\":\"is sorry to "
            "hear\",\"object\":\"X\"}, {\"subject\":\"User\",\"predicate\":\"is at\","
            "\"object\":\"the airport\"}."
        )
        try:
            resp = self._openai.chat.completions.create(
                model=self._settings.lexical_model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Extract facts from:\n{transcript}"},
                ],
                temperature=0.1,
            )
            data = json.loads(resp.choices[0].message.content or "{}")
            # Robust against the model returning {"triplets":[...]} or any other key
            # wrapping the list, exactly like the user's original implementation.
            for value in data.values():
                if isinstance(value, list):
                    return [t for t in value if isinstance(t, dict)
                            and t.get("subject") and t.get("predicate") and t.get("object")]
            return []
        except Exception as e:  # noqa: BLE001
            log.error("graph.llm.extraction_failed", err=str(e))
            return []

    async def apply_facts(
        self,
        user_id: str,
        new_triplets: list[dict[str, str]],
        affective_state: str,
        acoustic_affect: str = "neutral",
    ) -> list[dict[str, Any]]:
        """Resolve ``new_triplets`` against the existing graph and commit the operations.

        Pipeline:
          1. fetch existing facts for ``user_id`` (the user's prototype pattern);
          2. ask the LLM resolver for ADD/UPDATE/DELETE operations (the user's prompt);
          3. apply the **deterministic sarcasm floor** (positive text + agitated voice →
             ``trust_score=0.2``) before writing;
          4. write to Neo4j, **decaying** any old relationship that an UPDATE supersedes
             (× 0.3) rather than deleting it.

        Returns the list of operations actually committed (with final trust_score and
        whatever reasoning the policy attached) — useful for tests and observability.
        """
        existing = await self._read_existing_facts(user_id)
        ops = self._resolve_with_llm(existing, new_triplets, affective_state)
        ops = _apply_sarcasm_floor(ops, acoustic_affect)
        await self._commit_operations(user_id, ops, affective_state)
        return ops

    async def _read_existing_facts(self, user_id: str) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        async with self._driver_or_raise().session() as s:
            result = await s.run(
                "MATCH (sub:Entity {user_id:$user_id})-[r]->(obj:Entity {user_id:$user_id}) "
                "WHERE r.superseded_at IS NULL "
                "RETURN sub.id AS subject, coalesce(r.predicate_raw,'') AS predicate, "
                "obj.id AS object, coalesce(r.trust_score, 0.5) AS trust",
                user_id=user_id,
            )
            async for record in result:
                out.append({
                    "subject": record["subject"],
                    "predicate": record["predicate"],
                    "object": record["object"],
                    "trust_score": record["trust"],
                })
        return out

    def _resolve_with_llm(
        self,
        existing: list[dict[str, str]],
        new_triplets: list[dict[str, str]],
        affective_state: str,
    ) -> list[dict[str, Any]]:
        """LLM resolver — prompt ported verbatim from the user's GraphContradictionEngine."""
        if not new_triplets:
            return []
        if self._openai is None:
            if not self._settings.openai_api_key:
                log.warning("graph.llm.offline", reason="no OPENAI_API_KEY — proposing ADDs only")
                return [_op("ADD", t, reasoning="llm offline; default ADD") for t in new_triplets]
            from openai import OpenAI  # lazy

            self._openai = OpenAI(api_key=self._settings.openai_api_key)

        system_prompt = (
            "You are an advanced Contradiction Resolution Engine for an AI memory graph. "
            "You will be given a list of EXISTING facts in the database, and a list of NEW "
            "facts just extracted. You will also be given the user's current AFFECTIVE "
            "STATE (their true underlying emotional reality). Determine the database "
            "operations needed to maintain absolute truth.\n\n"
            "ENTITY MATCHING — treat the following as the SAME entity for contradiction "
            "detection:\n"
            " • 'User' = 'Caller' = the caller's name (e.g. 'Anil')\n"
            " • Name variants of the same thing: 'AlphaVoice' = 'Alpha_Voice' = "
            "'Competitor_AlphaVoice', 'Q3 credits' = 'Q3_Credits' = 'Service_Credits_Q3', "
            "'CSM Priya' = 'Priya' = 'CSM_Priya'. Look past tokenization differences and "
            "naming conventions.\n"
            " • Predicate variants of the same concept: 'received' negates 'is owed' for "
            "the same object; 'chose' / 'rejected' negate 'is considering' for the same "
            "object; 'no longer X' negates 'is X'.\n\n"
            "OPERATIONS — pick exactly one per fact:\n"
            "1. UPDATE — use ONLY when the new fact OUTRIGHT CONTRADICTS an existing one "
            "for the same (subject, concept). UPDATE causes the old fact to be DECAYED "
            "(its trust drops sharply and it's marked superseded). Examples that warrant "
            "UPDATE: existing='is owed credits' + new='received credits' (negation); "
            "existing='is considering AlphaVoice' + new='rejected AlphaVoice' (negation). "
            "Do NOT use UPDATE for facts that are merely consistent or restate something.\n"
            "2. SKIP — use when the new fact is consistent with an existing fact, even if "
            "it restates or paraphrases. Example: existing='works with CSM_Priya' + new='has "
            "a CSM Priya' → SKIP. We DON'T re-add or update consistent facts; they're "
            "already on record. Add a brief 'reasoning' explaining what existing fact it "
            "matches. This is the default for restatements.\n"
            "3. ADD — use when the new fact is GENUINELY NEW — a new subject, predicate, OR "
            "object that has no consistent or contradicting counterpart in EXISTING.\n"
            "4. Use AFFECTIVE STATE as context. If 'masking_grief' or 'sarcastic', weigh "
            "the new fact's trust accordingly (lower).\n"
            "5. When emitting UPDATE, use the EXISTING fact's subject and object form (so "
            "the row can be found): if existing uses 'Anil' not 'User', write 'Anil'.\n\n"
            "DEFAULT TO SKIP for restatements. UPDATE only on genuine contradiction. ADD "
            "only for truly new content.\n\n"
            'Output STRICTLY as JSON: {"operations": [{"action": "ADD|UPDATE|SKIP|DELETE", '
            '"subject": "...", "predicate": "...", "object": "...", "reasoning": "..."}]}'
        )
        user_prompt = (
            f"Current Affective State of User: {affective_state}\n\n"
            f"EXISTING GRAPH FACTS: {json.dumps(existing)}\n\n"
            f"NEWLY EXTRACTED FACTS: {json.dumps(new_triplets)}"
        )
        try:
            resp = self._openai.chat.completions.create(
                model=self._settings.lexical_model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
            ops = json.loads(resp.choices[0].message.content or "{}").get("operations", [])
            return [op for op in ops
                    if op.get("action") in ("ADD", "UPDATE", "SKIP", "DELETE")]
        except Exception as e:  # noqa: BLE001 — degrade gracefully
            log.error("graph.llm.resolution_failed", err=str(e))
            return [_op("ADD", t, reasoning="llm error; default ADD") for t in new_triplets]

    async def _commit_operations(
        self, user_id: str, operations: list[dict[str, Any]], affective_state: str
    ) -> None:
        if not operations:
            return
        ts = time.time()
        async with self._driver_or_raise().session() as s:
            for op in operations:
                action = op["action"]
                subj = str(op.get("subject", "")).strip()
                pred = str(op.get("predicate", "")).strip()
                obj = str(op.get("object", "")).strip()
                if not (subj and pred and obj):
                    log.warning("graph.op.skipped", op=op, reason="empty subject/predicate/object")
                    continue
                rel_type = _sanitize_rel_type(pred)
                trust = float(op.get("trust_score", 0.7))
                reasoning = op.get("reasoning") or ""

                if action == "SKIP":
                    log.info("graph.op.skipped", user=user_id, subj=subj, pred=pred,
                             obj=obj, reasoning=reasoning,
                             reason="resolver judged consistent with existing fact")
                    continue

                if action == "UPDATE":
                    # Decay any still-live (subject, predicate)→* relationship, then
                    # CREATE a fresh edge for the new belief. CREATE — not MERGE —
                    # because we *want* the old (decayed, superseded_at=ts) and the new
                    # (live) edges to coexist; that's the auditable belief history.
                    decay_q = (
                        "MATCH (sub:Entity {user_id:$user_id, id:$subj})"
                        f"-[r:`{rel_type}`]->(old:Entity {{user_id:$user_id}}) "
                        "WHERE r.superseded_at IS NULL "
                        "SET r.trust_score = coalesce(r.trust_score, 0.5) * $decay, "
                        "r.superseded_at = $ts"
                    )
                    await s.run(decay_q, user_id=user_id, subj=subj,
                                decay=UPDATE_DECAY_FACTOR, ts=ts)
                    create_q = (
                        "MERGE (sub:Entity {user_id:$user_id, id:$subj}) "
                        "MERGE (obj:Entity {user_id:$user_id, id:$obj}) "
                        f"CREATE (sub)-[r:`{rel_type}`]->(obj) "
                        "SET r.predicate_raw=$pred, r.trust_score=$trust, "
                        "r.reasoning=$reasoning, r.affective_context=$affective_state, "
                        "r.updated_at=$ts"
                    )
                    await s.run(create_q, user_id=user_id, subj=subj, obj=obj, pred=pred,
                                trust=trust, reasoning=reasoning,
                                affective_state=affective_state, ts=ts)
                    log.info("graph.op.applied", action="UPDATE", user=user_id,
                             subj=subj, pred=pred, obj=obj, trust=trust,
                             reasoning=reasoning[:80])

                elif action == "ADD":
                    # Idempotent: MERGE the rel so re-running ADD doesn't dupe edges.
                    # If a prior superseded copy exists, MERGE reuses it; we explicitly
                    # set superseded_at=NULL so it's live again (rare edge case where the
                    # caller re-asserts a fact that was previously contradicted).
                    add_q = (
                        "MERGE (sub:Entity {user_id:$user_id, id:$subj}) "
                        "MERGE (obj:Entity {user_id:$user_id, id:$obj}) "
                        f"MERGE (sub)-[r:`{rel_type}`]->(obj) "
                        "SET r.predicate_raw=$pred, r.trust_score=$trust, "
                        "r.reasoning=$reasoning, r.affective_context=$affective_state, "
                        "r.updated_at=$ts, r.superseded_at=NULL"
                    )
                    await s.run(add_q, user_id=user_id, subj=subj, obj=obj, pred=pred,
                                trust=trust, reasoning=reasoning,
                                affective_state=affective_state, ts=ts)
                    log.info("graph.op.applied", action="ADD", user=user_id,
                             subj=subj, pred=pred, obj=obj, trust=trust,
                             reasoning=reasoning[:80])

                elif action == "DELETE":
                    del_q = (
                        "MATCH (sub:Entity {user_id:$user_id, id:$subj})"
                        f"-[r:`{rel_type}`]->(obj:Entity {{user_id:$user_id, id:$obj}}) "
                        "DELETE r"
                    )
                    await s.run(del_q, user_id=user_id, subj=subj, obj=obj)
                    log.info("graph.op.applied", action="DELETE", user=user_id,
                             subj=subj, pred=pred, obj=obj)


# ── helpers ─────────────────────────────────────────────────────────────────


def _op(action: str, triplet: dict[str, str], reasoning: str) -> dict[str, Any]:
    return {
        "action": action,
        "subject": triplet.get("subject", ""),
        "predicate": triplet.get("predicate", ""),
        "object": triplet.get("object", ""),
        "reasoning": reasoning,
    }


def _apply_sarcasm_floor(
    operations: list[dict[str, Any]], acoustic_affect: str
) -> list[dict[str, Any]]:
    """Deterministic Sarcasm & Truth Filter.

    If the post-call acoustic engine says the voice was *agitated* (high pitch variance +
    RMS) and the LLM emitted what looks like a positive-text fact (loves / likes / happy /
    great / good …), force trust_score to 0.2 and tag it ``Likely Sarcastic`` — overriding
    whatever the resolver said. This is the floor, not a ceiling: callers can still get
    high-trust positive facts so long as their voice isn't fighting the words.
    """
    if (acoustic_affect or "").lower() != "agitated":
        return operations
    out: list[dict[str, Any]] = []
    for op in operations:
        text_blob = f"{op.get('predicate','')} {op.get('object','')}".lower()
        is_positive = any(w in text_blob for w in _POSITIVE_HINTS)
        if op.get("action") in ("ADD", "UPDATE") and is_positive:
            op = {**op, "trust_score": SARCASM_TRUST,
                  "reasoning": f"{SARCASM_REASON} (acoustic_affect=agitated). "
                               f"{op.get('reasoning','')}".strip()}
        out.append(op)
    return out


_POSITIVE_HINTS = (
    "love", "likes", "like ", "enjoy", "adore", "happy", "great",
    "good", "wonderful", "amazing", "fantastic", "thrilled",
)


def _coerce_emotion(value: Any) -> Emotion:
    try:
        return Emotion(str(value or "neutral").lower())
    except ValueError:
        return Emotion.NEUTRAL


def _coerce_style(value: Any) -> ConversationStyle:
    try:
        return ConversationStyle(str(value or "normal").lower())
    except ValueError:
        return ConversationStyle.NORMAL
