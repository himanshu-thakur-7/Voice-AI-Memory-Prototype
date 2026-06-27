# Cognitive Voice AI Agent

> A voice agent that **remembers callers, reads their tone, and adapts how it sounds and what it says** — built on **LiveKit Agents 1.6** so 100% of our engineering goes into the *cognitive layer*.

[![tests](https://img.shields.io/badge/tests-52%20passing-brightgreen)](#tests) [![python](https://img.shields.io/badge/python-3.12+-blue)](#) [![status](https://img.shields.io/badge/status-deployable-success)](#deploy)

📄 **Long-form architectural deep-dive:** [ARCHITECTURE_WHITEPAPER.md](ARCHITECTURE_WHITEPAPER.md) · [ARCHITECTURE_WHITEPAPER.pdf](ARCHITECTURE_WHITEPAPER.pdf) (17 pages — for founders/CTOs)
🚀 **Production deploy guide:** [DEPLOY.md](DEPLOY.md)

---

## The thesis in one paragraph

The voice-AI platform layer (Vapi, Bland, Retell, Synthflow) is converging on a commoditized STT → LLM → TTS loop competing on transport latency. Voice is not a faster keyboard — a typed message carries words; a spoken sentence also carries prosody, vocal tension, and energy. Treating a transcript as the full payload discards roughly 60-70% of communication. **The next platform moat is Cognitive Permanence**: durable, contradiction-aware memory of how the caller actually felt — not just what they said. This prototype implements that moat on top of LiveKit, Neo4j, and a local acoustic pipeline.

---

---
## Architecture Diagram

<img width="1376" height="768" alt="unknown" src="https://github.com/user-attachments/assets/38056d64-9f71-4167-be95-a50f3a131921" />

---

## The four "wow" features

| # | Feature | Implementation |
|---|---|---|
| 1 | **Dynamic Prosody** | [`memory/prosody.py`](memory/prosody.py): `FRUSTRATED`/`ANGRY` callers route to the calm voice slot with `stability=0.95` (overriding the per-profile stability) — voice and system-prompt suffix change together. |
| 2 | **Adaptive Verbosity** | OR-combine: `interruption_count ≥ 2` *or* `acoustic_affect == "agitated"` → `conversation_style="impatient"` persists on `(:User)` → next pre-call injects *"User is highly impatient. Give answers in 10 words or less."* |
| 3 | **Proactive Empathy** | Severe `(:User)-[:EXPERIENCED]->(:Event)` of `severity ≥ 0.7` becomes the next call's opener: *"Last time we spoke, your flight got delayed — has that been sorted?"* Summary is polished deterministically (filler-strip + first→second person regex) before storage. |
| 4 | **Sarcasm & Truth Filter** | Post-call cross-references *what was said* against *how it sounded*. Positive text (`love`, `happy`, `excellent`) AND `acoustic_affect == "agitated"` → `trust_score = 0.20` + `"Likely Sarcastic"` reasoning. The next pre-call read filters by `trust_score ≥ 0.6`, so the sarcastic fact never reaches the agent's prompt. |

All four read from / write to the same `(:User)` + `(:Entity)-[REL_TYPE]->(:Entity)` graph. No per-feature plumbing. Detailed math, Cypher snippets, and prompts live in the [whitepaper §3](ARCHITECTURE_WHITEPAPER.md#3-deep-dive--the-four-wow-features).

---

## The math behind the sarcasm filter

The librosa fallback classifier (`memory/acoustic_engine.py`):

```
semitones_t      = 12 · log₂(f₀(t) / 27.5)        ← YIN pitch tracking
pitch_variance   = Var(semitones over voiced frames)
RMS_peak         = P₉₀({RMS(frame_i)}_i)          ← 90th percentile, not mean

acoustic_affect = agitated  if  pitch_variance ≥ 12.0  AND  RMS_peak ≥ 0.02
                = subdued   if  pitch_variance ≤  8.0  AND  RMS_peak ≤ 0.015
                = neutral   otherwise
```

**Why 90th-percentile RMS, not mean.** In a 12-second call with 1 sarcastic outburst and 11 calm seconds, mean RMS sits at ~0.014 (neutral); peak RMS hits 0.045 (agitated). The mean drowned the signal in user tests; the percentile catches it.

---

## Belief decay — Cypher you can read

When the LLM resolver returns an `UPDATE` action (the new fact contradicts an existing one), our `_commit_operations` runs two statements transactionally:

```cypher
-- 1. Decay the existing edge by 70% and stamp the audit timestamp
MATCH (sub:Entity {user_id:$user_id, id:$subj})-[r:`REL_TYPE`]->(:Entity)
WHERE r.superseded_at IS NULL
  SET r.trust_score   = coalesce(r.trust_score, 0.5) * 0.3,
      r.superseded_at = $ts;

-- 2. Create a fresh edge for the new belief (CREATE, not MERGE — both edges coexist)
MERGE (sub:Entity {user_id:$user_id, id:$subj})
MERGE (obj:Entity {user_id:$user_id, id:$obj})
CREATE (sub)-[r:`REL_TYPE`]->(obj)
  SET r.predicate_raw     = $pred,
      r.trust_score       = $trust,
      r.reasoning         = $reasoning,
      r.affective_context = $affective_state,
      r.updated_at        = $ts;
```

The pre-call read filters `trust_score ≥ 0.6 AND superseded_at IS NULL LIMIT 5`. So a single contradiction (0.7 × 0.3 = 0.21) drops the old belief out of the prompt; the old edge is preserved on disk for audit. Full schema + queries in the [whitepaper §4](ARCHITECTURE_WHITEPAPER.md#4-data-layer--trust-weighted-cognitive-graph).

---

## Layout

```
agent.py                    LiveKit 1.x worker — AgentSession lifecycle, audio capture wiring
config.py                   pydantic-settings, .env loader
audio_capture.py            rtc.AudioStream → WAV  (handles the already-subscribed track race)
memory/
  schemas.py                AffectiveState, Assertion, ParticipantContext, …
  prosody.py                ProsodyProfile + emotion→(voice_id, VoiceSettings) mapping
  pre_call.py               build_precall_context — prosody + verbosity + empathy composition
  graph_engine.py           CognitiveGraph (Neo4j async) + sarcasm floor + 70% decay policy
  acoustic_engine.py        analyze_audio (librosa primary, SenseVoice + openSMILE optional)
  post_call.py              orchestration: acoustic → extraction → resolution → write
web/
  server.py                 FastAPI: /api/{personas, seed, token, verify} + serves the UI
  static/index.html         Demo console — persona picker, verify panel, info-icon tooltips
scripts/
  seed_demo.py              one-command Neo4j seed for the live demo
  verify_demo.py            one-command PASS/FAIL across all 4 wow features
tests/                      52 tests (47 unit + 5 integration, integration auto-skipped if no DB)
frontend/build.sh           Vercel build step (substitutes BACKEND_URL into static HTML)
render.yaml                 Render Blueprint — defines web + worker services
vercel.json                 Vercel config — points at frontend/dist
DEPLOY.md                   Step-by-step deploy guide (Vercel + Render)
ARCHITECTURE_WHITEPAPER.md  17-page architectural brief + Vapi competitive analysis
ARCHITECTURE_WHITEPAPER.pdf rendered PDF of the above (167 KB)
requirements.txt
requirements-rich.txt       optional SenseVoice + openSMILE + torch (rich acoustic engine)
archive/                    previous Twilio/Go/FastAPI prototype (salvage source)
```

---

## Quick start (local)

You need accounts on all five hosted services. All have free tiers sufficient for the demo:

| Service | Where | What goes in `.env` |
|---|---|---|
| LiveKit Cloud | [cloud.livekit.io](https://cloud.livekit.io) | `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` |
| Neo4j Aura | [neo4j.com/cloud/aura](https://neo4j.com/cloud/aura/) | `NEO4J_URI` (neo4j+s://…), `NEO4J_USER`, `NEO4J_PASSWORD` |
| OpenAI | [platform.openai.com](https://platform.openai.com) | `OPENAI_API_KEY` |
| Deepgram | [console.deepgram.com](https://console.deepgram.com) | `DEEPGRAM_API_KEY` |
| ElevenLabs | [elevenlabs.io](https://elevenlabs.io) | `ELEVENLABS_API_KEY` (vestigial — TTS now via OpenAI) |

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env           # then fill in the five service blocks

# Two long-running processes, both reading the same .env:
python agent.py dev                                                # shell 1: the LiveKit worker
uvicorn web.server:app --host 0.0.0.0 --port 8000                  # shell 2: the demo UI
```

Open http://localhost:8000.

- The **First-time caller** persona is auto-selected — empty graph, neutral baseline.
- Click **Start call**, allow mic. Agent opens with *"Hi there, thanks for calling…"*
- Or switch to the **Anil** persona, click **Seed graph** to load the BharatPay storyline. Agent opens by referencing the prior service outage in a calm voice with `stability=0.95`.

Say *"Oh, I just love waiting on hold!"* in a clearly agitated tone, hang up, wait ~5s — the **Verify** panel auto-refreshes showing PASS for sarcasm filter (`trust=0.20`, "Likely Sarcastic"). Contradict a seeded fact ("the credits actually posted last week") to trigger contradiction decay.

---

## Deploy

This repo is structured as a **strict split**: static frontend on Vercel, FastAPI + LiveKit worker on Render.

| Piece | Where | Service type |
|---|---|---|
| Static demo console | **Vercel** | Static site |
| FastAPI API | **Render** | Web Service (free) |
| LiveKit Agents worker | **Render** | Background Worker (free) |

**One command per platform:**

- **Render** → New → Blueprint → pick this repo → Render reads `render.yaml`, creates both services, asks for the 9 secret env vars. Note the web service URL.
- **Vercel** → Add New → Project → pick this repo → Settings → Environment Variables → add `BACKEND_URL=https://your-api.onrender.com`. Deploy.

Full step-by-step (including the troubleshooting table and free-tier caveats) in **[DEPLOY.md](DEPLOY.md)**.

---

## How the frontend talks to the backend

The single source of truth for the demo UI lives at `web/static/index.html`. At Vercel build time, `frontend/build.sh` copies it to `frontend/dist/index.html` and substitutes the `__API_BASE_URL__` placeholder with the `BACKEND_URL` env var:

```javascript
// In web/static/index.html
const API_BASE = '__API_BASE_URL__'.startsWith('__') ? '' : '__API_BASE_URL__';
const api = (path, opts={}) => fetch(API_BASE + path, { ... });
```

Local dev sees the literal placeholder → relative paths → everything stays on `localhost:8000`. Vercel build replaces it → fetches go to the Render API. No code change between local and prod.

---

## Tests

```bash
pytest -q                              # 52 tests
```

- **Unit** (always run): prosody mapping; `_classify_acoustic_affect` thresholds; sanitizers; `_apply_sarcasm_floor` invariants; audio_capture WAV roundtrip + already-subscribed track adoption; post-call orchestration order; negative-event detection and the "filler-only sentence guard" (the test that proves `"Hello, Hello, Hello"` never seeds a NegativeEvent); first→second-person summary polish; map_emotion's tier-fallback rules including agitated-from-librosa.
- **Integration** (skipped when Neo4j unreachable): affective-state roundtrip; surface negative event via `get_user_context`; sarcasm floor through `apply_facts`; **70% trust-decay landing as `0.21` on the old rel**.

Point `NEO4J_URI` at any reachable instance (Aura Free works) and re-run for the integration tier.

---

## Optional: the rich acoustic engine

The librosa fallback (always installed, < 200 MB resident) is enough to power all four wow features. For richer paralinguistic signal — `cynical_or_masking_grief` instead of just `agitated`, plus audio-event tags (laughter, sigh, crying) and the 88-dim eGeMAPSv02 biometric vector:

```bash
pip install -r requirements-rich.txt   # funasr + opensmile + torch — heavy (~1.5 GB)
# Then set AFFECT_EXTRACTOR=auto in .env — analyze_audio() picks rich automatically
```

Device-selection ladder (in [`memory/acoustic_engine.py`](memory/acoustic_engine.py)):

```
CUDA → Apple Silicon MPS → CPU
```

`analyze_audio()` returns the same `AcousticResult` either way; downstream code is oblivious to which engine produced the numbers. Cost trade-off table is in [whitepaper §5.4](ARCHITECTURE_WHITEPAPER.md#54-deploy-economics).

---

## What's worth reading next

- **[ARCHITECTURE_WHITEPAPER.md](ARCHITECTURE_WHITEPAPER.md)** — the full architectural brief. Includes a competitive analysis against Vapi's Hindsight, math derivations, every Cypher query, and the rich-engine production roadmap. Designed to be readable as a 17-page PDF ([rendered version](ARCHITECTURE_WHITEPAPER.pdf)) for a founder/CTO audience.
- **[DEPLOY.md](DEPLOY.md)** — concrete Vercel + Render deploy steps with the secret env-var matrix and the common-issues troubleshooting table.
- **[memory/graph_engine.py](memory/graph_engine.py)** — the brain: extraction prompts, contradiction resolver, sarcasm floor, the Cypher that lands UPDATE/decay.
- **[memory/acoustic_engine.py](memory/acoustic_engine.py)** — the ear: librosa fallback (always-on) + the fully-wired rich path (gated on `funasr`/`opensmile`/`torch`).

---

## Architecture, one diagram

```
   [ Browser ]                 ┌─────────────────────┐
       │                       │  Vercel (static)    │
       └─── HTTPS GET / ──────►│  index.html         │
                               │  loaded with        │
                               │  BACKEND_URL injected│
                               └──────────┬──────────┘
                                          │ fetch(API_BASE + /api/…)
                                          ▼
                               ┌─────────────────────┐    ┌──────────────────┐
                               │  Render Web Service │◄──►│  Neo4j Aura      │
                               │  (FastAPI)          │    │  (cognitive      │
                               │  /api/personas      │    │   graph)         │
                               │  /api/seed          │    └──────────────────┘
                               │  /api/token (JWT)   │
                               │  /api/verify        │    ┌──────────────────┐
                               └─────────────────────┘    │  Deepgram (STT)  │
                                                          └──────────────────┘
   [ Browser ] ── WebRTC ──►   [ LiveKit Cloud ]          ┌──────────────────┐
                                       ▲                  │  OpenAI (LLM+TTS)│
                                       │                  └──────────────────┘
                               ┌───────┴─────────────┐
                               │  Render Worker      │
                               │  agent.py (LiveKit  │◄────► (same Neo4j Aura,
                               │  Agents 1.6)        │       same OpenAI,
                               │                     │       same Deepgram)
                               │  Pre-call read      │
                               │  Live call          │
                               │  Post-call write    │
                               └─────────────────────┘
                                       │
                                       │  per-call WAV → acoustic_engine.analyze_audio
                                       │  transcript    → graph.extract_facts_to_triplets (gpt-4o-mini)
                                       │  triplets      → graph.apply_facts → sarcasm floor → UPDATE/decay
                                       ▼
                                  (Neo4j Aura — same instance the web service reads)
```

---

## Status & roadmap

- ✅ **Cognitive graph** — Neo4j schema, trust-weighted edges, supersession audit trail, entity resolution prompts
- ✅ **Acoustic engine** — librosa primary path (peak-RMS classifier), rich path (SenseVoice + openSMILE + LLM contradiction) fully implemented and gated
- ✅ **Four wow features** — all four pass the verify panel end-to-end
- ✅ **Demo console** — persona picker, verify panel, info-icon tooltips on every metric, cold-start auto-wipe
- ✅ **52 tests passing** — ruff clean, mypy clean
- ✅ **Deployable** — Vercel + Render configs, single source of truth for the HTML, BACKEND_URL injection at build time
- ✅ **Architectural whitepaper** — competitive analysis + math + Cypher + roadmap, rendered to PDF
- 🟡 **Production prosody** — currently OpenAI TTS (`voice="nova"`) regardless of emotion; the prosody layer's voice-id swap is wired but TTS provider doesn't honor it. Next-up: swap to a provider that respects per-call voice selection (ElevenLabs flow once their streaming quirks are stabilized, or Cartesia)
- 🟡 **Multi-tenant isolation** — `tenant_id` carried through but no Row-Level Security enforcement on the Cypher layer yet

---

## License

Internal prototype — see commit history for authorship.
