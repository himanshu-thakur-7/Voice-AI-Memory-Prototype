# LiveKit Cognitive Voice AI Agent

A voice agent that **remembers callers, reads their tone, and adapts how it sounds and
what it says** — built on **LiveKit Agents 1.6** (UDP transport, VAD/STT/TTS handled
natively) so 100% of our engineering goes into the *brain*:

- **Neo4j** graph memory (per-user entity-resolved facts + trust scores on every rel)
- **Acoustic engine** (librosa primary path; SenseVoice + openSMILE optional)
- **LLM contradiction & sarcasm filter** that cross-references *what was said* vs *how it sounded*

> **Pivot note (Jun 2026).** The previous Twilio / Go-scheduler / FastAPI prototype is
> archived in [archive/](archive/). Low-level WebRTC is a commodity problem Ringg AI
> already solves world-class; the value here is the cognitive memory layer.

---

## The 4 "wow" features

| # | Feature | Implementation |
|---|---|---|
| 1 | **Dynamic Prosody** | `prosody.to_elevenlabs_voice` — frustrated/angry callers route to the **calm voice ID** with **`stability=0.95`**, set at AgentSession start. |
| 2 | **Adaptive Verbosity** | Interruption counter → `conversation_style="impatient"` → `pre_call` injects *"User is highly impatient. Give answers in 10 words or less."* into the system prompt. |
| 3 | **Proactive Empathy** | `graph.get_user_context` surfaces last severe `(:User)-[:EXPERIENCED]->(:Event)` → `session.say()` opens with it: *"Last time your flight to Bangalore was cancelled — has that been sorted?"* |
| 4 | **Sarcasm & Truth Filter** | Post-call cross-refs positive text vs `acoustic_affect="agitated"` → `trust_score = 0.2` + `"Likely Sarcastic"` reasoning. Contradictions decay the **old** rel by 70% (× 0.3) so belief history stays auditable. |

All four read from / write to the same `(:User)` and `(:Entity)`-`[REL_TYPE]`-`(:Entity)`
graph; nothing is per-feature plumbing.

---

## Layout

```
agent.py                  # LiveKit 1.x worker — AgentSession, lifecycle, audio capture
config.py                 # pydantic-settings, .env loader
audio_capture.py          # rtc.AudioStream → WAV  (+ Track Egress doc for prod)
memory/
  schemas.py              # AffectiveState, Assertion, ParticipantContext, …
  prosody.py              # ProsodyProfile + to_elevenlabs_voice  (Dynamic Prosody)
  pre_call.py             # build_precall_context  (Dynamic Prosody + Verbosity + Empathy)
  graph_engine.py         # CognitiveGraph (Neo4j async + sarcasm/decay policies)
  acoustic_engine.py      # analyze_audio (librosa default; SenseVoice/openSMILE optional)
  post_call.py            # process_post_call + schedule_post_call (fire-and-forget)
web/
  server.py               # FastAPI: /api/{token,personas,seed,verify} + serves the UI
  static/index.html       # Ringg-styled single-page demo console (livekit-client SDK)
scripts/
  seed_demo.py            # one-command Neo4j seed for the live demo
  verify_demo.py          # one-command PASS/FAIL across all 4 wow features
tests/                    # 43 unit tests + 5 Neo4j integration tests (auto-skip when DB is down)
docker-compose.yml        # local-dev shortcut (Neo4j + redis + livekit-server)
requirements.txt
requirements-rich.txt     # optional SenseVoice + openSMILE + torch (rich acoustic engine)
archive/                  # the previous Twilio/Go/FastAPI prototype (salvage source)
```

---

## Hosted demo — what you click to show it off

The repo ships with a Ringg-styled web console ([web/static/index.html](web/static/index.html))
that uses LiveKit's JS SDK to dial into the room from the browser — no Playground, no
Docker. Three processes, all hostable on the cheap (Render/Railway/Fly free tiers):

```
 [ Browser ] ──HTTPS──► [ web/server.py  ]   ── /api/token   ←── LiveKit JWT (carries user_id/tenant_id)
                       (token + verify)      ── /api/seed     ←── primes Neo4j for the persona
                                              ── /api/verify   ←── reads the 4 wow features back

 [ Browser ] ──WebRTC─►   [ LiveKit Cloud ]   ◄──registers──   [ agent.py worker ]
                                                                 │
                                                              [ Neo4j Aura ]
```

### What you set up once (all free tiers)

| Service | Where | What you copy into `.env` |
|---|---|---|
| **LiveKit Cloud** | https://cloud.livekit.io | `LIVEKIT_URL` (wss://your-project.livekit.cloud), `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` |
| **Neo4j Aura** (AuraDB Free) | https://neo4j.com/cloud/aura/ | `NEO4J_URI` (neo4j+s://...), `NEO4J_USER=neo4j`, `NEO4J_PASSWORD` |
| **OpenAI** | https://platform.openai.com | `OPENAI_API_KEY` |
| **ElevenLabs** | https://elevenlabs.io | `ELEVENLABS_API_KEY` + two voice IDs (default + calm) |
| **Deepgram** | https://console.deepgram.com | `DEEPGRAM_API_KEY` |

The full env-var reference with comments is in [.env.example](.env.example).

### Local run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env           # fill in the five service blocks above

# Two processes, both reading the same .env. Open two shells:

# Shell 1: the agent worker — registers with LiveKit Cloud, waits for calls.
python agent.py dev

# Shell 2: the web demo (FastAPI + the UI).
uvicorn web.server:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000. Pick a persona (the **Anil** one demonstrates all four wow
features), click **Seed graph**, click **Start call**, allow your mic. The agent opens by
asking about Anil's cancelled flight, in a calm voice, with short replies. Say *"I love
waiting on hold"* in a clearly agitated tone, hang up, wait ~5s, and the **Verify** panel
auto-refreshes showing PASS for each feature.

### Hosting it (so you can send a link)

Both processes can run on any platform that gives you a long-running container with
outbound HTTPS+WebSocket. The lightest path I'd recommend:

**Web service** (the FastAPI + UI) — Render *Web Service*:
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn web.server:app --host 0.0.0.0 --port $PORT`
- Env vars: the five blocks from `.env.example`. **Don't ship API secrets in git** —
  set them in Render's dashboard.

**Agent worker** (the LiveKit worker) — Render *Background Worker* (no public port):
- Start command: `python agent.py start`  (note: `start`, not `dev` — production mode)
- Same env vars as the web service.

That's it. Your browser hits the web URL, picks a persona, clicks Start call → the
browser connects directly to LiveKit Cloud → the worker (also connected to LiveKit Cloud)
joins the room → the cognitive layer runs against Neo4j Aura.

> ⚠️ Heads-up on costs: LiveKit Cloud's free tier gives ~10k minutes/mo. OpenAI is the
> meaningful per-call cost (`gpt-4o` for live + `gpt-4o-mini` for post-call extraction
> ≈ $0.005 / minute). Aura Free is fine for the prototype's scale.

### One-shot verification (CLI alternative to the UI)

Same flow without a browser — useful when you SSH into the worker after a demo and want
to inspect what landed:

```bash
python -m scripts.seed_demo        # seeds demo-anil
# (run the call however you want — Playground, web UI, your own client)
python -m scripts.verify_demo      # asserts PASS/FAIL on each wow feature; exits 0 on clean
```

---

## Optional: the rich acoustic engine

The librosa fallback is enough for the sarcasm demo above. For the *deeper* paralinguistic
signal (acoustic emotion tags, audio events, eGeMAPS biometrics, LLM contradiction pass):

```bash
pip install -r requirements-rich.txt   # funasr + opensmile + torch (heavy, GPU-friendly)
# AFFECT_EXTRACTOR=auto in .env → picks rich automatically when these deps are present
```

`analyze_audio()` returns the same `AcousticResult` either way; downstream code is
oblivious.

---

## Tests

```bash
pytest -q                              # 43 unit tests + 5 Neo4j integration tests
```

- **Unit (always run)**: prosody mapping, pre-call composition, the sanitizer +
  sarcasm-floor + coercion helpers, audio_capture WAV roundtrip, acoustic affect
  classification on a synthesized 220 Hz tone, post-call orchestration order +
  negative-event detection.
- **Integration (skipped when Neo4j isn't on `bolt://localhost:7687`)**: roundtrip
  affective state, surface negative event through `get_user_context`, sarcasm floor
  via `apply_facts`, and **the actual 70% decay** landing as `0.27` on the old rel.

For the integration tier point `NEO4J_URI` at any reachable instance (Aura Free works) and re-run.

---

## Architecture, one diagram

```
[ User Device ] ──WebRTC──► [ LiveKit Server ] ──job──► PYTHON WORKER (agent.py)
                                  (Silero VAD, Deepgram STT,                │
                                   OpenAI LLM, ElevenLabs TTS)              │
                                                                            │
   on participant_connected ──────────────────────────────────────────────┤ PRE-CALL
                                                                            │  • Neo4j read: affective_state, conversation_style,
                                                                            │    top-trusted facts, last severe (:Event)
                                                                            │  • Compose: system_prompt + voice_id + voice_settings + greeting
                                                                            │  • AgentSession(vad, stt, llm, tts).start(); session.say(greeting)
                                                                            │
   (live call: rtc.AudioStream → caller_audio.wav; session.on(interrupt_event) → counter++)
                                                                            │
   on participant_disconnected ───────────────────────────────────────────┤ POST-CALL (fire-and-forget)
                                                                            │  • acoustic_engine.analyze_audio() in asyncio.to_thread
                                                                            │  • graph.extract_facts_to_triplets() (LLM, to_thread)
                                                                            │  • graph.apply_facts() ← SARCASM FLOOR + 70% DECAY land here
                                                                            │  • graph.record_affective_state() ← writes (:User) props + (:Event)
                                                                            ▼
                                                                  [ NEO4J GRAPH DB ]
                          (:User {tenant_id,id, conversation_style, emotion, valence, arousal, paralinguistics})
                          (:User)-[:EXPERIENCED]->(:Event {kind, summary, severity, ts})
                          (:Entity {user_id, id})-[REL_TYPE {predicate_raw, trust_score, reasoning, updated_at, superseded_at?}]->(:Entity {user_id, id})
```

---

## Status

- ✅ **Step 1** — scaffold + archive
- ✅ **Step 2** — `graph_engine.py` (Neo4j async + sarcasm/trust/decay policies) + `audio_capture.py`
- ✅ **Step 3** — `acoustic_engine.py` (librosa + SenseVoice optional) + `post_call.py` + disconnect wiring
- ✅ **Step 4** — `scripts/seed_demo.py` + `scripts/verify_demo.py` + this README; **43 unit tests pass**, 5 integration tests gated on a live DB
- ✅ **Step 5** — Hosted demo UI ([web/server.py](web/server.py) + [web/static/index.html](web/static/index.html)) with LiveKit token mint, persona seed, live-call WebRTC, and a Verify panel that polls the graph after disconnect; Aura-ready (`neo4j+s://`); deployment recipe for Render
