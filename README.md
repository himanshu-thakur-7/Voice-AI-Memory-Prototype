# LiveKit Cognitive Voice AI Agent

A voice agent that **remembers callers, reads their tone, and adapts how it sounds and
what it says** — built on **LiveKit Agents** (UDP transport, VAD/STT/TTS plumbing) so 100%
of our engineering goes into the *brain*:

- **Neo4j** graph memory (entity-resolved facts + trust scores per relationship)
- **Acoustic engine** (librosa primary; SenseVoice + openSMILE optional)
- **LLM contradiction & sarcasm filter** that cross-references *what was said* vs *how it sounded*

> **Pivot note (Jun 2026).** The previous Twilio / Go-scheduler / FastAPI prototype is in
> `archive/`. Low-level WebRTC is a commodity problem Ringg AI already solves world-class;
> the value here is the cognitive memory layer.

## The 4 "wow" features

| # | Feature | Where it lives |
|---|---|---|
| 1 | **Sarcasm & Truth Filter** | `memory/post_call.py` + `graph_engine.apply_facts` — positive text + agitated tone → `trust_score = 0.2`, `"Likely Sarcastic"` reasoning |
| 2 | **Adaptive Verbosity** | Interruption counter → `conversation_style="impatient"` → `pre_call` injects *"10 words or less"* |
| 3 | **Proactive Empathy** | `get_user_context` surfaces last severe negative event → `session.say()` opens with it |
| 4 | **Dynamic Prosody** | `prosody.to_elevenlabs_voice` — frustrated/angry → calm voice, `stability=0.95` |

## Layout

```
agent.py                 # LiveKit worker entrypoint (1.x AgentSession)
config.py                # pydantic-settings, .env loader
audio_capture.py         # rtc.AudioStream → WAV   (Step 2)
memory/
  schemas.py             # AffectiveState, Assertion, ParticipantContext, PrecallResult, …
  prosody.py             # ProsodyProfile + to_elevenlabs_voice
  pre_call.py            # build_precall_context (Dynamic Prosody + Verbosity + Empathy)
  graph_engine.py        # Neo4j async + sarcasm/contradiction/decay   (Step 2)
  acoustic_engine.py     # analyze_audio (librosa + optional rich)     (Step 3)
  post_call.py           # process_post_call / schedule_post_call      (Step 3)
docker-compose.yml       # neo4j + redis + livekit-server (dev)
requirements.txt
requirements-rich.txt    # optional SenseVoice / openSMILE / torch
archive/                 # previous Twilio/Go/FastAPI prototype (salvage source)
```

## Quick start

```bash
docker compose up -d                 # neo4j (browser :7474), redis, livekit-server (dev)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                 # fill OPENAI/ELEVENLABS/DEEPGRAM (LiveKit dev keys preloaded)
python agent.py dev                  # registers with LIVEKIT_URL
```

Open the **LiveKit Agents Playground**, join the dev room with metadata
`{"user_id":"u1","tenant_id":"t1"}`. With no graph state seeded yet, the agent behaves as a
stock voice assistant. After Steps 2-3, seeding a frustrated `AffectiveState` + a
`Flight_Cancellation` event makes it open with the empathy line in the calm voice and
brief replies.

## Status

- ✅ **Step 1** — scaffold (`config`, `schemas`, `prosody`, `pre_call`, `agent`, deps, compose, .env)
- ☐ **Step 2** — `graph_engine.py` (Neo4j async) + `audio_capture.py`
- ☐ **Step 3** — `acoustic_engine.py` + `post_call.py` + disconnect wiring
- ☐ **Step 4** — tests + end-to-end demo
