# Advanced Voice AI Prototype

A low-latency, telephony-grade voice agent: **Twilio ⇄ Go Fair-Share orchestrator ⇄ Python streaming engine**,
with an **Affective Memory** layer that adapts prosody to the caller's emotional state and self-corrects stale
facts after every call.

Built as a focused demonstration of the pieces an all-in-one voice platform typically *doesn't* expose: an
explicit concurrency orchestrator, a swappable dual engine, and a contradiction-aware memory loop.

```
 Twilio ──webhook──► Go Scheduler ──gRPC──► Python backend ──ws──► STT ─► LLM ─► TTS ──► Twilio
                      (fair-share)                    │
                                          Affective Memory (pre-call prosody + post-call contradiction engine)
```

## Design honesty up front

This prototype refuses to repeat three common-but-wrong claims (see [`docs/latency-budget.md`](docs/latency-budget.md)):

1. **Sub-500ms is not reliable on a hand-rolled cascade** (~700ms–1.2s p50 is honest). Genuine sub-second is the
   job of the **realtime speech-to-speech engine**, which we also ship — flip `ENGINE=realtime`.
2. **OpenAI "Whisper streaming" doesn't exist** (`whisper-1` is batch). The default streaming STT is
   **OpenAI realtime transcription** (`gpt-4o-transcribe`) — it **reuses your OpenAI key**, so no separate STT
   account. Swappable to **Deepgram**, **ElevenLabs Scribe v2**, or the access-gated **Ringg Parrot** adapter.
3. **ElevenLabs `voice_settings` are locked at socket init**, so dynamic prosody uses the **multi-context** socket.

## Two engines, one interface

| `ENGINE=cascade` (default) | `ENGINE=realtime` |
|---|---|
| Twilio → STT → GPT-4o (stream) → ElevenLabs Flash (`ulaw_8000`) | OpenAI Realtime API, `g711_ulaw` in/out |
| Per-stage control, branded voice, fine prosody | Lowest latency, native VAD + barge-in |
| ~800ms p50 target | sub-second target |

Both implement `app/engines/base.py:VoiceEngine`, so `main.py` is engine-agnostic.

## Stack

- **Telephony:** Twilio bidirectional Media Streams (`<Connect><Stream>`), µ-law 8k / 20ms frames.
- **Orchestrator:** Go — webhook, Fair-Share scheduler (channels/goroutines), gRPC, TwiML. *Heavily commented for
  Go onboarding.*
- **Engine:** Python 3.13 / FastAPI / asyncio.
- **STT:** OpenAI realtime transcription (default, reuses OpenAI key) · Deepgram Nova-3 · ElevenLabs Scribe v2 · Ringg Parrot (access-gated, optional).
- **LLM:** GPT-4o (streaming; `LLM_MODEL` configurable).
- **TTS:** ElevenLabs `eleven_flash_v2_5`, `ulaw_8000`, multi-context.
- **Memory:** Mem0 / Mem0g (ADD/UPDATE/DELETE/NOOP) over Postgres+pgvector and FalkorDB.

## Quickstart

```bash
cp .env.example .env          # an OPENAI_API_KEY alone powers both STT + LLM; add ELEVENLABS for real TTS
make proto                    # generate gRPC stubs (needs protoc + grpcio-tools)
make up                       # docker compose: scheduler + backend + postgres + falkordb + redis
make test                     # go + python test suites
make demo                     # replay a recorded call into /media and print the latency breakdown
make ui                       # Ringg-styled test console → http://localhost:8000
```

## Test console

`make ui` serves a small **Ringg-styled dashboard** (`backend/app/static/index.html`) that exercises the *real*
system in a browser — no Twilio needed, and fully functional with no API keys (mock providers):

- **Assistants** — the Cascade and Realtime engines as cards, with live/mock provider badges.
- **Chat test** — type to the agent; each turn runs the real pre-call prosody hook → LLM → TTS, returns the
  reply + playable audio + **live-measured latency**.
- **Caller emotional state** — seed an emotion and watch the agent's prosody adapt on the next turn (the affective
  feedback loop, made interactive).
- **Analyze call** — runs the same **Contradiction Engine** the post-call worker uses; teach a fact (“my plan is
  pro”), change it (“…enterprise”), and watch it resolve **ADD → UPDATE**. Memory is session-scoped, so every
  reload is a clean slate.

Point a Twilio number's Voice webhook at `https://<your-ngrok-host>/voice` and set
`MEDIA_WS_URL=wss://<your-ngrok-host>/media`.

## Layout

```
proto/        gRPC contract (Go ⇄ Python)
scheduler/    Module 1 — Go orchestrator
backend/      Modules 2 & 3 — Python engine + memory
docs/         architecture + the honest latency budget
```

## Notes for reviewers

- **STT reuses the OpenAI key** (`backend/app/providers/stt_openai.py`, `gpt-4o-transcribe` realtime WebSocket) —
  no separate STT account. The access-gated **Ringg Parrot** adapter (`stt_ringg.py`) is kept as a ready-to-finish
  drop-in (only its transport methods are TODO) for whoever has Ringg access.
- The contradiction engine wraps **Mem0**, whose extract→update phase *is* the ADD/UPDATE/DELETE/NOOP loop; a thin
  deterministic guard logs and can force-overwrite flagged keys since the LLM step is probabilistic.
