# Architecture

Three processes, one call.

```
 PSTN caller
     │  (1) inbound call
     ▼
  Twilio ──(2) HTTP POST /voice──►  Go Scheduler  ── Module 1 ─────────────────────┐
     ▲                                  │  Fair-Share admission (channels/goroutines)│
     │                                  │  (3) gRPC RegisterCall(CallContext) ───────┼──► Python backend
     │  (5) TwiML <Connect><Stream>  ◄──┘  (4) returns wss media URL                 │      (warms sockets,
     │                                                                               │       runs pre-call lookup)
     │  (6) open bidirectional Media Stream (wss, µ-law 8k / 20ms frames)
     ▼
  Python FastAPI  ── Module 2 ──────────────────────────────────────────────────────
     │   /media WebSocket
     │   ┌──────────────── VoiceEngine (selected per call) ─────────────────┐
     │   │  CascadeEngine   : STT ─► GPT-4o(stream) ─► ElevenLabs(ulaw_8000) │
     │   │  RealtimeEngine  : OpenAI Realtime API (g711_ulaw in/out)         │
     │   └──────────────────────────────────────────────────────────────────┘
     │   barge-in: VAD → cancel LLM → flush TTS → Twilio `clear` → truncate to last `mark`
     │
     └── on WS close → enqueue post-call job ── Module 3 ──────────────────────────────
            paralinguistic extraction — rich engine (SenseVoice acoustic emotion + audio
            events, openSMILE eGeMAPS biometrics, librosa temporal/pitch, + LLM
            "contradiction" pass: words-vs-voice → final_affective_state); falls back to a
            dependency-free heuristic when the heavy deps aren't installed
            + Contradiction Engine (Mem0 / Mem0g: ADD / UPDATE / DELETE / NOOP)
            over pgvector (semantic) + FalkorDB (per-user graph)  → updates AffectiveState
                                              │
              feeds the NEXT call's pre-call prosody injection ◄┘  (the affective feedback loop)
```

## Why three processes

- **Go owns concurrency & admission.** A telephony front door is a fan-in of many simultaneous calls; Go's
  goroutines + channels are the right tool for fair-share scheduling and backpressure. Keeping it separate means
  the latency-sensitive Python media loop is never blocked by admission logic.
- **Python owns the ML media loop.** The STT/LLM/TTS providers all ship first-class async Python SDKs; the
  streaming pipeline is naturally expressed with `asyncio`.
- **gRPC is the seam.** Strongly-typed `CallContext` hand-off, and it lets the backend warm provider sockets +
  run the pre-call memory lookup *before* Twilio opens the media stream — removing a cold-start penalty from the
  first, most-noticed turn.

## Module map

| Module | Process | Entry file | What it does |
|---|---|---|---|
| 1 — Scheduler/Orchestrator | Go | `scheduler/cmd/scheduler/main.go` | Webhook, Fair-Share scheduler, TwiML, gRPC client |
| 2 — Streaming engine | Python | `backend/app/main.py` | `/media` WS loop, dual VoiceEngine, barge-in |
| 3 — Affective memory | Python | `backend/app/workers/postcall.py`, `backend/app/memory/precall.py` | Pre-call prosody, post-call extraction + contradiction engine |

See [latency-budget.md](latency-budget.md) for the honest latency posture.
