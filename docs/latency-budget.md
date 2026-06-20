# Latency Budget — the honest version

> **TL;DR:** A hand-rolled, multi-vendor cascade (Twilio → STT → GPT-4o → ElevenLabs → Twilio) does **not**
> reliably hit sub-500ms measured from *"user stops speaking → first audio back."* Budget **~700ms–1.2s p50**.
> Genuine sub-second comes from a **speech-to-speech** model (OpenAI Realtime API) that collapses STT+LLM+TTS into
> one hop. We ship both engines so the trade-off is a config flag, not a rewrite.

## Why the spec's "sub-500ms" is misleading

The metric *"user stops speaking → first TTS audio"* **includes the endpointing/turn-detection silence wait** —
the 300–500ms of silence a system must observe to be confident the user actually finished. Almost every vendor
"sub-300ms" / "sub-500ms" number starts the clock *after* that wait and is measured *platform-side* (excluding the
carrier + user network legs). So the number a caller *feels* is routinely ~500ms higher than the brochure.

## Cascade budget (p50, well-tuned, streaming everywhere)

| Stage | Typical | Notes |
|---|---:|---|
| Endpointing / turn-detection silence wait | 300–500 ms | **The dominant, usually-hidden cost.** Tunable but trades against false cut-offs. |
| µ-law decode + resample to STT rate | <5–25 ms | Negligible. Don't optimize here. |
| STT final transcript after endpoint | 50–250 ms | **OpenAI `gpt-4o-transcribe` (default, reuses your OpenAI key) ~sub-150ms**; Deepgram Nova-3 ~247ms TTFS; ElevenLabs Scribe v2 ~150ms; Ringg Parrot ~60ms (their figure, access-gated). |
| Network hops (orchestrator → LLM) | 30–100 ms | Collapses with colocation. |
| **GPT-4o time-to-first-token** | **400–760 ms** | The other big one. p95 can be 800–1200ms. `gpt-4o-mini` / newer models are faster. |
| LLM → TTS hop + ElevenLabs Flash first audio | 75–200 ms | `eleven_flash_v2_5`, `ulaw_8000`, ~75ms inference + network. |
| Re-packetize to 20ms µ-law + jitter buffer | 30–95 ms | Keep your own jitter buffer small. |
| **Realistic total (p50)** | **~700 ms – 1.2 s** | Aggressive best case ~600–800ms; **p95 routinely 1.2–1.5s**. |

**The two levers that matter:** endpointing and LLM TTFT. STT/TTS/transcoding are not where the time hides.

## Realtime engine budget

OpenAI Realtime API: ~500ms TTFB (US), ~800ms voice-to-voice is the *good* target. STT, reasoning, TTS, VAD and
barge-in live inside one streaming model, so you skip the serial stacking. Accepts `g711_ulaw` → drops onto Twilio
Media Streams directly. Still qualify it: "reliably sub-500ms at p95 including a real VAD threshold + two telephony
legs" remains aggressive — commit to **sub-800ms**, treat sub-500ms as a best-case median.

## How this repo earns its latency

- **Stream and overlap every stage** — STT partials → first sentence/clause → GPT-4o `stream=true` → tokens
  chunked at sentence boundaries → ElevenLabs `stream-input`. Nothing waits for a full result. (`engines/cascade.py`)
- **Zero-resample audio path** — ElevenLabs emits `ulaw_8000`; Twilio inbound µ-law feeds STT directly where the
  provider supports it (Scribe, Ringg). (`telephony/audio.py`)
- **Warm sockets** — provider WebSockets are kept open per call; no per-turn TLS handshake.
- **Tight LLM turns** — short/cached system prompt, low `max_completion_tokens`. (`providers/llm_openai.py`)
- **Honest measurement** — `tests/sim_twilio.py` prints the real per-stage breakdown so we validate against this
  table instead of trusting brochures.

## A note on Ringg's `<400ms`

Ringg AI advertises `<400ms` — credible because it's a **single, colocated, all-in-one platform** (telephony + STT
+ LLM + TTS under one roof), the same reason Twilio ConversationRelay can quote `<0.5s` for its *platform turn gap*.
That is a different measurement than four independent vendors across the public internet. The lesson this repo
encodes: if you need that number, you either own the whole stack (Ringg's model) or use a single speech-to-speech
model (our realtime engine) — you don't get it by stitching.

## Sources

Twilio core-latency guide · Twilio ConversationRelay · Deepgram streaming-latency & endpointing docs ·
AssemblyAI Universal-Streaming · ElevenLabs latency docs + Flash announcement · OpenAI Realtime + prompt-caching
docs · Artificial Analysis GPT-4o providers · Daily.co STT benchmark · LiveKit turn-detection docs ·
ringg.ai/models/speech-to-text/v1. (Full URLs captured in the research pass that produced this doc.)
