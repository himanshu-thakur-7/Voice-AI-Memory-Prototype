"""Twilio Media Streams simulator.

Replays a synthetic µ-law conversation through the streaming engine IN-PROCESS (no real
Twilio, no API keys) and prints the measured per-turn latency breakdown, so you can see
the pipeline plumbing and compare it against docs/latency-budget.md.

    python -m tests.sim_twilio --turns 3 --engine cascade --print-latency

NOTE: with the mock providers these numbers reflect the *plumbing* (queueing, framing,
chunking, VAD endpointing), not real model latency. Plug in real keys to measure the real
STT EOT / LLM TTFT / TTS first-byte.
"""

from __future__ import annotations

import argparse
import asyncio

from app import metrics
from app.config import Settings
from app.engines.base import INBOUND_EOF, CallContext, Speaker
from app.engines.cascade import CascadeEngine
from app.engines.realtime import RealtimeEngine
from app.providers.factory import make_llm, make_stt, make_tts
from tests.util import utterance


async def _drive(turns: int, engine_name: str) -> list[dict]:
    sink: list[dict] = []
    metrics.use_sink(sink)
    settings = Settings()

    sent: list[str] = []

    async def send(t: str) -> None:
        sent.append(t)

    ctx = CallContext(call_sid="CA-sim", stream_sid="MZ-sim", engine=engine_name,
                      tenant_id="demo", user_id="caller-1")
    speaker = Speaker(send, ctx.stream_sid, pace=False)

    if engine_name == "realtime":
        engine = RealtimeEngine(settings)
    else:
        engine = CascadeEngine(settings, make_stt(settings), make_llm(settings), make_tts(settings))

    inbound: asyncio.Queue = asyncio.Queue()
    for _ in range(turns):
        for frame in utterance():
            inbound.put_nowait(frame)
    inbound.put_nowait(INBOUND_EOF)

    speaker_task = asyncio.create_task(speaker.run())
    await engine.run(inbound, speaker, ctx)
    await asyncio.sleep(0.05)
    speaker_task.cancel()

    media = sum(1 for s in sent if '"media"' in s)
    print(f"\n[sim] engine={engine_name}  turns={turns}  "
          f"outbound_media_frames={media}  marks={sum(1 for s in sent if chr(34)+'mark'+chr(34) in s)}")
    return sink


def _print_latency(rows: list[dict]) -> None:
    if not rows:
        print("[sim] no turns measured")
        return
    cols = ["stt_final_ms", "llm_ttft_ms", "tts_first_byte_ms", "mouth_to_ear_ms"]
    print("\n  turn | " + " | ".join(f"{c:>17}" for c in cols))
    print("  " + "-" * 78)
    for i, r in enumerate(rows):
        print(f"  {i:>4} | " + " | ".join(f"{str(r.get(c)):>17}" for c in cols))
    print("\n  (mock-path plumbing only — see docs/latency-budget.md for the honest real-vendor budget)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--engine", choices=["cascade", "realtime"], default="cascade")
    ap.add_argument("--print-latency", action="store_true")
    args = ap.parse_args()

    rows = asyncio.run(_drive(args.turns, args.engine))
    if args.print_latency:
        _print_latency(rows)


if __name__ == "__main__":
    main()
