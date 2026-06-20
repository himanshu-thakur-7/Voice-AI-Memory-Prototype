"""Per-turn latency instrumentation.

The honest latency budget (docs/latency-budget.md) lives or dies on three numbers:
STT end-of-turn, LLM time-to-first-token, and TTS time-to-first-byte. TurnTimer
captures them per conversational turn so the simulator can print real measurements
instead of trusting vendor brochures.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


def _now_ms() -> float:
    return time.monotonic() * 1000.0


@dataclass
class TurnTimer:
    """Stopwatch for a single user→bot turn. All times are ms relative to user EOT."""

    user_eot_ms: float = field(default_factory=_now_ms)
    stt_final_ms: float | None = None
    llm_first_token_ms: float | None = None
    tts_first_byte_ms: float | None = None
    first_audio_to_caller_ms: float | None = None

    def mark_stt_final(self) -> None:
        self.stt_final_ms = _now_ms() - self.user_eot_ms

    def mark_llm_first_token(self) -> None:
        if self.llm_first_token_ms is None:
            self.llm_first_token_ms = _now_ms() - self.user_eot_ms

    def mark_tts_first_byte(self) -> None:
        if self.tts_first_byte_ms is None:
            self.tts_first_byte_ms = _now_ms() - self.user_eot_ms

    def mark_first_audio_out(self) -> None:
        if self.first_audio_to_caller_ms is None:
            self.first_audio_to_caller_ms = _now_ms() - self.user_eot_ms

    def summary(self) -> dict[str, float | None]:
        return {
            "stt_final_ms": _round(self.stt_final_ms),
            "llm_ttft_ms": _round(self.llm_first_token_ms),
            "tts_first_byte_ms": _round(self.tts_first_byte_ms),
            "mouth_to_ear_ms": _round(self.first_audio_to_caller_ms),
        }


def _round(v: float | None) -> float | None:
    return round(v, 1) if v is not None else None


# Optional collection sink: the simulator (tests/sim_twilio.py) sets this to gather
# per-turn latency summaries and print the honest measured breakdown.
_SINK: list[dict] | None = None


def use_sink(sink: list[dict] | None) -> None:
    global _SINK
    _SINK = sink


def record(summary: dict) -> None:
    if _SINK is not None:
        _SINK.append(summary)
