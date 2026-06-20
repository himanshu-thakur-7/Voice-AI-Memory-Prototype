"""Streaming STT interface + an offline mock.

The cascade engine feeds raw µ-law 20 ms frames in and consumes ``Transcript`` objects
out. Real providers (Deepgram, Ringg Parrot, ElevenLabs Scribe) do their own server-side
endpointing; the mock uses the local EnergyVAD so the loop has realistic turn-taking with
zero credentials.
"""

from __future__ import annotations

import abc
import itertools
from collections.abc import AsyncIterator
from dataclasses import dataclass

from app.vad.turn import SPEECH_END, SPEECH_START, EnergyVAD


@dataclass
class Transcript:
    text: str
    is_final: bool


class STTProvider(abc.ABC):
    name = "base"

    @abc.abstractmethod
    def stream(self, frames: AsyncIterator[bytes]) -> AsyncIterator[Transcript]:
        """Consume µ-law frames, yield partial/final transcripts (final == end-of-turn)."""
        raise NotImplementedError


class MockSTT(STTProvider):
    """Endpoints with the energy VAD and emits a canned utterance per turn.

    It cannot really transcribe (no model), but it reproduces the *timing* that matters
    for the streaming loop and barge-in: speech detected → silence → final transcript.
    """

    name = "mock"

    def __init__(self, phrases: list[str] | None = None) -> None:
        self._phrases = phrases or [
            "hi there, i have a question about my order",
            "actually, can you tell me my account balance",
            "thanks, that's all i needed",
        ]

    async def stream(self, frames: AsyncIterator[bytes]) -> AsyncIterator[Transcript]:
        vad = EnergyVAD()
        cycle = itertools.cycle(self._phrases)
        spoke = False
        async for frame in frames:
            ev = vad.update(frame)
            if ev == SPEECH_START:
                spoke = True
            elif ev == SPEECH_END and spoke:
                spoke = False
                yield Transcript(text=next(cycle), is_final=True)
