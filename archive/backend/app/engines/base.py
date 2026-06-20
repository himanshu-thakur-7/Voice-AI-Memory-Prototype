"""Engine-facing primitives shared by the cascade and realtime engines.

The session in ``main.py`` is engine-agnostic: it builds a CallContext, a Speaker
(outbound audio), and an inbound audio queue, then hands them to whichever
``VoiceEngine`` is selected. Everything an engine needs to talk to the caller lives
here, so the engines themselves stay focused on the STT→LLM→TTS choreography.
"""

from __future__ import annotations

import abc
import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field

from app.telephony import twilio_protocol as proto
from app.telephony.audio import FRAME_MS, frame_ulaw

# Sentinel pushed onto the inbound queue when the caller's media stream ends.
INBOUND_EOF = None


@dataclass
class ProsodyProfile:
    """How the bot should *sound*. Selected by the pre-call affective lookup.

    On ElevenLabs these map to voice_settings (locked at TTS-context init — see
    providers/tts_elevenlabs.py); on the realtime engine they become instruction text.
    """

    label: str = "neutral"
    stability: float = 0.5          # lower = more expressive / wider emotional range
    style: float = 0.0              # >0 amplifies emotional intensity (adds latency)
    similarity_boost: float = 0.75
    speed: float = 1.0              # 0.7..1.2
    system_prompt_suffix: str = ""  # appended to the LLM system prompt
    realtime_instructions: str = "" # used by the realtime engine instead of voice_settings

    @classmethod
    def neutral(cls) -> ProsodyProfile:
        return cls()


@dataclass
class CallContext:
    call_sid: str = ""
    stream_sid: str = ""
    from_number: str = ""
    to_number: str = ""
    tenant_id: str = "default"
    user_id: str = ""
    engine: str = "cascade"
    affective_hint: str = ""
    custom: dict[str, str] = field(default_factory=dict)
    # Filled by the pre-call hook (Module 3a):
    prosody: ProsodyProfile = field(default_factory=ProsodyProfile.neutral)
    system_prompt: str = ""


# Async function that writes a text frame to the WebSocket.
WSSend = Callable[[str], Awaitable[None]]


class Speaker:
    """Outbound audio pump with first-class barge-in.

    Engines call ``play()`` with µ-law chunks; ``run()`` re-packetises them into paced
    20 ms frames and writes them to Twilio. ``clear()`` implements barge-in: it bumps a
    generation counter (so the frame loop abandons the chunk it is mid-way through),
    drops everything still queued on our side, and sends Twilio a ``clear`` to flush the
    audio already buffered on *their* side.
    """

    def __init__(self, ws_send: WSSend, stream_sid: str, *, pace: bool = True) -> None:
        self._send = ws_send
        self._stream_sid = stream_sid
        self._pace = pace
        self._q: asyncio.Queue[_Item] = asyncio.Queue()
        self._gen = 0
        #: Set while the bot has audio queued or in flight. The VAD consults this to
        #: decide whether inbound speech is a *barge-in* (vs. the start of a normal turn).
        self.speaking = asyncio.Event()
        #: Name of the most recent mark Twilio confirmed as played — i.e. how much of
        #: the reply the caller actually heard. Used to truncate history on barge-in.
        self.last_acked_mark: str | None = None

    async def play(self, ulaw: bytes) -> None:
        if ulaw:
            self.speaking.set()
            await self._q.put(_Audio(ulaw))

    async def mark(self, name: str) -> None:
        await self._q.put(_Mark(name))

    async def clear(self) -> None:
        """Barge-in flush. Safe to call repeatedly."""
        self._gen += 1
        _drain(self._q)
        self.speaking.clear()
        await self._send(proto.clear_message(self._stream_sid))

    def ack_mark(self, name: str) -> None:
        """Called by the reader when Twilio echoes a 'mark' back to us."""
        self.last_acked_mark = name

    async def run(self) -> None:
        """Drives outbound audio until cancelled."""
        while True:
            item = await self._q.get()
            gen = self._gen
            try:
                if isinstance(item, _Audio):
                    for f in frame_ulaw(item.data):
                        if self._gen != gen:
                            break  # barge-in mid-chunk: drop the rest
                        await self._send(proto.media_message(self._stream_sid, f))
                        if self._pace:
                            await asyncio.sleep(FRAME_MS / 1000.0)
                elif isinstance(item, _Mark):
                    await self._send(proto.mark_message(self._stream_sid, item.name))
            finally:
                self._q.task_done()
            if self._q.empty():
                self.speaking.clear()


class VoiceEngine(abc.ABC):
    """A conversational engine. Implementations: CascadeEngine, RealtimeEngine."""

    name: str = "base"

    @abc.abstractmethod
    async def run(
        self,
        inbound: asyncio.Queue,
        speaker: Speaker,
        ctx: CallContext,
    ) -> None:
        """Consume inbound µ-law frames (bytes; ``INBOUND_EOF`` ends the stream) and
        drive the caller-facing conversation via ``speaker``. Returns when the call ends.
        """
        raise NotImplementedError


async def iter_queue(q: asyncio.Queue) -> AsyncIterator[bytes]:
    """Yield items from a queue until the EOF sentinel."""
    while True:
        item = await q.get()
        if item is INBOUND_EOF:
            return
        yield item


# ── internal queue item types ────────────────────────────────────────────────


@dataclass
class _Audio:
    data: bytes


@dataclass
class _Mark:
    name: str


_Item = _Audio | _Mark


def _drain(q: asyncio.Queue) -> None:
    while not q.empty():
        try:
            q.get_nowait()
            q.task_done()
        except asyncio.QueueEmpty:  # pragma: no cover
            break
