"""CascadeEngine — Twilio → STT → GPT-4o → ElevenLabs, streaming and overlapped.

The whole point is that NOTHING waits for a full result:
  • a ``fanout`` task feeds every inbound frame to both the VAD (barge-in) and the STT;
  • when STT endpoints a turn, ``_respond`` streams LLM tokens, cuts them into sentences,
    and pipes each sentence into TTS — so the caller hears the first words while the model
    is still composing the rest;
  • a barge-in (caller speaks while the bot is talking) cancels generation, flushes TTS,
    and clears Twilio's buffer.

See docs/latency-budget.md for why this overlap (not the model speeds) is what earns the
latency target — and why the honest target is ~800 ms p50, not sub-500 ms.
"""

from __future__ import annotations

import asyncio

from app import metrics
from app.config import Settings
from app.engines.base import (
    INBOUND_EOF,
    CallContext,
    Speaker,
    VoiceEngine,
    iter_queue,
)
from app.logging import get_logger
from app.metrics import TurnTimer
from app.providers.llm_openai import LLM
from app.providers.stt_base import STTProvider
from app.providers.tts_elevenlabs import TTS
from app.vad.turn import SPEECH_START, EnergyVAD

log = get_logger(__name__)

GREETING = "Hi, thanks for calling. How can I help you today?"


class CascadeEngine(VoiceEngine):
    name = "cascade"

    def __init__(self, settings: Settings, stt: STTProvider, llm: LLM, tts: TTS) -> None:
        self._settings = settings
        self._stt = stt
        self._llm = llm
        self._tts = tts
        #: Public so main.py can hand the transcript to the post-call worker (Module 3b).
        self.history: list[dict] = []

    async def run(self, inbound: asyncio.Queue, speaker: Speaker, ctx: CallContext) -> None:
        self.history = [{"role": "system", "content": self._system_prompt(ctx)}]
        stt_q: asyncio.Queue = asyncio.Queue()
        barge_in = asyncio.Event()

        log.info("cascade.start", call=ctx.call_sid, stt=self._stt.name,
                 llm=self._llm.name, tts=self._tts.name, prosody=ctx.prosody.label)

        # The bot greets first (typical for outbound/inbound IVR feel).
        await self._speak(GREETING, speaker, ctx, barge_in, TurnTimer(), record=True)

        async def fanout() -> None:
            """Fan each inbound frame to the VAD (barge-in) and the STT input."""
            vad = EnergyVAD()
            while True:
                frame = await inbound.get()
                if frame is INBOUND_EOF:
                    await stt_q.put(INBOUND_EOF)
                    return
                if vad.update(frame) == SPEECH_START and speaker.speaking.is_set():
                    barge_in.set()  # caller cut in over the bot
                await stt_q.put(frame)

        async def converse() -> None:
            async for t in self._stt.stream(iter_queue(stt_q)):
                if t.is_final and t.text.strip():
                    await self._respond(t.text, speaker, ctx, barge_in)

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(fanout())
                tg.create_task(converse())
        except* Exception as eg:  # TaskGroup wraps child errors in an ExceptionGroup
            for exc in eg.exceptions:
                log.error("cascade.task_error", call=ctx.call_sid, err=str(exc))
        log.info("cascade.end", call=ctx.call_sid)

    async def _respond(
        self, user_text: str, speaker: Speaker, ctx: CallContext, barge_in: asyncio.Event
    ) -> None:
        timer = TurnTimer()       # clock starts at user end-of-turn
        timer.mark_stt_final()    # STT already finalized this turn
        barge_in.clear()
        self.history.append({"role": "user", "content": user_text})
        log.info("turn.user", call=ctx.call_sid, text=user_text)

        produced: list[str] = []
        turn = len(self.history)
        async for i, sentence in _enumerate(self._llm.stream_sentences(self.history, timer)):
            if barge_in.is_set():
                break
            produced.append(sentence)
            async for ulaw in self._tts.synthesize(sentence, ctx.prosody, timer):
                if barge_in.is_set():
                    break
                await speaker.play(ulaw)
                timer.mark_first_audio_out()
            await speaker.mark(f"{ctx.call_sid}:{turn}:{i}")

        # Record only what we actually said. A fuller implementation would trim this to
        # speaker.last_acked_mark — i.e. the audio Twilio confirms the caller HEARD.
        if produced:
            self.history.append({"role": "assistant", "content": " ".join(produced)})
        if barge_in.is_set():
            await speaker.clear()  # flush Twilio's buffered audio → crisp interruption
            log.info("turn.barge_in", call=ctx.call_sid, heard_mark=speaker.last_acked_mark)
        summary = timer.summary()
        log.info("turn.latency", call=ctx.call_sid, **summary)
        metrics.record(summary)

    async def _speak(
        self, text: str, speaker: Speaker, ctx: CallContext,
        barge_in: asyncio.Event, timer: TurnTimer, *, record: bool,
    ) -> None:
        """Speak a bot-initiated line (e.g. the greeting) with the same barge-in rules."""
        async for ulaw in self._tts.synthesize(text, ctx.prosody, timer):
            if barge_in.is_set():
                break
            await speaker.play(ulaw)
        await speaker.mark(f"{ctx.call_sid}:greeting")
        if record:
            self.history.append({"role": "assistant", "content": text})

    def _system_prompt(self, ctx: CallContext) -> str:
        if ctx.system_prompt:
            return ctx.system_prompt
        base = self._settings.base_system_prompt
        if ctx.prosody.system_prompt_suffix:
            base = f"{base} {ctx.prosody.system_prompt_suffix}"
        return base


async def _enumerate(aiter):  # noqa: ANN001
    i = 0
    async for item in aiter:
        yield i, item
        i += 1
