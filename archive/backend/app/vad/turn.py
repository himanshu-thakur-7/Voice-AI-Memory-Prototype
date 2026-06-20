"""Voice-activity detection + turn endpointing.

This is a deliberately small, dependency-free **energy** VAD so the prototype runs and
its barge-in path is testable offline. In production you would fuse this with the STT
provider's semantic end-of-turn (Deepgram Flux / AssemblyAI Universal-Streaming) or a
neural VAD (Silero) — see docs/latency-budget.md, where endpointing is identified as the
single biggest (and most-hidden) latency lever.

The state machine emits four events the engines care about:
    SILENCE      — nothing happening
    SPEECH_START — the caller just began speaking  (→ barge-in if the bot is talking)
    SPEECH       — the caller is still speaking
    SPEECH_END   — the caller has paused long enough to be considered "done" (endpoint)
"""

from __future__ import annotations

from app.telephony.audio import FRAME_MS, rms_energy, ulaw_to_pcm16

SILENCE = "silence"
SPEECH_START = "speech_start"
SPEECH = "speech"
SPEECH_END = "speech_end"


class EnergyVAD:
    def __init__(
        self,
        *,
        threshold: float = 0.025,
        start_frames: int = 3,     # ~60 ms of voice to declare speech start
        silence_ms: int = 500,     # trailing silence to declare end-of-turn
    ) -> None:
        self.threshold = threshold
        self.start_frames = start_frames
        self.silence_frames = max(1, silence_ms // FRAME_MS)
        self._speaking = False
        self._active_run = 0
        self._silent_run = 0

    @property
    def speaking(self) -> bool:
        return self._speaking

    def update(self, ulaw_frame: bytes) -> str:
        energy = rms_energy(ulaw_to_pcm16(ulaw_frame))
        active = energy >= self.threshold

        if not self._speaking:
            if active:
                self._active_run += 1
                if self._active_run >= self.start_frames:
                    self._speaking = True
                    self._silent_run = 0
                    return SPEECH_START
            else:
                self._active_run = 0
            return SILENCE

        # currently in a speech turn
        if active:
            self._silent_run = 0
            return SPEECH
        self._silent_run += 1
        if self._silent_run >= self.silence_frames:
            self._speaking = False
            self._active_run = 0
            return SPEECH_END
        return SPEECH
