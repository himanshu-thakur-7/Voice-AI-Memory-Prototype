"""Rich post-call paralinguistic & tone analysis (SenseVoice + openSMILE + librosa + LLM).

The production-grade affective extractor for the post-call worker. It:
  1. transcribes with acoustic-emotion + audio-event tags (SenseVoice / FunAudioLLM),
  2. pulls eGeMAPS voice biometrics — jitter, shimmer, loudness, pitch (openSMILE),
  3. measures temporal/pitch dynamics — silence ratio, pauses, pitch trajectory (librosa),
  4. runs an LLM "contradiction" pass that reconciles *what was said* against *how it
     sounded* (cheerful words + tense voice → masked stress) into a `final_affective_state`.

Design rules that keep it consistent with the rest of the repo:
  • Heavy deps (funasr, opensmile, torch, librosa) are OPTIONAL and lazily imported. If any
    is missing, ``is_available()`` is False and the post-call worker falls back to the
    dependency-free heuristic in ``affective.py`` — the repo still runs and tests pass.
  • SenseVoice is offline/batch — correct for POST-call, never the realtime loop.
  • Models load once (singleton) — loading per call is very expensive.
  • The blocking model work is wrapped in ``asyncio.to_thread`` by the caller.

Adapted from the user's AffectiveMemoryExtractor.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import tempfile
from collections import Counter
from typing import Any

from app.config import Settings
from app.engines.base import CallContext
from app.logging import get_logger
from app.memory.schemas import AffectiveState, Emotion
from app.telephony.audio import ulaw_to_wav

log = get_logger(__name__)

_REQUIRED = ("funasr", "opensmile", "librosa", "torch")


def is_available() -> bool:
    """True if every heavy dependency is importable (without importing them)."""
    return all(importlib.util.find_spec(m) is not None for m in _REQUIRED)


# ── the extractor ────────────────────────────────────────────────────────────


class ParalinguisticExtractor:
    """Loads the SenseVoice + openSMILE models once and analyzes audio files."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._sv: Any = None
        self._smile: Any = None
        self._device = "cpu"
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        import opensmile  # lazy
        import torch  # lazy
        from funasr import AutoModel  # lazy

        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        log.info("paralinguistics.loading", device=self._device)
        self._sv = AutoModel(
            model="iic/SenseVoiceSmall",
            trust_remote_code=True,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=self._device,
            disable_update=True,
        )
        self._smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        self._loaded = True
        log.info("paralinguistics.loaded")

    def analyze(self, audio_path: str) -> dict | None:
        """Run the full dual acoustic/lexical pipeline. Returns the paralinguistics dict
        (mirrors memory_payload["paralinguistics"]) + transcript, or None on failure."""
        try:
            self._load()
            sv = self._sv.generate(input=audio_path, language="auto", use_itn=True, batch_size=64)
            raw_text = " ".join(r.get("text", "") for r in sv)
            transcript, base_emotion, events = _parse_sensevoice(raw_text)

            biometrics = self._opensmile(audio_path)
            temporal = _temporal_and_pitch(audio_path)
            acoustics = {**biometrics, **temporal}

            llm = self._contradiction(transcript, base_emotion, acoustics)
            return {
                "base_acoustic_emotion": base_emotion,
                "lexical_sentiment": llm.get("lexical_sentiment", "neutral"),
                "final_affective_state": llm.get("final_affective_state", base_emotion),
                "llm_reasoning": llm.get("reasoning", ""),
                "detected_events": sorted(set(events)),
                "acoustic_biometrics": acoustics,
                "transcript": transcript,
            }
        except Exception as e:  # noqa: BLE001 — any model/runtime failure → fall back
            log.warning("paralinguistics.failed", err=str(e))
            return None

    def _opensmile(self, audio_path: str) -> dict[str, float]:
        try:
            df = self._smile.process_file(audio_path)

            def get(name: str) -> float:
                cols = [c for c in df.columns if name.lower() in c.lower()]
                return float(df[cols[0]].iloc[0]) if cols else 0.0

            return {
                "loudness_mean": get("Loudness_sma3"),
                "pitch_mean": get("F0semitoneFrom27.5Hz"),
                "jitter_local": get("jitterLocal"),
                "shimmer_local": get("shimmerLocal"),
                "speaking_rate": get("equivalentSoundLevel"),
            }
        except Exception as e:  # noqa: BLE001
            log.warning("paralinguistics.opensmile_failed", err=str(e))
            return {}

    def _contradiction(self, transcript: str, acoustic_emotion: str, acoustics: dict) -> dict:
        """LLM pass reconciling transcript vs. acoustics into the true affective state."""
        if not transcript.strip() or not self._settings.openai_api_key:
            return {"lexical_sentiment": "neutral", "final_affective_state": acoustic_emotion,
                    "reasoning": "lexical LLM offline"}
        try:
            from openai import OpenAI  # lazy

            client = OpenAI(api_key=self._settings.openai_api_key)
            resp = client.chat.completions.create(
                model=self._settings.openai_lexical_model,
                response_format={"type": "json_object"},
                temperature=0.2,
                messages=[
                    {"role": "system", "content": _LEXICAL_SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"Transcript: '{transcript}'\n"
                        f"Acoustic Emotion: {acoustic_emotion}\n"
                        f"Jitter (Tension): {acoustics.get('jitter_local', 0)}\n"
                        f"Silence Ratio: {acoustics.get('silence_ratio', 0)}\n"
                        f"Pause Count: {acoustics.get('pause_count', 0)}\n"
                        f"Pitch Trajectory: {acoustics.get('pitch_trajectory', 'unknown')}"
                    )},
                ],
            )
            return json.loads(resp.choices[0].message.content or "{}")
        except Exception as e:  # noqa: BLE001
            log.warning("paralinguistics.lexical_failed", err=str(e))
            return {"lexical_sentiment": "neutral", "final_affective_state": acoustic_emotion,
                    "reasoning": "lexical error"}


_EXTRACTOR: ParalinguisticExtractor | None = None


def _extractor(settings: Settings) -> ParalinguisticExtractor:
    global _EXTRACTOR
    if _EXTRACTOR is None:
        _EXTRACTOR = ParalinguisticExtractor(settings)
    return _EXTRACTOR


# ── integration: bytes → AffectiveState ──────────────────────────────────────


async def run_paralinguistics(
    caller_ulaw: bytes, ctx: CallContext, settings: Settings
) -> AffectiveState | None:
    """Try the rich extractor on the captured caller audio. Returns an AffectiveState,
    or None if disabled / deps missing / no audio (caller should fall back)."""
    if settings.affect_extractor == "heuristic":
        return None
    if not caller_ulaw or not is_available():
        if settings.affect_extractor == "rich":
            log.warning("paralinguistics.unavailable", reason="deps missing or no audio")
        return None

    # All blocking work (temp-file write + model inference) runs off the event loop.
    profile = await asyncio.to_thread(_analyze_wav_bytes, settings, ulaw_to_wav(caller_ulaw))
    if not profile:
        return None
    return profile_to_state(profile, ctx)


def _analyze_wav_bytes(settings: Settings, wav_bytes: bytes) -> dict | None:
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        with open(path, "wb") as f:
            f.write(wav_bytes)
        return _extractor(settings).analyze(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def profile_to_state(profile: dict, ctx: CallContext) -> AffectiveState:
    """Map the rich paralinguistic profile onto our coarse AffectiveState."""
    base = profile.get("base_acoustic_emotion", "neutral")
    final = profile.get("final_affective_state", "")
    biometrics = profile.get("acoustic_biometrics", {})
    emotion = map_emotion(base, final)
    valence, arousal = _valence_arousal(emotion, biometrics)
    # A contradiction (lexical sentiment disagreeing with the acoustic read) is itself a
    # strong signal — we keep confidence high when the LLM produced a reconciled state.
    confidence = 0.75 if final and final != base else 0.6
    return AffectiveState(
        tenant_id=ctx.tenant_id, user_id=ctx.user_id, emotion=emotion,
        valence=valence, arousal=arousal, confidence=confidence,
        features={k: v for k, v in biometrics.items() if isinstance(v, int | float)},
        paralinguistics=profile,
    )


def map_emotion(base_acoustic_emotion: str, final_affective_state: str) -> Emotion:
    """Collapse SenseVoice's coarse emotion + the LLM's nuanced state into our enum.

    The nuanced ``final_affective_state`` (e.g. 'cynical_or_masking_grief',
    'tense_suppressed', 'psychopathic_threat') wins when present; otherwise we fall back to
    the acoustic label.
    """
    f = (final_affective_state or "").lower()
    buckets: list[tuple[tuple[str, ...], Emotion]] = [
        (("threat", "psychopath", "aggress", "rage", "hostil", "angry"), Emotion.ANGRY),
        (("grief", "defeat", "apath", "sad", "depress", "mourn", "despair"), Emotion.SAD),
        (("tense", "suppress", "stress", "anx", "panic", "fear", "nervous"), Emotion.ANXIOUS),
        (("frustrat", "annoy", "irritat", "cynical"), Emotion.FRUSTRATED),
        (("joy", "happy", "content", "cheer", "pleased", "elated"), Emotion.HAPPY),
    ]
    for keys, emotion in buckets:
        if any(k in f for k in keys):
            return emotion
    return {
        "happy": Emotion.HAPPY, "sad": Emotion.SAD,
        "angry": Emotion.ANGRY, "neutral": Emotion.NEUTRAL,
    }.get((base_acoustic_emotion or "").lower(), Emotion.NEUTRAL)


_VA_BASE = {
    Emotion.NEUTRAL: (0.0, 0.3), Emotion.FRUSTRATED: (-0.6, 0.7),
    Emotion.ANGRY: (-0.8, 0.85), Emotion.ANXIOUS: (-0.3, 0.6),
    Emotion.HAPPY: (0.7, 0.6), Emotion.SAD: (-0.5, 0.2),
}


def _valence_arousal(emotion: Emotion, biometrics: dict) -> tuple[float, float]:
    valence, arousal = _VA_BASE[emotion]
    jitter = float(biometrics.get("jitter_local", 0) or 0)  # vocal tension bumps arousal
    arousal = max(0.0, min(1.0, arousal + min(0.15, jitter * 2)))
    return round(valence, 3), round(arousal, 3)


# ── pure helpers (testable without the heavy deps) ───────────────────────────


def _parse_sensevoice(raw_text: str) -> tuple[str, str, list[str]]:
    """SenseVoice emits rich tags, e.g. '<|zh|><|NEUTRAL|><|Speech|> hi <|Laughter|>'."""
    emotions = re.findall(r"<\|(HAPPY|SAD|ANGRY|NEUTRAL)\|>", raw_text)
    dominant = Counter(e.lower() for e in emotions).most_common(1)[0][0] if emotions else "neutral"
    events = [e.lower() for e in re.findall(
        r"<\|(Laughter|Applause|Cough|Sneeze|Cry|Music)\|>", raw_text)]
    clean = re.sub(r"<\|.*?\|>", "", raw_text).strip()
    return clean, dominant, events


def _temporal_and_pitch(audio_path: str) -> dict[str, Any]:
    """librosa temporal features: silence ratio, pause count, pitch trajectory."""
    try:
        import librosa  # lazy
        import numpy as np  # lazy

        y, sr = librosa.load(audio_path, sr=16000)
        total = librosa.get_duration(y=y, sr=sr)
        intervals = librosa.effects.split(y, top_db=30)
        voiced = sum((end - start) / sr for start, end in intervals)
        silence_ratio = 1.0 - (voiced / total) if total > 0 else 0.0
        pause_count = max(0, len(intervals) - 1)

        f0 = librosa.yin(y, fmin=50, fmax=500)
        f0v = f0[f0 > 0]
        trajectory = "flat/steady"
        if len(f0v) > 20:
            k = max(1, int(len(f0v) * 0.3))
            drop = float(np.mean(f0v[:k]) - np.mean(f0v[-k:]))
            if drop > 15:
                trajectory = "falling (vocal fry, defeat, or fatigue)"
            elif drop < -15:
                trajectory = "rising (seeking validation, questioning, or escalating panic)"
        return {"silence_ratio": round(silence_ratio, 3), "pause_count": pause_count,
                "pitch_trajectory": trajectory}
    except Exception as e:  # noqa: BLE001
        log.warning("paralinguistics.temporal_failed", err=str(e))
        return {"silence_ratio": 0.0, "pause_count": 0, "pitch_trajectory": "unknown"}


_LEXICAL_SYSTEM_PROMPT = (
    "You are an expert psychological profiler and paralinguistics engine. You will evaluate "
    "a transcript alongside deep acoustic biometrics.\n"
    "- Jitter > 0.05 indicates vocal cord tension (stress/suppression).\n"
    "- High Silence Ratio (> 0.20) & frequent pauses indicate grief, heavy cognitive load, or "
    "hesitation.\n"
    "- Pitch Trajectory (Falling) indicates defeat, giving up, or apathy (vocal fry).\n"
    "Detect contradictions between the text and how they sound, and output the TRUE underlying "
    "affective state.\n\n"
    "Output strictly as JSON:\n"
    "{\n"
    '  "lexical_sentiment": "positive|negative|neutral",\n'
    '  "final_affective_state": "e.g., psychopathic_threat, cynical_or_masking_grief, '
    'tense_suppressed, genuine_joy",\n'
    '  "reasoning": "1 sentence explaining why"\n'
    "}"
)
