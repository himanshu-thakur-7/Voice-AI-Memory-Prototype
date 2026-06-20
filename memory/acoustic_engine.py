"""Post-call acoustic engine.

Two paths, one public surface (``analyze_audio``):

* **Librosa fallback** — always available. Computes ``pitch_variance`` (variance of the
  YIN f0 contour) and ``rms_energy`` from the WAV, then maps to
  ``acoustic_affect ∈ {agitated, subdued, neutral}``. This is the path the brief
  specifies; it runs without GPUs and is what the sarcasm filter keys off in
  :py:func:`memory.graph_engine.apply_facts`.

* **Rich engine** — :class:`ParalinguisticExtractor` ported from the user's
  ``AffectiveMemoryExtractor`` (SenseVoice + openSMILE + librosa + LLM contradiction
  pass). When ``funasr``/``opensmile``/``torch`` are installed, it runs INSTEAD of the
  fallback (settings.affect_extractor controls), yielding the additional transcript,
  audio events, eGeMAPS biometrics, and the LLM-reconciled ``final_affective_state``.

Both paths return the same :class:`AcousticResult`; downstream callers don't care which
one ran. Both are **sync** — heavy work; the caller wraps in ``asyncio.to_thread``.
"""

from __future__ import annotations

import importlib.util
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import structlog

from config import Settings
from memory.schemas import Emotion

log = structlog.get_logger(__name__)

# Heavy deps that gate the rich path. The librosa fallback only needs librosa+numpy.
_RICH_DEPS = ("funasr", "opensmile", "torch")


def is_rich_available() -> bool:
    """True if every rich-engine dependency is importable (without importing them)."""
    return all(importlib.util.find_spec(m) is not None for m in _RICH_DEPS)


# ── public result type ──────────────────────────────────────────────────────


@dataclass
class AcousticResult:
    """The full read-out of one audio file. The fields the brief mandates are at the top;
    the rest is what the rich engine adds when its deps are installed."""

    # Required by the brief ───────────────────────────────────────────────────
    pitch_variance: float = 0.0          # variance of f0 (semitone-ish); 0 = silent/flat
    rms_energy: float = 0.0              # mean RMS over the file, in [0, 1]
    acoustic_affect: str = "neutral"     # {"agitated", "subdued", "neutral"}

    # Rich (defaulted; populated when the SenseVoice path runs) ───────────────
    transcript: str = ""                 # SenseVoice's offline transcription, if available
    base_acoustic_emotion: str = "neutral"  # SenseVoice tag → {happy/sad/angry/neutral}
    final_affective_state: str = ""      # LLM-reconciled label, e.g. "cynical_or_masking_grief"
    lexical_sentiment: str = "neutral"   # what the WORDS said: positive|negative|neutral
    llm_reasoning: str = ""
    detected_events: list[str] = field(default_factory=list)  # laughter/cry/...
    acoustic_biometrics: dict[str, Any] = field(default_factory=dict)
    engine: str = "librosa"              # "librosa" or "rich"

    def to_paralinguistics_dict(self) -> dict[str, Any]:
        """Shape we persist on the User node — mirrors the user's original payload."""
        return {
            "base_acoustic_emotion": self.base_acoustic_emotion,
            "final_affective_state": self.final_affective_state,
            "lexical_sentiment": self.lexical_sentiment,
            "llm_reasoning": self.llm_reasoning,
            "detected_events": self.detected_events,
            "acoustic_biometrics": {
                **self.acoustic_biometrics,
                "pitch_variance": self.pitch_variance,
                "rms_energy": self.rms_energy,
            },
            "acoustic_affect": self.acoustic_affect,
            "engine": self.engine,
        }


# ── public entrypoint ───────────────────────────────────────────────────────


def analyze_audio(audio_path: str, settings: Settings) -> AcousticResult:
    """Run the acoustic analysis. Routes by ``settings.affect_extractor``:

    * ``"librosa"`` — always the lightweight fallback;
    * ``"rich"``    — force the rich engine; if its deps are missing, log + fallback;
    * ``"auto"``    — rich if installed, else librosa.

    Any failure (corrupted WAV, missing model, etc.) downgrades to neutral defaults rather
    than raising; the post-call worker should never crash a call's bookkeeping.
    """
    if not audio_path:
        return AcousticResult()

    want_rich = settings.affect_extractor in ("rich", "auto") and is_rich_available()
    if settings.affect_extractor == "rich" and not is_rich_available():
        log.warning("acoustic.rich_unavailable",
                    reason="install requirements-rich.txt — falling back to librosa")

    if want_rich:
        try:
            return _get_rich_extractor(settings).analyze(audio_path)
        except Exception as e:  # noqa: BLE001
            log.warning("acoustic.rich_failed", err=str(e))
            # fall through to librosa

    return _librosa_only(audio_path)


# ── librosa fallback (Task B from the brief, kept dependency-light) ─────────


def _librosa_only(audio_path: str) -> AcousticResult:
    """Heuristic pitch_variance + RMS → acoustic_affect ∈ {agitated, subdued, neutral}.

    Thresholds tuned against typical telephone-bandwidth conversational speech:
      - **agitated** : RMS is loud AND pitch swings widely  (anger, panic, sarcasm).
      - **subdued**  : RMS is quiet AND pitch is flat        (grief, defeat, fatigue).
      - **neutral**  : anything in between.
    """
    try:
        import librosa  # lazy
        import numpy as np  # lazy

        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        if y.size == 0:
            return AcousticResult()

        rms = float(np.mean(librosa.feature.rms(y=y))) if y.size else 0.0
        # YIN-based f0; pitch_variance = variance over voiced frames in semitones.
        f0 = librosa.yin(y, fmin=50, fmax=500)
        voiced = f0[f0 > 0]
        if voiced.size > 20:
            semitones = 12.0 * np.log2(voiced / 27.5)  # match SenseVoice's reference
            pitch_var = float(np.var(semitones))
        else:
            pitch_var = 0.0

        affect = _classify_acoustic_affect(pitch_var, rms)
        log.info("acoustic.librosa", path=audio_path,
                 pitch_variance=round(pitch_var, 3), rms=round(rms, 4), affect=affect)
        return AcousticResult(
            pitch_variance=round(pitch_var, 3),
            rms_energy=round(rms, 4),
            acoustic_affect=affect,
            engine="librosa",
        )
    except Exception as e:  # noqa: BLE001 — corrupted WAV, sox issues, etc.
        log.warning("acoustic.librosa_failed", err=str(e), path=audio_path)
        return AcousticResult()


def _classify_acoustic_affect(pitch_variance: float, rms_energy: float) -> str:
    """Map (pitch_variance, rms_energy) → {agitated, subdued, neutral}.

    The thresholds are conservative on purpose: an over-eager 'agitated' classification
    would trip the sarcasm floor too often, eroding trust in correct positive facts.
    """
    if pitch_variance >= 25.0 and rms_energy >= 0.04:
        return "agitated"
    if pitch_variance <= 8.0 and rms_energy <= 0.015:
        return "subdued"
    return "neutral"


# ── rich engine (port of the user's AffectiveMemoryExtractor) ───────────────


_RICH: ParalinguisticExtractor | None = None


def _get_rich_extractor(settings: Settings) -> ParalinguisticExtractor:
    """Singleton — model loads are expensive; load once per process."""
    global _RICH
    if _RICH is None:
        _RICH = ParalinguisticExtractor(settings)
    return _RICH


class ParalinguisticExtractor:
    """Port of the user's :class:`AffectiveMemoryExtractor`, adapted to:
      * lazy-import all heavy deps so the worker boots without them,
      * use our :class:`Settings` (no ``os.environ`` reads),
      * return :class:`AcousticResult` for uniform downstream handling.

    LLM prompts are kept VERBATIM from the user's prototype.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._sense_voice: Any = None
        self._smile: Any = None
        self._openai: Any = None
        self._loaded = False

    def _load_models(self) -> None:
        if self._loaded:
            return
        import opensmile  # lazy
        import torch  # lazy
        from funasr import AutoModel  # lazy

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        log.info("acoustic.rich.loading", device=device)
        self._sense_voice = AutoModel(
            model="iic/SenseVoiceSmall",
            trust_remote_code=True,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=device,
            disable_update=True,
        )
        self._smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        if self._settings.openai_api_key:
            from openai import OpenAI

            self._openai = OpenAI(api_key=self._settings.openai_api_key)
        self._loaded = True
        log.info("acoustic.rich.loaded")

    def analyze(self, audio_path: str) -> AcousticResult:
        self._load_models()

        # 1) SenseVoice — transcript + tagged emotion + audio events.
        sv = self._sense_voice.generate(input=audio_path, language="auto",
                                        use_itn=True, batch_size=64)
        raw_text = " ".join(r.get("text", "") for r in sv)
        transcript, base_emotion, events = _parse_sensevoice(raw_text)

        # 2) openSMILE — eGeMAPS biometrics (jitter/shimmer/loudness/pitch_mean).
        biometrics = self._opensmile(audio_path)

        # 3) librosa — temporal/pitch dynamics + the brief's required metrics.
        temporal = _temporal_and_pitch(audio_path)
        biometrics.update(temporal)

        # 4) LLM contradiction pass — words vs. voice → final_affective_state.
        llm = self._contradiction_pass(transcript, base_emotion, biometrics)

        # 5) Derive the brief's acoustic_affect from the rich data we now have. If the LLM
        #    declared sarcasm/masking, force "agitated" so the sarcasm filter fires.
        affect = _affect_from_rich(
            final_state=llm.get("final_affective_state", ""),
            base_emotion=base_emotion,
            biometrics=biometrics,
        )

        return AcousticResult(
            pitch_variance=float(biometrics.get("pitch_variance", 0.0)),
            rms_energy=float(biometrics.get("rms_energy", 0.0)),
            acoustic_affect=affect,
            transcript=transcript,
            base_acoustic_emotion=base_emotion,
            final_affective_state=llm.get("final_affective_state", "") or base_emotion,
            lexical_sentiment=llm.get("lexical_sentiment", "neutral"),
            llm_reasoning=llm.get("reasoning", ""),
            detected_events=sorted(set(events)),
            acoustic_biometrics=biometrics,
            engine="rich",
        )

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
            log.warning("acoustic.opensmile_failed", err=str(e))
            return {}

    def _contradiction_pass(
        self, transcript: str, acoustic_emotion: str, acoustics: dict[str, Any]
    ) -> dict[str, Any]:
        """LLM reconciliation — prompt ported verbatim from the user's prototype."""
        if not transcript.strip() or self._openai is None:
            return {"lexical_sentiment": "neutral",
                    "final_affective_state": acoustic_emotion,
                    "reasoning": "LLM offline."}

        system_prompt = (
            "You are an expert psychological profiler and paralinguistics engine. "
            "You will evaluate a transcript alongside deep acoustic biometrics.\n"
            "- Jitter > 0.05 indicates vocal cord tension (stress/suppression).\n"
            "- High Silence Ratio (> 0.20) & frequent pauses indicate grief, heavy "
            "cognitive load, or hesitation.\n"
            "- Pitch Trajectory (Falling) indicates defeat, giving up, or apathy "
            "(vocal fry).\n"
            "Detect contradictions between the text and how they sound, and output the "
            "TRUE underlying affective state.\n\n"
            'Output strictly as JSON: {"lexical_sentiment": "positive|negative|neutral", '
            '"final_affective_state": "e.g., psychopathic_threat, cynical_or_masking_grief, '
            'tense_suppressed, genuine_joy", "reasoning": "1 sentence explaining why"}'
        )
        user_prompt = (
            f"Transcript: '{transcript}'\n"
            f"Acoustic Emotion: {acoustic_emotion}\n"
            f"Jitter (Tension): {acoustics.get('jitter_local', 0)}\n"
            f"Silence Ratio: {acoustics.get('silence_ratio', 0)}\n"
            f"Pause Count: {acoustics.get('pause_count', 0)}\n"
            f"Pitch Trajectory: {acoustics.get('pitch_trajectory', 'unknown')}"
        )
        try:
            resp = self._openai.chat.completions.create(
                model=self._settings.lexical_model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            return json.loads(resp.choices[0].message.content or "{}")
        except Exception as e:  # noqa: BLE001
            log.error("acoustic.llm_failed", err=str(e))
            return {"lexical_sentiment": "neutral",
                    "final_affective_state": acoustic_emotion,
                    "reasoning": "Error"}


# ── pure helpers (testable without any heavy deps) ──────────────────────────


def _parse_sensevoice(raw_text: str) -> tuple[str, str, list[str]]:
    """Strip SenseVoice's rich tags. E.g. ``<|zh|><|HAPPY|><|Laughter|> hi``."""
    emotions = re.findall(r"<\|(HAPPY|SAD|ANGRY|NEUTRAL)\|>", raw_text)
    dominant = (Counter(e.lower() for e in emotions).most_common(1)[0][0]
                if emotions else "neutral")
    events = [e.lower() for e in re.findall(
        r"<\|(Laughter|Applause|Cough|Sneeze|Cry|Music)\|>", raw_text)]
    clean = re.sub(r"<\|.*?\|>", "", raw_text).strip()
    return clean, dominant, events


def _temporal_and_pitch(audio_path: str) -> dict[str, Any]:
    """librosa-only temporal features: silence ratio, pauses, pitch trajectory, plus the
    brief's required ``pitch_variance`` and ``rms_energy`` so the rich engine can populate
    those fields too (avoids running librosa twice)."""
    try:
        import librosa  # lazy
        import numpy as np  # lazy

        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        if y.size == 0:
            return {"pitch_variance": 0.0, "rms_energy": 0.0, "silence_ratio": 0.0,
                    "pause_count": 0, "pitch_trajectory": "unknown"}
        total = float(librosa.get_duration(y=y, sr=sr))
        intervals = librosa.effects.split(y, top_db=30)
        voiced_sec = float(sum((end - start) / sr for start, end in intervals))
        silence_ratio = (1.0 - voiced_sec / total) if total > 0 else 0.0
        pause_count = max(0, len(intervals) - 1)

        rms = float(np.mean(librosa.feature.rms(y=y)))
        f0 = librosa.yin(y, fmin=50, fmax=500)
        voiced = f0[f0 > 0]
        if voiced.size > 20:
            semitones = 12.0 * np.log2(voiced / 27.5)
            pitch_var = float(np.var(semitones))
            k = max(1, int(len(voiced) * 0.3))
            drop = float(np.mean(voiced[:k]) - np.mean(voiced[-k:]))
            if drop > 15:
                trajectory = "falling (vocal fry, defeat, or fatigue)"
            elif drop < -15:
                trajectory = "rising (questioning or escalating panic)"
            else:
                trajectory = "flat/steady"
        else:
            pitch_var = 0.0
            trajectory = "unknown"
        return {
            "pitch_variance": round(pitch_var, 3),
            "rms_energy": round(rms, 4),
            "silence_ratio": round(silence_ratio, 3),
            "pause_count": pause_count,
            "pitch_trajectory": trajectory,
        }
    except Exception as e:  # noqa: BLE001
        log.warning("acoustic.temporal_failed", err=str(e))
        return {"pitch_variance": 0.0, "rms_energy": 0.0, "silence_ratio": 0.0,
                "pause_count": 0, "pitch_trajectory": "unknown"}


def _affect_from_rich(final_state: str, base_emotion: str, biometrics: dict[str, Any]) -> str:
    """When the rich engine ran, prefer its semantic signal (final_affective_state) to
    set acoustic_affect — falling back to the librosa heuristic on the metrics it computed."""
    needle = (final_state or "").lower()
    # Sarcasm / masking signals from the LLM → force agitated so the sarcasm floor fires.
    if any(k in needle for k in ("sarcas", "mask", "psychopath", "threat", "cynical")):
        return "agitated"
    if any(k in needle for k in ("grief", "defeat", "apath", "subdued", "fatigue")):
        return "subdued"
    if base_emotion in ("angry",) or any(k in needle for k in ("rage", "hostil", "panic")):
        return "agitated"
    if base_emotion == "sad":
        return "subdued"
    return _classify_acoustic_affect(
        float(biometrics.get("pitch_variance", 0.0)),
        float(biometrics.get("rms_energy", 0.0)),
    )


# ── coarse-emotion mapping for AffectiveState ────────────────────────────────


def map_emotion(base_acoustic_emotion: str, final_affective_state: str) -> Emotion:
    """Collapse the rich engine's two emotion outputs onto our 6-value enum.

    The nuanced ``final_affective_state`` (e.g. 'cynical_or_masking_grief',
    'tense_suppressed', 'psychopathic_threat') wins when present; otherwise we fall back
    to the SenseVoice acoustic label. Buckets match the wording the LLM prompt above
    declares it can emit.
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


# ── coarse valence/arousal estimate (only used when the rich engine ran) ────


_VA_BASE = {
    Emotion.NEUTRAL: (0.0, 0.3), Emotion.FRUSTRATED: (-0.6, 0.7),
    Emotion.ANGRY: (-0.8, 0.85), Emotion.ANXIOUS: (-0.3, 0.6),
    Emotion.HAPPY: (0.7, 0.6), Emotion.SAD: (-0.5, 0.2),
}


def valence_arousal(emotion: Emotion, biometrics: dict[str, Any]) -> tuple[float, float]:
    """Coarse (valence, arousal) from emotion bucket + jitter (vocal tension)."""
    valence, arousal = _VA_BASE[emotion]
    jitter = float(biometrics.get("jitter_local", 0) or 0)
    arousal = max(0.0, min(1.0, arousal + min(0.15, jitter * 2)))
    return round(valence, 3), round(arousal, 3)
