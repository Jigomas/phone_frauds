"""
FraudDetector: main class that combines ASR + keyword + semantic + prosodic signals.

Usage:
    detector = FraudDetector()
    result = detector.predict("path/to/audio.wav")
    # result = {"label": 0, "fraud_score": 0.82, "transcript": "...", ...}
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

# Default ensemble weights (can be overridden after tune_threshold.py)
DEFAULT_WEIGHTS = {
    "keyword": 0.45,
    "semantic": 0.40,
    "prosodic": 0.15,
}
DEFAULT_THRESHOLD = 0.50  # fraud_score > threshold → label=0 (fraud)

# Whisper model to use. Override via env WHISPER_MODEL.
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")


@dataclass
class DetectionResult:
    filename: str
    label: int            # 0 = fraud, 1 = legit
    fraud_score: float    # [0, 1]
    keyword_score: float
    semantic_score: float
    prosodic_score: float
    transcript: str
    error: str = ""


class FraudDetector:
    def __init__(
        self,
        weights: dict[str, float] | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        device: str = "auto",
        whisper_model: str = WHISPER_MODEL,
        asr: str = "auto",  # "whisper" | "vosk" | "auto" (vosk on CPU, whisper on GPU)
    ):
        self.weights = weights or DEFAULT_WEIGHTS
        self.threshold = threshold
        self.whisper_model = whisper_model
        self._asr_model = None

        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        if asr == "auto":
            self.asr = "vosk" if self.device == "cpu" else "whisper"
        else:
            self.asr = asr

        compute_type = "float16" if self.device == "cuda" else "int8"
        self._compute_type = compute_type

    def _load_whisper(self):
        if self._asr_model is None:
            from faster_whisper import WhisperModel
            print(f"Loading Whisper model '{self.whisper_model}' on {self.device}...", flush=True)
            self._asr_model = WhisperModel(
                self.whisper_model,
                device=self.device,
                compute_type=self._compute_type,
            )
            print("Whisper model loaded.", flush=True)
        return self._asr_model

    def transcribe(self, audio_path: str) -> tuple[str, list[dict]]:
        """Returns (full_text, words_with_timestamps)."""
        if self.asr == "vosk":
            from .asr_vosk import transcribe_vosk
            return transcribe_vosk(audio_path)

        model = self._load_whisper()
        segments, _ = model.transcribe(
            audio_path,
            language="ru",
            word_timestamps=True,
            beam_size=5,
            vad_filter=True,
        )
        words: list[dict] = []
        parts: list[str] = []
        for seg in segments:
            parts.append(seg.text.strip())
            if seg.words:
                for w in seg.words:
                    words.append({"word": w.word, "start": w.start, "end": w.end})
        return " ".join(parts), words

    def predict(self, audio_path: str) -> DetectionResult:
        filename = os.path.basename(audio_path)
        print(f"  -> transcribing {filename}...", flush=True)
        try:
            return self._predict(audio_path, filename)
        except Exception as exc:
            return DetectionResult(
                filename=filename,
                label=1,
                fraud_score=0.0,
                keyword_score=0.0,
                semantic_score=0.0,
                prosodic_score=0.0,
                transcript="",
                error=str(exc),
            )

    def _predict(self, audio_path: str, filename: str) -> DetectionResult:
        from .keywords import keyword_score
        from .semantic import semantic_score
        from .prosodic import prosodic_score

        transcript, words = self.transcribe(audio_path)

        kw = keyword_score(transcript)
        sem = semantic_score(transcript)
        pros = prosodic_score(audio_path, words if words else None)

        w = self.weights
        score = (
            w["keyword"] * kw
            + w["semantic"] * sem
            + w["prosodic"] * pros
        )

        label = 0 if score > self.threshold else 1

        return DetectionResult(
            filename=filename,
            label=label,
            fraud_score=round(score, 4),
            keyword_score=round(kw, 4),
            semantic_score=round(sem, 4),
            prosodic_score=round(pros, 4),
            transcript=transcript,
        )
