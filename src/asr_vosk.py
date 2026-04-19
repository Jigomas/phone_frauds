"""
Vosk-based ASR — fast CPU transcription for Russian.
Model is downloaded once to ~/.cache/vosk/
"""

from __future__ import annotations

import json
import os
import wave

_model = None
VOSK_MODEL_NAME = os.environ.get("VOSK_MODEL", "vosk-model-small-ru-0.22")
VOSK_MODEL_URL  = f"https://alphacephei.com/vosk/models/{VOSK_MODEL_NAME}.zip"
VOSK_CACHE_DIR  = os.path.join(os.path.expanduser("~"), ".cache", "vosk")


def _get_model():
    global _model
    if _model is not None:
        return _model

    from vosk import Model, SetLogLevel
    SetLogLevel(-1)

    model_path = os.path.join(VOSK_CACHE_DIR, VOSK_MODEL_NAME)
    if not os.path.exists(model_path):
        print(f"Downloading Vosk Russian model to {model_path}...", flush=True)
        import urllib.request, zipfile
        os.makedirs(VOSK_CACHE_DIR, exist_ok=True)
        zip_path = model_path + ".zip"
        urllib.request.urlretrieve(VOSK_MODEL_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(VOSK_CACHE_DIR)
        os.remove(zip_path)
        print("Vosk model ready.", flush=True)

    _model = Model(model_path)
    return _model


def transcribe_vosk(audio_path: str) -> tuple[str, list[dict]]:
    """Returns (full_text, words_with_timestamps)."""
    import subprocess, tempfile

    # Convert to 16kHz mono WAV if needed
    tmp = None
    try:
        with wave.open(audio_path, "rb") as wf:
            sr, ch = wf.getframerate(), wf.getnchannels()
    except Exception:
        sr, ch = 0, 0

    if sr != 16000 or ch != 1:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", tmp.name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
        audio_path = tmp.name

    from vosk import KaldiRecognizer
    model = _get_model()

    words: list[dict] = []
    parts: list[str] = []

    with wave.open(audio_path, "rb") as wf:
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)
        while True:
            data = wf.readframes(4000)
            if not data:
                break
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                if res.get("text"):
                    parts.append(res["text"])
                for w in res.get("result", []):
                    words.append({"word": w["word"], "start": w["start"], "end": w["end"]})
        final = json.loads(rec.FinalResult())
        if final.get("text"):
            parts.append(final["text"])
        for w in final.get("result", []):
            words.append({"word": w["word"], "start": w["start"], "end": w["end"]})

    if tmp:
        os.unlink(tmp.name)

    return " ".join(parts), words
