"""Whisper transcription via pywhispercpp.

Runs whisper.cpp under the hood — pure C++, no PyTorch, ARM-friendly.
Models cached in /data/models so they persist across addon restarts.

Transcription is sync (whisper.cpp blocks); we run it in an executor so the
asyncio capture loop isn't blocked while we transcribe ~30 sec of audio.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

MODELS_DIR = Path("/data/models")


class Transcriber:
    def __init__(self, model_name: str = "base.en") -> None:
        self.model_name = model_name
        self._model = None  # lazy-loaded on first use

    def _ensure_model(self):
        if self._model is not None:
            return
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        # pywhispercpp.Model handles model download from HuggingFace if not cached
        from pywhispercpp.model import Model

        log.info("loading whisper model %s (downloads ~150 MB on first use)…", self.model_name)
        self._model = Model(
            self.model_name,
            models_dir=str(MODELS_DIR),
            n_threads=4,        # use all 4 CM4 cores
            print_progress=False,
            print_realtime=False,
        )
        log.info("whisper model %s loaded", self.model_name)

    async def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe a numpy audio array to text."""
        if len(audio) == 0:
            return ""

        # whisper expects float32 normalized [-1, 1] at 16 kHz
        if audio.dtype == np.int16:
            x = audio.astype(np.float32) / 32768.0
        else:
            x = audio.astype(np.float32)

        if sample_rate != 16000:
            log.warning(
                "sample_rate=%d not 16000 — whisper expects 16k, results may degrade",
                sample_rate,
            )

        loop = asyncio.get_event_loop()
        try:
            self._ensure_model()
            segments = await loop.run_in_executor(None, self._model.transcribe, x)
        except Exception as e:
            log.error("transcription failed: %s", e)
            return ""

        text = " ".join(s.text.strip() for s in segments).strip()
        log.info("transcribed %d sec → %d chars", len(audio) // sample_rate, len(text))
        return text
