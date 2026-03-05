"""
vad/silero_vad_offline.py
─────────────────────────
Fully offline, self-contained Silero VAD wrapper.
No `silero_vad` pip package required at runtime.

Usage
-----
    from vad.silero_vad_offline import OnnxVAD

    vad = OnnxVAD("/path/to/silero_vad.onnx")
    timestamps = vad.get_speech_timestamps(audio_tensor, return_seconds=True)
"""

from __future__ import annotations

import warnings
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Low-level ONNX session wrapper  (mirrors silero_vad's OnnxWrapper exactly)
# ─────────────────────────────────────────────────────────────────────────────

class _OnnxSession:
    """Thin stateful wrapper around an ONNX Runtime session for Silero VAD."""

    SUPPORTED_SR = {8000, 16000}

    def __init__(self, model_path: str, force_cpu: bool = False) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        providers = (
            ["CPUExecutionProvider"]
            if force_cpu and "CPUExecutionProvider" in ort.get_available_providers()
            else None
        )
        kwargs = dict(sess_options=opts)
        if providers:
            kwargs["providers"] = providers

        self._session = ort.InferenceSession(model_path, **kwargs)
        self._reset_states()

    # ── state management ──────────────────────────────────────────────────

    def _reset_states(self, batch_size: int = 1) -> None:
        self._state = torch.zeros((2, batch_size, 128), dtype=torch.float32)
        self._context: torch.Tensor = torch.zeros(0)
        self._last_sr: int = 0
        self._last_batch: int = 0

    def reset_states(self, batch_size: int = 1) -> None:
        self._reset_states(batch_size)

    # ── input validation ──────────────────────────────────────────────────

    def _validate(self, x: torch.Tensor, sr: int):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() > 2:
            raise ValueError(f"Audio has too many dimensions: {x.dim()}")
        if sr != 16000 and sr % 16000 == 0:
            x = x[:, :: sr // 16000]
            sr = 16000
        if sr not in self.SUPPORTED_SR:
            raise ValueError(f"Unsupported sample rate {sr}. Supported: {self.SUPPORTED_SR}")
        if sr / x.shape[1] > 31.25:
            raise ValueError("Audio chunk is too short.")
        return x, sr

    # ── forward ───────────────────────────────────────────────────────────

    def __call__(self, x: torch.Tensor, sr: int) -> torch.Tensor:
        x, sr = self._validate(x, sr)

        num_samples  = 512 if sr == 16000 else 256
        context_size = 64  if sr == 16000 else 32

        if x.shape[-1] != num_samples:
            raise ValueError(
                f"Expected {num_samples} samples for sr={sr}, got {x.shape[-1]}. "
                "Pass audio in fixed-size chunks."
            )

        batch = x.shape[0]
        if not self._last_batch:
            self._reset_states(batch)
        if self._last_sr and self._last_sr != sr:
            self._reset_states(batch)
        if self._last_batch and self._last_batch != batch:
            self._reset_states(batch)

        if not len(self._context):
            self._context = torch.zeros(batch, context_size)

        x = torch.cat([self._context, x], dim=1)
        ort_in = {
            "input": x.numpy(),
            "state": self._state.numpy(),
            "sr":    np.array(sr, dtype=np.int64),
        }
        out, state = self._session.run(None, ort_in)
        self._state   = torch.from_numpy(state)
        self._context = x[..., -context_size:]
        self._last_sr = sr
        self._last_batch = batch

        return torch.from_numpy(out)


# ─────────────────────────────────────────────────────────────────────────────
# High-level VAD class
# ─────────────────────────────────────────────────────────────────────────────

class OnnxVAD:
    """
    Offline Silero VAD — loads a local .onnx file, no cloud download needed.

    Parameters
    ----------
    model_path : str
        Absolute or relative path to `silero_vad.onnx`.
    force_cpu : bool
        Force CPU inference even if CUDA is available.
    """

    def __init__(self, model_path: str, force_cpu: bool = False) -> None:
        self._model = _OnnxSession(model_path, force_cpu=force_cpu)

    # ── public API ────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_speech_timestamps(
        self,
        audio: torch.Tensor,
        *,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_speech_duration_ms: int = 250,
        max_speech_duration_s: float = float("inf"),
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        return_seconds: bool = False,
        neg_threshold: Optional[float] = None,
        progress_cb: Optional[Callable[[float], None]] = None,
    ) -> List[dict]:
        """
        Split long audio into speech chunks.

        Returns
        -------
        List of dicts with keys ``start`` and ``end`` (samples or seconds).
        """
        audio = self._to_1d(audio, sampling_rate)

        if sampling_rate > 16000 and sampling_rate % 16000 == 0:
            step = sampling_rate // 16000
            audio = audio[::step]
            sampling_rate = 16000
            warnings.warn("Sample rate is a multiple of 16 kHz — downsampled to 16 000 Hz.")
        else:
            step = 1

        if sampling_rate not in {8000, 16000}:
            raise ValueError("Silero VAD supports 8 000 and 16 000 Hz only.")

        win = 512 if sampling_rate == 16000 else 256
        self._model.reset_states()

        if neg_threshold is None:
            neg_threshold = max(threshold - 0.15, 0.01)

        min_speech_s  = sampling_rate * min_speech_duration_ms  / 1000
        pad_s         = sampling_rate * speech_pad_ms           / 1000
        max_speech_s  = sampling_rate * max_speech_duration_s - win - 2 * pad_s
        min_silence_s = sampling_rate * min_silence_duration_ms / 1000
        total         = len(audio)

        # ── probability scan ──────────────────────────────────────────────
        probs = []
        for start in range(0, total, win):
            chunk = audio[start : start + win]
            if len(chunk) < win:
                chunk = F.pad(chunk, (0, win - len(chunk)))
            probs.append(self._model(chunk, sampling_rate).item())
            if progress_cb:
                progress_cb(min(start + win, total) / total * 100)

        # ── state machine ─────────────────────────────────────────────────
        triggered  = False
        speeches   = []
        cur_speech: dict = {}
        temp_end = prev_end = next_start = 0

        for i, prob in enumerate(probs):
            cur = win * i

            if prob >= threshold and temp_end:
                temp_end = 0
                if next_start < prev_end:
                    next_start = cur

            if prob >= threshold and not triggered:
                triggered = True
                cur_speech["start"] = cur
                continue

            if triggered and (cur - cur_speech["start"] > max_speech_s):
                if prev_end:
                    cur_speech["end"] = prev_end
                    speeches.append(cur_speech)
                    cur_speech = {}
                    if next_start < prev_end:
                        triggered = False
                    else:
                        cur_speech["start"] = next_start
                        triggered = True
                else:
                    cur_speech["end"] = cur
                    speeches.append(cur_speech)
                    cur_speech = {}
                    triggered = False
                prev_end = next_start = temp_end = 0
                continue

            if prob < neg_threshold and triggered:
                if not temp_end:
                    temp_end = cur
                if cur - temp_end < min_silence_s:
                    continue
                cur_speech["end"] = temp_end
                if cur_speech["end"] - cur_speech["start"] > min_speech_s:
                    speeches.append(cur_speech)
                cur_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False

        if cur_speech and (total - cur_speech["start"]) > min_speech_s:
            cur_speech["end"] = total
            speeches.append(cur_speech)

        # ── padding pass ──────────────────────────────────────────────────
        for i, sp in enumerate(speeches):
            if i == 0:
                sp["start"] = int(max(0, sp["start"] - pad_s))
            if i < len(speeches) - 1:
                gap = speeches[i + 1]["start"] - sp["end"]
                if gap < 2 * pad_s:
                    sp["end"]              += int(gap // 2)
                    speeches[i + 1]["start"] = int(max(0, speeches[i + 1]["start"] - gap // 2))
                else:
                    sp["end"] = int(min(total, sp["end"] + pad_s))
                    speeches[i + 1]["start"] = int(max(0, speeches[i + 1]["start"] - pad_s))
            else:
                sp["end"] = int(min(total, sp["end"] + pad_s))

        if return_seconds:
            dur = total / sampling_rate
            for sp in speeches:
                sp["start"] = max(round(sp["start"] / sampling_rate, 3), 0)
                sp["end"]   = min(round(sp["end"]   / sampling_rate, 3), dur)
        elif step > 1:
            for sp in speeches:
                sp["start"] *= step
                sp["end"]   *= step

        return speeches

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _to_1d(audio: torch.Tensor, sr: int) -> torch.Tensor:
        if not torch.is_tensor(audio):
            audio = torch.tensor(audio)
        while audio.dim() > 1:
            audio = audio.squeeze(0)
        if audio.dim() > 1:
            raise ValueError("Audio must be 1-D (mono). Got shape: " + str(audio.shape))
        return audio
