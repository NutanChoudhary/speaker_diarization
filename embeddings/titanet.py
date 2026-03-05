"""
embeddings/titanet.py
─────────────────────
Speaker embedding extraction via a local TitaNet ONNX model.

Parallelism strategy
────────────────────
ONNX Runtime sessions are NOT thread-safe for concurrent `.run()` calls on
the same session object.  To achieve true parallelism we therefore create one
independent InferenceSession per worker thread (the model is read-only, so
this is safe and cheap — ONNX weights are memory-mapped).

The public API is unchanged:

    embedder = TitaNetEmbedder(model_path, num_workers=4)
    valid_segs, embeddings = embedder.extract_batch(segments, audio_np, sr=16000)
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional, Tuple

import librosa
import numpy as np
import onnxruntime as ort


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_SR     = 16_000
_N_MELS = 80
_N_FFT  = 512
_HOP    = 160
_WIN    = 400


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def compute_log_mel(
    audio     : np.ndarray,
    sr        : int = _SR,
    n_mels    : int = _N_MELS,
    n_fft     : int = _N_FFT,
    hop_length: int = _HOP,
    win_length: int = _WIN,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute log-mel spectrogram.

    Returns
    -------
    log_mel : np.ndarray  shape (1, n_mels, T)  float32
    length  : np.ndarray  shape (1,)             int64
    """
    mel = librosa.feature.melspectrogram(
        y          = audio,
        sr         = sr,
        n_fft      = n_fft,
        hop_length = hop_length,
        win_length = win_length,
        n_mels     = n_mels,
        window     = "hann",
        center     = False,
        power      = 2.0,
    )
    log_mel = np.log(mel + 1e-9, out=mel).astype(np.float32)
    log_mel = log_mel[np.newaxis]                        # (1, n_mels, T)
    length  = np.array([log_mel.shape[-1]], dtype=np.int64)
    return log_mel, length


# ─────────────────────────────────────────────────────────────────────────────
# Per-thread session pool
# ─────────────────────────────────────────────────────────────────────────────

class _SessionPool:
    """
    Maintains one ORT InferenceSession per worker thread via a threading.local
    store.  Sessions are created lazily on first use in each thread, so we
    never share a session across threads.
    """

    def __init__(self, model_path: str, intra_threads: int) -> None:
        self._model_path    = model_path
        self._intra_threads = intra_threads
        self._local         = threading.local()

    def _make_session(self) -> ort.InferenceSession:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.intra_op_num_threads     = self._intra_threads
        opts.inter_op_num_threads     = 1
        opts.enable_mem_pattern       = True
        opts.enable_cpu_mem_arena     = True
        return ort.InferenceSession(
            self._model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

    @property
    def session(self) -> ort.InferenceSession:
        """Return (or lazily create) the session for the calling thread."""
        if not hasattr(self._local, "session"):
            self._local.session = self._make_session()
        return self._local.session

    def get_io_names(self) -> Tuple[str, str, str]:
        """Return (input0_name, input1_name, embedding_output_name)."""
        sess = self.session          # uses the calling-thread session
        inp  = sess.get_inputs()
        out  = sess.get_outputs()
        return inp[0].name, inp[1].name, out[1].name


# ─────────────────────────────────────────────────────────────────────────────
# Embedder
# ─────────────────────────────────────────────────────────────────────────────

class TitaNetEmbedder:
    """
    TitaNet speaker embedder backed by ONNX Runtime.

    Parameters
    ----------
    model_path     : Path to ``titanet_large.onnx``.
    num_workers    : Number of parallel extraction threads.
                     1  → sequential (original behaviour, zero overhead).
                     >1 → one independent ORT session per thread.
    intra_threads  : ORT intra-op thread count per session (0 = auto-detect).
                     When num_workers > 1, keep this low (1–2) to avoid
                     over-subscription; when num_workers == 1 you can raise it.
    """

    def __init__(
        self,
        model_path   : str,
        num_workers  : int = 1,
        intra_threads: int = 0,
    ) -> None:
        self._num_workers = max(1, num_workers)

        # Auto-tune intra_threads to avoid CPU over-subscription
        if intra_threads == 0 and self._num_workers > 1:
            import os
            cpu_count    = os.cpu_count() or 4
            intra_threads = max(1, cpu_count // self._num_workers)

        self._pool = _SessionPool(model_path, intra_threads)

        # Resolve I/O names once (from the main thread's session)
        self._in0, self._in1, self._out1 = self._pool.get_io_names()

    # ── single segment ────────────────────────────────────────────────────

    def extract_one(self, audio_segment: np.ndarray, sr: int = _SR) -> np.ndarray:
        """Extract an L2-normalised embedding for one audio segment."""
        log_mel, length = compute_log_mel(audio_segment, sr=sr)
        feed = {self._in0: log_mel, self._in1: length}
        emb  = self._pool.session.run([self._out1], feed)[0].squeeze().astype(np.float32)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    # ── public batch entry point ──────────────────────────────────────────

    def extract_batch(
        self,
        segments      : List[dict],
        audio_np      : np.ndarray,
        sr            : int   = _SR,
        min_duration_s: float = 0.5,
        progress_cb   : Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[List[dict], List[np.ndarray]]:
        """
        Extract embeddings for all segments, optionally in parallel.

        Parameters
        ----------
        segments       : list of dicts with ``'start'`` / ``'end'`` keys (seconds).
        audio_np       : full recording as float32 numpy array.
        sr             : sample rate.
        min_duration_s : segments shorter than this are silently skipped.
        progress_cb    : optional callable(completed, total).

        Returns
        -------
        valid_segments : list[dict]  — in the original segment order
        embeddings     : list[np.ndarray]  — one per valid segment
        """
        valid = [
            seg for seg in segments
            if (seg["end"] - seg["start"]) >= min_duration_s
        ]
        if not valid:
            return [], []

        if self._num_workers == 1:
            return self._extract_sequential(valid, audio_np, sr, progress_cb)
        else:
            return self._extract_parallel(valid, audio_np, sr, progress_cb)

    # ── sequential (num_workers == 1) ─────────────────────────────────────

    def _extract_sequential(
        self,
        valid      : List[dict],
        audio_np   : np.ndarray,
        sr         : int,
        progress_cb: Optional[Callable[[int, int], None]],
    ) -> Tuple[List[dict], List[np.ndarray]]:
        embeddings = []
        n = len(valid)
        for i, seg in enumerate(valid):
            s   = int(seg["start"] * sr)
            e   = int(seg["end"]   * sr)
            emb = self.extract_one(audio_np[s:e], sr=sr)
            embeddings.append(emb)
            if progress_cb:
                progress_cb(i + 1, n)
        return valid, embeddings

    # ── parallel (num_workers > 1) ────────────────────────────────────────

    def _extract_parallel(
        self,
        valid      : List[dict],
        audio_np   : np.ndarray,
        sr         : int,
        progress_cb: Optional[Callable[[int, int], None]],
    ) -> Tuple[List[dict], List[np.ndarray]]:
        """
        Submit each segment as an independent future.

        Key design points
        ─────────────────
        • Each thread gets its own ORT session via _SessionPool (thread-local).
        • Results are collected by original index so order is preserved.
        • progress_cb is called under a lock so the counter is exact.
        • audio_np is read-only inside workers — no copy needed.
        """
        n            = len(valid)
        embeddings   = [None] * n          # pre-allocate to preserve order
        lock         = threading.Lock()
        completed    = [0]

        def _worker(idx: int, seg: dict) -> Tuple[int, np.ndarray]:
            s   = int(seg["start"] * sr)
            e   = int(seg["end"]   * sr)
            log_mel, length = compute_log_mel(audio_np[s:e], sr=sr)
            feed = {self._in0: log_mel, self._in1: length}
            # Each thread uses its own lazily-created session
            emb = (self._pool.session
                       .run([self._out1], feed)[0]
                       .squeeze()
                       .astype(np.float32))
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb /= norm
            return idx, emb

        with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            futures = {
                executor.submit(_worker, idx, seg): idx
                for idx, seg in enumerate(valid)
            }
            for future in as_completed(futures):
                idx, emb = future.result()   # re-raises any worker exception
                embeddings[idx] = emb
                with lock:
                    completed[0] += 1
                    if progress_cb:
                        progress_cb(completed[0], n)

        return valid, embeddings
