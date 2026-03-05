#!/usr/bin/env python3
"""
diarize.py
──────────
End-to-end offline speaker diarization pipeline.

Pipeline
────────
  1. Load audio  (librosa → mono float32 @ 16 kHz)
  2. VAD         (offline Silero ONNX)
  3. Embedding   (TitaNet ONNX, parallel extraction)
  4. Clustering  (agglomerative cosine, average linkage)
  5. Output      (RTTM + CTM)

Usage
─────
  python diarize.py                                        # uses CONFIG block below
  python diarize.py --audio my.wav                         # single file override
  python diarize.py --audio my.wav --workers 8 --threshold 0.70
  python diarize.py --batch-dir /path/to/audio/            # batch mode
  python diarize.py --batch-dir /path/ --batch-workers 4   # batch with fixed workers
  python diarize.py --batch-dir /path/ --dry-run           # profile only, no processing
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import psutil
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(__file__))

from vad.silero_vad_offline   import OnnxVAD
from embeddings.titanet       import TitaNetEmbedder
from clustering.agglomerative import cluster_speakers
from output.reporter          import DiarizationResult, write_rttm, write_ctm, print_report

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger("diarize")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  —  edit these paths before running
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = dict(
    audio_path        = "/mnt/disk5/nutan/output.wav",
    silero_onnx_path  = "/mnt/disk5/nutan/final_mle/silero_vad.onnx",
    titanet_onnx_path = "/mnt/disk5/nutan/nemo/nemo_git/titanet_large.onnx",
    output_dir        = "/mnt/disk5/nutan/diarization_output",

    # VAD
    vad_threshold           = 0.5,
    min_speech_duration_ms  = 250,
    min_silence_duration_ms = 100,
    speech_pad_ms           = 30,

    # Embedding / clustering
    similarity_threshold  = 0.75,
    min_segment_duration  = 0.5,
    num_embedding_workers = 4,

    # Batch
    batch_input_dir       = None,
    batch_extensions      = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"],
    batch_workers         = 4,        # None → auto from resources
    batch_fail_fast       = False,

    # Resource limits
    max_memory_gb         = None,        # None → 80% of available RAM
    max_cpu_percent       = 85.0,
    resource_poll_secs    = 2.0,
    gpu_memory_reserve_gb = 1.0,
)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".aac", ".wma"}


# ══════════════════════════════════════════════════════════════════════════════
#  CONSOLE HELPERS
#  In batch mode (_BATCH_MODE = True) intermediate per-step prints are
#  swallowed; only the final per-file result line and the batch summary print.
# ══════════════════════════════════════════════════════════════════════════════

_BATCH_MODE = False   # flipped to True by run_batch()

def _banner(msg: str) -> None:
    if not _BATCH_MODE:
        print(f"\n\033[1;96m  ▶  {msg}\033[0m")

def _ok(msg: str) -> None:
    if not _BATCH_MODE:
        print(f"     \033[92m✓\033[0m  {msg}")

def _warn(msg: str) -> None:
    print(f"     \033[93m⚠\033[0m  {msg}")

def _err(msg: str) -> None:
    print(f"     \033[91m✗\033[0m  {msg}")

def _progress(pct: float) -> None:
    if not _BATCH_MODE:
        filled = int(pct / 5)
        bar    = "█" * filled + "░" * (20 - filled)
        print(f"\r     [{bar}] {pct:5.1f}%", end="", flush=True)

def _section(title: str, width: int = 70) -> None:
    line = "─" * width
    print(f"\n\033[1;94m{line}\033[0m")
    print(f"\033[1;94m  {title}\033[0m")
    print(f"\033[1;94m{line}\033[0m")

def _inline_print(msg: str) -> None:
    """Only prints in single-file mode."""
    if not _BATCH_MODE:
        print(msg, end="")


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AudioProfile:
    path:             str
    file_size_mb:     float
    duration_s:       float
    sample_rate:      int
    channels:         int
    encoding:         str
    estimated_ram_mb: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class FileResult:
    audio_path:   str
    status:       str          # "success" | "failed" | "skipped"
    error:        Optional[str] = None
    rttm_path:    Optional[str] = None
    ctm_path:     Optional[str] = None
    num_speakers: Optional[int] = None
    num_segments: Optional[int] = None
    duration_s:   Optional[float] = None
    speech_s:     Optional[float] = None
    wall_time_s:  Optional[float] = None
    profile:      Optional[dict]  = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class BatchSummary:
    batch_dir:        str
    output_dir:       str
    started_at:       str
    finished_at:      str
    total_files:      int
    succeeded:        int
    failed:           int
    skipped:          int
    total_audio_s:    float
    total_wall_s:     float
    realtime_factor:  float
    peak_cpu_percent: float
    peak_ram_gb:      float
    workers_used:     int
    files:            List[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
#  RESOURCE MONITOR
# ══════════════════════════════════════════════════════════════════════════════

class ResourceMonitor:
    def __init__(self, max_cpu_pct: float = 85.0,
                 max_ram_gb: Optional[float] = None,
                 gpu_reserve_gb: float = 1.0):
        self.max_cpu_pct    = max_cpu_pct
        self.max_ram_gb     = max_ram_gb or (psutil.virtual_memory().total / 1e9 * 0.80)
        self.gpu_reserve_gb = gpu_reserve_gb
        self._peak_cpu      = 0.0
        self._peak_ram      = 0.0
        self._has_gpu       = torch.cuda.is_available()

    def snapshot(self) -> dict:
        mem  = psutil.virtual_memory()
        cpu  = psutil.cpu_percent(interval=0.2)
        ram  = mem.used / 1e9
        snap = dict(cpu_pct=cpu, ram_used_gb=ram,
                    ram_total_gb=mem.total / 1e9,
                    ram_avail_gb=mem.available / 1e9)
        if self._has_gpu:
            try:
                free_gpu, total_gpu = torch.cuda.mem_get_info(0)
                snap["gpu_used_gb"]  = (total_gpu - free_gpu) / 1e9
                snap["gpu_total_gb"] = total_gpu / 1e9
                snap["gpu_free_gb"]  = free_gpu / 1e9
            except Exception:
                pass
        self._peak_cpu = max(self._peak_cpu, cpu)
        self._peak_ram = max(self._peak_ram, ram)
        return snap

    def can_schedule(self, required_ram_mb: float = 0) -> Tuple[bool, str]:
        snap = self.snapshot()
        if snap["cpu_pct"] > self.max_cpu_pct:
            return False, f"CPU {snap['cpu_pct']:.0f}% > limit {self.max_cpu_pct:.0f}%"
        needed = snap["ram_used_gb"] + required_ram_mb / 1024
        if needed > self.max_ram_gb:
            return False, f"RAM would reach {needed:.1f} GB > limit {self.max_ram_gb:.1f} GB"
        if self._has_gpu:
            free_gb = snap.get("gpu_free_gb", 99)
            if free_gb < self.gpu_reserve_gb:
                return False, f"GPU free {free_gb:.1f} GB < reserve {self.gpu_reserve_gb:.1f} GB"
        return True, "ok"

    @property
    def peak_cpu(self) -> float:
        return self._peak_cpu

    @property
    def peak_ram_gb(self) -> float:
        return self._peak_ram


def _optimal_workers(profiles: List[AudioProfile], cfg: dict,
                     monitor: ResourceMonitor) -> int:
    if cfg.get("batch_workers"):
        return int(cfg["batch_workers"])
    cpu_cores   = psutil.cpu_count(logical=False) or 2
    avail_ram   = psutil.virtual_memory().available / 1e9
    has_gpu     = torch.cuda.is_available()
    avg_ram_mb  = (sum(p.estimated_ram_mb for p in profiles) / len(profiles)
                   if profiles else 1000.0)
    ram_limited = max(1, int((avail_ram * 1024 * 0.80) / avg_ram_mb))
    cpu_limited = max(1, cpu_cores - 1)
    gpu_limited = 2 if has_gpu else cpu_limited
    workers     = min(ram_limited, cpu_limited, gpu_limited, 8)
    log.info(f"Workers → cpu={cpu_cores}, ram={avail_ram:.0f}GB, "
             f"avg_ram/file={avg_ram_mb:.0f}MB, gpu={has_gpu} → {workers}")
    return workers


# ══════════════════════════════════════════════════════════════════════════════
#  BATCH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def discover_audio_files(directory: str, extensions: List[str]) -> List[str]:
    exts = {e.lower() for e in extensions}
    found = []
    for root, _, files in os.walk(directory):
        for fname in sorted(files):
            if Path(fname).suffix.lower() in exts:
                found.append(os.path.join(root, fname))
    return sorted(found)


def profile_audio_file(path: str) -> AudioProfile:
    stat    = os.stat(path)
    size_mb = stat.st_size / 1e6
    try:
        info        = sf.info(path)
        duration_s  = info.duration
        sample_rate = info.samplerate
        channels    = info.channels
        encoding    = info.subtype
    except Exception:
        try:
            y, sr       = librosa.load(path, sr=None, mono=False, duration=5.0)
            channels    = 1 if y.ndim == 1 else y.shape[0]
            sample_rate = sr
            encoding    = "unknown"
            duration_s  = size_mb * 1e6 / (sample_rate * channels * 2)
        except Exception:
            duration_s = 0.0; sample_rate = 16000; channels = 1; encoding = "unknown"
    estimated_ram_mb = duration_s * 16000 * 4 / 1e6 + 500.0
    return AudioProfile(path=path, file_size_mb=round(size_mb, 2),
                        duration_s=round(duration_s, 3), sample_rate=sample_rate,
                        channels=channels, encoding=encoding,
                        estimated_ram_mb=round(estimated_ram_mb, 1))


def _print_profile_table(profiles: List[AudioProfile]) -> None:
    _section("Audio File Profiles")
    hdr = f"  {'File':<40} {'Size MB':>8} {'Duration':>10} {'SR':>7} {'Ch':>3} {'Est RAM MB':>11}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for p in profiles:
        name = Path(p.path).name
        if len(name) > 38:
            name = "…" + name[-37:]
        dur_str = f"{p.duration_s:.1f}s" if p.duration_s < 60 else f"{p.duration_s/60:.1f}m"
        print(f"  {name:<40} {p.file_size_mb:>8.1f} {dur_str:>10} "
              f"{p.sample_rate:>7} {p.channels:>3} {p.estimated_ram_mb:>11.0f}")
    total_dur  = sum(p.duration_s for p in profiles)
    total_size = sum(p.file_size_mb for p in profiles)
    print("  " + "─" * (len(hdr) - 2))
    print(f"  {'TOTAL':<40} {total_size:>8.1f} {total_dur/60:>9.1f}m")


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE-FILE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run(cfg: dict) -> DiarizationResult:
    """
    Process one audio file. In batch mode all intermediate prints are
    suppressed by the _BATCH_MODE flag; only the final RTTM/CTM paths and
    per-speaker report are printed (by run_batch) after the file completes.
    """
    os.makedirs(cfg["output_dir"], exist_ok=True)
    audio_name = os.path.splitext(os.path.basename(cfg["audio_path"]))[0]
    rttm_path  = os.path.join(cfg["output_dir"], f"{audio_name}.rttm")
    ctm_path   = os.path.join(cfg["output_dir"], f"{audio_name}.ctm")

    # ── 1. Load audio ─────────────────────────────────────────────────────
    _banner("Step 1 / 4  —  Loading audio")
    t0 = time.perf_counter()
    audio_np, sr_orig = sf.read(cfg["audio_path"], dtype="float32", always_2d=False)
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    if sr_orig != 16000:
        audio_np = librosa.resample(audio_np, orig_sr=sr_orig, target_sr=16000)
    audio    = torch.from_numpy(audio_np)
    duration = len(audio_np) / 16000
    _ok(f"File      : {cfg['audio_path']}")
    _ok(f"Duration  : {duration:.3f} s  ({duration/60:.1f} min)")
    _ok(f"Samples   : {len(audio_np):,}  @  16 000 Hz")
    _ok(f"Loaded in : {time.perf_counter()-t0:.2f} s")

    # ── 2. VAD ────────────────────────────────────────────────────────────
    _banner("Step 2 / 4  —  Voice Activity Detection (Silero ONNX)")
    t0 = time.perf_counter()
    vad = OnnxVAD(cfg["silero_onnx_path"])
    _inline_print("     Running VAD  ")
    timestamps = vad.get_speech_timestamps(
        audio,
        threshold               = cfg["vad_threshold"],
        min_speech_duration_ms  = cfg["min_speech_duration_ms"],
        min_silence_duration_ms = cfg["min_silence_duration_ms"],
        speech_pad_ms           = cfg["speech_pad_ms"],
        return_seconds          = True,
        progress_cb             = _progress,
    )
    if not _BATCH_MODE:
        print()
    _ok(f"Speech segments  : {len(timestamps)}")
    _ok(f"VAD time         : {time.perf_counter()-t0:.2f} s")
    speech_total = sum(s["end"] - s["start"] for s in timestamps)
    _ok(f"Speech coverage  : {speech_total:.2f} s  "
        f"({speech_total / duration * 100:.1f}% of audio)")
    if not timestamps:
        if not _BATCH_MODE:
            print("\n  \033[93m⚠  No speech detected. Exiting.\033[0m\n")
            sys.exit(0)
        return DiarizationResult(
            audio_path=cfg["audio_path"], segments=[], labels=[], duration=duration
        )

    # ── 3. Embeddings ─────────────────────────────────────────────────────
    _banner("Step 3 / 4  —  Speaker Embedding (TitaNet ONNX, parallel)")
    t0 = time.perf_counter()
    embedder  = TitaNetEmbedder(cfg["titanet_onnx_path"],
                                num_workers=cfg["num_embedding_workers"])
    completed = [0]
    def _emb_progress(idx: int, total: int) -> None:
        completed[0] += 1
        _progress(completed[0] / total * 100)

    _inline_print("     Extracting   ")
    valid_segments, embeddings = embedder.extract_batch(
        timestamps, audio_np,
        sr             = 16000,
        min_duration_s = cfg["min_segment_duration"],
        progress_cb    = _emb_progress,
    )
    if not _BATCH_MODE:
        print()
    _ok(f"Valid segments   : {len(valid_segments)}  "
        f"(skipped {len(timestamps) - len(valid_segments)} short segments)")
    _ok(f"Embedding time   : {time.perf_counter()-t0:.2f} s")

    # ── 4. Clustering ─────────────────────────────────────────────────────
    _banner("Step 4 / 4  —  Speaker Clustering (agglomerative cosine)")
    t0 = time.perf_counter()
    labels       = cluster_speakers(embeddings, cfg["similarity_threshold"])
    num_speakers = len(set(labels))
    _ok(f"Speakers found   : {num_speakers}")
    _ok(f"Cluster time     : {time.perf_counter()-t0:.2f} s")

    # ── 5. Output ─────────────────────────────────────────────────────────
    result = DiarizationResult(
        audio_path=cfg["audio_path"],
        segments=valid_segments,
        labels=labels,
        duration=duration,
    )
    write_rttm(result, rttm_path)
    write_ctm(result,  ctm_path)

    if not _BATCH_MODE:
        print_report(result, rttm_path=rttm_path, ctm_path=ctm_path)
        print(f"  \033[1;92m  Files written:\033[0m")
        print(f"    RTTM  →  {rttm_path}")
        print(f"    CTM   →  {ctm_path}\n")

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  BATCH  —  per-file worker (runs in a thread)
# ══════════════════════════════════════════════════════════════════════════════

_print_lock = threading.Lock()


def _process_one(audio_path: str, cfg: dict, profile: AudioProfile) -> FileResult:
    t_start  = time.perf_counter()
    file_cfg = dict(cfg, audio_path=audio_path)
    try:
        result     = run(file_cfg)
        audio_name = os.path.splitext(os.path.basename(audio_path))[0]
        rttm_path  = os.path.join(cfg["output_dir"], f"{audio_name}.rttm")
        ctm_path   = os.path.join(cfg["output_dir"], f"{audio_name}.ctm")
        speech_s   = (sum(s["end"] - s["start"] for s in result.segments)
                      if result.segments else 0.0)

        # ── Print final per-file result atomically ─────────────────────
        with _print_lock:
            print_report(result, rttm_path=rttm_path, ctm_path=ctm_path)
            print(f"    RTTM  →  {rttm_path}")
            print(f"    CTM   →  {ctm_path}\n")

        return FileResult(
            audio_path   = audio_path,
            status       = "success",
            rttm_path    = rttm_path,
            ctm_path     = ctm_path,
            num_speakers = len(set(result.labels)) if result.labels else 0,
            num_segments = len(result.segments),
            duration_s   = result.duration,
            speech_s     = speech_s,
            wall_time_s  = round(time.perf_counter() - t_start, 2),
            profile      = profile.as_dict(),
        )
    except Exception as exc:
        log.error(f"[FAILED] {audio_path}: {exc}")
        return FileResult(
            audio_path  = audio_path,
            status      = "failed",
            error       = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            wall_time_s = round(time.perf_counter() - t_start, 2),
            profile     = profile.as_dict(),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  BATCH ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_batch(cfg: dict) -> BatchSummary:
    global _BATCH_MODE
    _BATCH_MODE = True   # suppress all intermediate per-step prints

    batch_dir  = cfg["batch_input_dir"]
    output_dir = cfg["output_dir"]
    extensions = cfg.get("batch_extensions", list(AUDIO_EXTENSIONS))

    os.makedirs(output_dir, exist_ok=True)
    started_at = datetime.now().isoformat(timespec="seconds")
    t_batch    = time.perf_counter()

    # ── 1. Discover ───────────────────────────────────────────────────────
    _section("Batch Mode  —  Discovering Files")
    audio_files = discover_audio_files(batch_dir, extensions)
    if not audio_files:
        _warn(f"No audio files found in '{batch_dir}' with extensions {extensions}")
        sys.exit(0)
    print(f"     \033[92m✓\033[0m  Found {len(audio_files)} file(s) in {batch_dir}")

    # ── 2. Profile ────────────────────────────────────────────────────────
    _section("Resource-Aware Scheduling  —  Profiling Files")
    profiles: Dict[str, AudioProfile] = {}
    for i, fp in enumerate(audio_files):
        profiles[fp] = profile_audio_file(fp)
        filled = int((i + 1) / len(audio_files) * 20)
        bar    = "█" * filled + "░" * (20 - filled)
        print(f"\r     [{bar}] {(i+1)/len(audio_files)*100:5.1f}%", end="", flush=True)
    print()
    profile_list = list(profiles.values())
    _print_profile_table(profile_list)

    # ── 3. Resource snapshot ──────────────────────────────────────────────
    _section("Resource-Aware Scheduling  —  System Resources")
    monitor = ResourceMonitor(
        max_cpu_pct    = cfg.get("max_cpu_percent", 85.0),
        max_ram_gb     = cfg.get("max_memory_gb"),
        gpu_reserve_gb = cfg.get("gpu_memory_reserve_gb", 1.0),
    )
    snap = monitor.snapshot()
    print(f"     \033[92m✓\033[0m  CPU cores  : {psutil.cpu_count(logical=False)} physical / "
          f"{psutil.cpu_count(logical=True)} logical")
    print(f"     \033[92m✓\033[0m  RAM        : {snap['ram_avail_gb']:.1f} GB available "
          f"/ {snap['ram_total_gb']:.1f} GB total")
    print(f"     \033[92m✓\033[0m  RAM limit  : {monitor.max_ram_gb:.1f} GB")
    if "gpu_total_gb" in snap:
        print(f"     \033[92m✓\033[0m  GPU        : {snap['gpu_free_gb']:.1f} GB free "
              f"/ {snap['gpu_total_gb']:.1f} GB total")
    else:
        print(f"     \033[92m✓\033[0m  GPU        : not detected")

    num_workers = _optimal_workers(profile_list, cfg, monitor)
    print(f"     \033[92m✓\033[0m  Workers    : {num_workers}")

    if cfg.get("dry_run"):
        _warn("--dry-run: exiting before processing.")
        sys.exit(0)

    # ── 4. Dispatch ───────────────────────────────────────────────────────
    _section(f"Processing {len(audio_files)} file(s)  —  {num_workers} parallel worker(s)")

    results: List[FileResult] = []
    fail_fast    = cfg.get("batch_fail_fast", False)
    sorted_files = sorted(audio_files, key=lambda f: profiles[f].estimated_ram_mb)
    done_count   = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_map = {}
        for fp in sorted_files:
            while True:
                ok, reason = monitor.can_schedule(profiles[fp].estimated_ram_mb)
                if ok:
                    break
                time.sleep(cfg.get("resource_poll_secs", 2.0))
            future_map[executor.submit(_process_one, fp, cfg, profiles[fp])] = fp

        for future in as_completed(future_map):
            fp  = future_map[future]
            res = future.result()
            results.append(res)
            done_count += 1
            icon = "\033[92m✓\033[0m" if res.status == "success" else "\033[91m✗\033[0m"
            rt   = (f"  RT×{res.duration_s / res.wall_time_s:.1f}"
                    if res.duration_s and res.wall_time_s else "")
            spk  = (f"  {res.num_speakers} speaker(s)"
                    if res.num_speakers is not None else "")
            print(f"  [{done_count:>3}/{len(audio_files)}]  {icon}  "
                  f"{Path(fp).name:<40}  {res.status:<8}{spk}{rt}", flush=True)
            if res.status == "failed":
                print(f"         \033[91m↳ {res.error.splitlines()[0]}\033[0m")
                if fail_fast:
                    _err("--fail-fast: aborting.")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

    # ── 5. Summary ────────────────────────────────────────────────────────
    finished_at   = datetime.now().isoformat(timespec="seconds")
    total_wall    = time.perf_counter() - t_batch
    succeeded     = [r for r in results if r.status == "success"]
    failed        = [r for r in results if r.status == "failed"]
    skipped       = [r for r in results if r.status == "skipped"]
    total_audio_s = sum(r.duration_s or 0 for r in results)
    rtf           = total_wall / total_audio_s if total_audio_s > 0 else 0.0

    summary = BatchSummary(
        batch_dir=batch_dir, output_dir=output_dir,
        started_at=started_at, finished_at=finished_at,
        total_files=len(audio_files), succeeded=len(succeeded),
        failed=len(failed), skipped=len(skipped),
        total_audio_s=round(total_audio_s, 2), total_wall_s=round(total_wall, 2),
        realtime_factor=round(rtf, 3),
        peak_cpu_percent=round(monitor.peak_cpu, 1),
        peak_ram_gb=round(monitor.peak_ram_gb, 2),
        workers_used=num_workers,
        files=[r.as_dict() for r in results],
    )

    json_path = os.path.join(output_dir, "batch_results.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary.as_dict(), fh, indent=2, ensure_ascii=False)

    _section("Batch Summary")
    print(f"     \033[92m✓\033[0m  Total      : {summary.total_files}  "
          f"(✓ {summary.succeeded}  ✗ {summary.failed}  — {summary.skipped})")
    print(f"     \033[92m✓\033[0m  Audio      : {total_audio_s/60:.1f} min")
    print(f"     \033[92m✓\033[0m  Wall time  : {total_wall/60:.1f} min")
    print(f"     \033[92m✓\033[0m  RT factor  : {rtf:.3f}×  "
          f"({'faster' if rtf < 1 else 'slower'} than real-time)")
    print(f"     \033[92m✓\033[0m  Peak CPU   : {summary.peak_cpu_percent:.1f}%")
    print(f"     \033[92m✓\033[0m  Peak RAM   : {summary.peak_ram_gb:.2f} GB")
    print(f"     \033[92m✓\033[0m  Workers    : {summary.workers_used}")
    print(f"\n  \033[1;92m  Batch JSON  →  {json_path}\033[0m\n")

    if failed:
        _section("Failed Files")
        for r in failed:
            _err(f"{Path(r.audio_path).name}")
            print(f"       {r.error.splitlines()[0] if r.error else 'unknown'}")

    return summary


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> dict:
    p = argparse.ArgumentParser(description="Offline speaker diarization pipeline")
    p.add_argument("--audio",         default=None, help="Single audio file")
    p.add_argument("--silero",        default=None, help="silero_vad.onnx path")
    p.add_argument("--titanet",       default=None, help="titanet_large.onnx path")
    p.add_argument("--outdir",        default=None, help="Output directory")
    p.add_argument("--threshold",     type=float,   default=None)
    p.add_argument("--workers",       type=int,     default=None, help="Embedding threads")
    p.add_argument("--batch-dir",     default=None, help="Directory of audio files")
    p.add_argument("--batch-workers", type=int,     default=None, help="Parallel file workers")
    p.add_argument("--extensions",    nargs="+",    default=None)
    p.add_argument("--fail-fast",     action="store_true")
    p.add_argument("--max-memory-gb", type=float,   default=None)
    p.add_argument("--max-cpu-pct",   type=float,   default=None)
    p.add_argument("--dry-run",       action="store_true")

    args = p.parse_args()
    cfg  = dict(CONFIG)

    if args.audio:         cfg["audio_path"]          = args.audio
    if args.silero:        cfg["silero_onnx_path"]     = args.silero
    if args.titanet:       cfg["titanet_onnx_path"]    = args.titanet
    if args.outdir:        cfg["output_dir"]            = args.outdir
    if args.threshold:     cfg["similarity_threshold"]  = args.threshold
    if args.workers:       cfg["num_embedding_workers"] = args.workers
    if args.batch_dir:     cfg["batch_input_dir"]       = args.batch_dir
    if args.batch_workers: cfg["batch_workers"]         = args.batch_workers
    if args.extensions:    cfg["batch_extensions"]      = args.extensions
    if args.fail_fast:     cfg["batch_fail_fast"]       = True
    if args.max_memory_gb: cfg["max_memory_gb"]         = args.max_memory_gb
    if args.max_cpu_pct:   cfg["max_cpu_percent"]       = args.max_cpu_pct
    if args.dry_run:       cfg["dry_run"]               = True

    return cfg


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = _parse_args()
    if cfg.get("batch_input_dir"):
        run_batch(cfg)
    else:
        run(cfg)
