"""
output/reporter.py
──────────────────
Professional diarization output:
  • RTTM file  (NIST standard, readable by pyannote / dscore)
  • CTM  file  (word-level compatible placeholder)
  • Rich console report  (timing table + per-speaker summary)

Usage
-----
    from output.reporter import DiarizationResult, write_rttm, print_report

    result = DiarizationResult(
        audio_path  = "/path/to/audio.wav",
        segments    = [{'start': 0.5, 'end': 2.1}, ...],
        labels      = [0, 1, 0, ...],
        duration    = 120.0,
    )
    write_rttm(result, "/path/to/out.rttm")
    print_report(result)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Turn:
    """One speaker turn."""
    start    : float
    end      : float
    speaker  : str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class DiarizationResult:
    """
    Holds the full diarization output.

    Parameters
    ----------
    audio_path : str
        Source audio file (used for RTTM / file naming).
    segments   : list of dicts with 'start' / 'end' keys (seconds).
    labels     : zero-based speaker index per segment (same order).
    duration   : total audio duration in seconds.
    """
    audio_path : str
    segments   : List[dict]
    labels     : List[int]
    duration   : float
    turns      : List[Turn] = field(init=False)

    def __post_init__(self) -> None:
        self.turns = sorted(
            [
                Turn(
                    start   = seg["start"],
                    end     = seg["end"],
                    speaker = f"SPEAKER_{lbl:02d}",
                )
                for seg, lbl in zip(self.segments, self.labels)
            ],
            key=lambda t: t.start,
        )

    @property
    def file_id(self) -> str:
        return os.path.splitext(os.path.basename(self.audio_path))[0]

    @property
    def num_speakers(self) -> int:
        return len({t.speaker for t in self.turns})

    def speaker_stats(self) -> dict:
        stats: dict[str, dict] = {}
        for t in self.turns:
            s = stats.setdefault(t.speaker, {"total": 0.0, "segments": 0})
            s["total"]    += t.duration
            s["segments"] += 1
        return dict(sorted(stats.items()))


# ─────────────────────────────────────────────────────────────────────────────
# RTTM writer  (NIST Rich Transcription Time Marks)
# ─────────────────────────────────────────────────────────────────────────────

def write_rttm(result: DiarizationResult, out_path: str) -> None:
    """
    Write a standard RTTM file.

    Format per line:
        SPEAKER <file_id> <chn> <start> <dur> <NA> <NA> <spkr> <NA> <NA>
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for t in result.turns:
            f.write(
                f"SPEAKER {result.file_id} 1 "
                f"{t.start:.3f} {t.duration:.3f} "
                f"<NA> <NA> {t.speaker} <NA> <NA>\n"
            )


# ─────────────────────────────────────────────────────────────────────────────
# CTM writer  (NIST Conversation Time Marks — segment-level placeholder)
# ─────────────────────────────────────────────────────────────────────────────

def write_ctm(result: DiarizationResult, out_path: str) -> None:
    """
    Write a segment-level CTM file (token field = speaker label).

    Format per line:
        <file_id> <chn> <start> <dur> <token> [<conf>]
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for t in result.turns:
            f.write(
                f"{result.file_id} A "
                f"{t.start:.3f} {t.duration:.3f} "
                f"{t.speaker} 1.000\n"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Console reporter
# ─────────────────────────────────────────────────────────────────────────────

# ANSI colour helpers (gracefully degrade on non-TTY)
def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m"

_BOLD   = lambda t: _c("1",     t)
_DIM    = lambda t: _c("2",     t)
_CYAN   = lambda t: _c("96",    t)
_GREEN  = lambda t: _c("92",    t)
_YELLOW = lambda t: _c("93",    t)
_BLUE   = lambda t: _c("94",    t)
_MAGENTA= lambda t: _c("95",    t)
_WHITE  = lambda t: _c("97",    t)

# One colour per speaker (cycles if > 8 speakers)
_SPK_COLOURS = ["94", "92", "93", "95", "96", "91", "33", "32"]

def _spk_colour(spk: str) -> str:
    idx = int(spk.split("_")[-1]) % len(_SPK_COLOURS)
    return _SPK_COLOURS[idx]


def print_report(result: DiarizationResult, rttm_path: str = "", ctm_path: str = "") -> None:
    """Print a professional diarization report to stdout."""
    W = 72   # column width

    def _header(text: str) -> None:
        print(_BOLD(_CYAN(f"\n  {'─' * (W - 4)}")))
        print(_BOLD(_WHITE(f"  {text}")))
        print(_BOLD(_CYAN(f"  {'─' * (W - 4)}")))

    def _rule() -> None:
        print(_DIM(f"  {'·' * (W - 4)}"))

    # ── title bar ────────────────────────────────────────────────────────
    print()
    print(_BOLD(_CYAN("  " + "═" * (W - 2))))
    print(_BOLD(_WHITE(f"  {'SPEAKER DIARIZATION REPORT':^{W - 2}}")))
    print(_BOLD(_CYAN("  " + "═" * (W - 2))))

    # ── meta ─────────────────────────────────────────────────────────────
    print()
    print(f"  {_BOLD('Audio file')}   :  {result.audio_path}")
    print(f"  {_BOLD('Duration')}     :  {result.duration:.3f} s "
          f"({result.duration/60:.1f} min)")
    print(f"  {_BOLD('Segments')}     :  {len(result.turns)}")
    print(f"  {_BOLD('Speakers')}     :  {result.num_speakers}")
    if rttm_path:
        print(f"  {_BOLD('RTTM')}         :  {rttm_path}")
    if ctm_path:
        print(f"  {_BOLD('CTM')}          :  {ctm_path}")

    # ── timeline table ────────────────────────────────────────────────────
    _header("TURN-LEVEL TIMELINE")
    col_w = [6, 9, 9, 8, 14, 20]
    hdr   = (
        f"  {'#':>{col_w[0]}}  "
        f"{'START':>{col_w[1]}}  "
        f"{'END':>{col_w[2]}}  "
        f"{'DUR':>{col_w[3]}}  "
        f"{'SPEAKER':<{col_w[4]}}  "
        f"{'ACTIVITY BAR':<{col_w[5]}}"
    )
    print(_BOLD(_DIM(hdr)))
    _rule()

    for i, t in enumerate(result.turns, 1):
        bar_len = max(1, int(t.duration * 5))
        bar     = "█" * bar_len
        cc      = _spk_colour(t.speaker)
        print(
            f"  {_DIM(str(i)):>{col_w[0]+9}}  "
            f"{t.start:>{col_w[1]}.3f}s  "
            f"{t.end:>{col_w[2]}.3f}s  "
            f"{t.duration:>{col_w[3]}.2f}s  "
            f"\033[{cc}m{t.speaker:<{col_w[4]}}\033[0m  "
            f"\033[{cc}m{bar}\033[0m"
        )

    # ── speaker summary ───────────────────────────────────────────────────
    _header("SPEAKER SUMMARY")
    stats = result.speaker_stats()
    shdr  = (
        f"  {'SPEAKER':<14}  {'TOTAL':>8}  "
        f"{'SEGMENTS':>9}  {'% AUDIO':>8}  {'SHARE BAR'}"
    )
    print(_BOLD(_DIM(shdr)))
    _rule()

    for spk, info in stats.items():
        pct     = info["total"] / result.duration * 100
        bar_len = max(1, int(pct / 2))
        bar     = "▓" * bar_len
        cc      = _spk_colour(spk)
        print(
            f"  \033[{cc}m{spk:<14}\033[0m  "
            f"{info['total']:>7.2f}s  "
            f"{info['segments']:>9}  "
            f"{pct:>7.1f}%  "
            f"\033[{cc}m{bar}\033[0m"
        )

    # ── overlap check ─────────────────────────────────────────────────────
    overlaps = _find_overlaps(result.turns)
    _header("OVERLAP ANALYSIS")
    if overlaps:
        for ov in overlaps:
            print(f"  {_YELLOW('⚠')}  {ov['start']:.3f}s → {ov['end']:.3f}s  "
                  f"({ov['spk_a']} ∩ {ov['spk_b']})  "
                  f"dur={ov['duration']:.3f}s")
    else:
        print(f"  {_GREEN('✓')}  No overlapping speech detected.")

    print()
    print(_BOLD(_CYAN("  " + "═" * (W - 2))))
    print()


def _find_overlaps(turns: list) -> list:
    overlaps = []
    for i in range(len(turns)):
        for j in range(i + 1, len(turns)):
            a, b = turns[i], turns[j]
            if b.start >= a.end:
                break
            overlap_start = max(a.start, b.start)
            overlap_end   = min(a.end,   b.end)
            if overlap_end > overlap_start:
                overlaps.append({
                    "start"    : overlap_start,
                    "end"      : overlap_end,
                    "duration" : overlap_end - overlap_start,
                    "spk_a"    : a.speaker,
                    "spk_b"    : b.speaker,
                })
    return overlaps
