import csv
import re
from pathlib import Path

import numpy as np
from pydub import AudioSegment

# ---------------- CONFIG ----------------
INPUT_FILES = [
    "B05___01_Acts________NUSBSSN2DA.mp3",
    "B05___02_Acts________NUSBSSN2DA.mp3",
]
OUTPUT_ROOT = Path("nuer_chunks")

SAMPLE_RATE = 16000          # MMS expects 16 kHz
MIN_S = 4.0                  # shortest allowed chunk
TARGET_S = 7.0               # preferred chunk length
MAX_S = 11.0                 # longest allowed chunk (safe for a T4)

FRAME_MS = 20                # energy analysis resolution
SMOOTH_MS = 200              # smoothing, so a single click isn't mistaken for a pause
DIST_PENALTY_DB_PER_S = 1.5  # higher = stick closer to TARGET_S, lower = prefer the quietest spot
OUT_FORMAT = "flac"          # "flac" (smaller) or "wav"
# ----------------------------------------


def folder_name(path: Path) -> str:
    """B05___01_Acts________NUS... -> Acts_01 (falls back to the file stem)."""
    m = re.search(r"___(\d+)_([A-Za-z]+)", path.stem)
    return f"{m.group(2)}_{m.group(1)}" if m else path.stem


def find_spans(samples: np.ndarray):
    """Return a list of (start_frame, end_frame) chunk boundaries."""
    frame = SAMPLE_RATE * FRAME_MS // 1000
    n_frames = len(samples) // frame
    frames = samples[: n_frames * frame].reshape(n_frames, frame)
    rms = np.sqrt((frames ** 2).mean(axis=1))

    k = max(1, SMOOTH_MS // FRAME_MS)
    smooth = np.convolve(rms, np.ones(k) / k, mode="same")
    db = 20 * np.log10(smooth + 1e-6)

    min_f = int(MIN_S * 1000 / FRAME_MS)
    max_f = int(MAX_S * 1000 / FRAME_MS)

    spans, start = [], 0
    while True:
        remaining = n_frames - start
        if remaining <= max_f:                      # last chunk
            spans.append((start, n_frames))
            break
        lo = start + min_f
        hi = min(start + max_f, n_frames - min_f)   # never leave a too-short tail
        idx = np.arange(lo, hi + 1)
        dist_s = (idx - start) * FRAME_MS / 1000 - TARGET_S
        score = db[idx] + DIST_PENALTY_DB_PER_S * np.abs(dist_s)
        cut = int(idx[np.argmin(score)])
        spans.append((start, cut))
        start = cut
    return spans


def process_file(path: str):
    path = Path(path)
    if not path.exists():
        print(f"!! File not found: {path}")
        return

    out_dir = OUTPUT_ROOT / folder_name(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio = AudioSegment.from_file(path).set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0

    spans = find_spans(samples)
    total_ms = len(audio)
    rows, durations = [], []

    for i, (s_f, e_f) in enumerate(spans, start=1):
        s_ms = s_f * FRAME_MS
        e_ms = total_ms if i == len(spans) else e_f * FRAME_MS
        chunk = audio[s_ms:e_ms]

        name = f"{folder_name(path)}_{i:04d}.{OUT_FORMAT}"
        chunk.export(out_dir / name, format=OUT_FORMAT)

        durations.append((e_ms - s_ms) / 1000)
        rows.append({"file_name": name, "start_s": round(s_ms / 1000, 2), "end_s": round(e_ms / 1000, 2)})

    # Timestamp manifest, useful later when matching chunks to transcript text
    with open(out_dir / "chunks_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file_name", "start_s", "end_s"])
        w.writeheader()
        w.writerows(rows)

    d = np.array(durations)
    print(f"{path.name} -> {out_dir}/  |  {len(d)} chunks  |  "
          f"min {d.min():.1f}s  mean {d.mean():.1f}s  max {d.max():.1f}s")


if __name__ == "__main__":
    for f in INPUT_FILES:
        process_file(f)