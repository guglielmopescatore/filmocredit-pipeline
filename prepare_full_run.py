#!/usr/bin/env python3
"""
Prepare the block-D run (full-length films, end to end) as an isolated git worktree.

Creates runs/FULL_pipeline at the pinned commit, without the reference episode
folders and databases, then fills it with:
  - data/raw/main_corpus_full/: hard links to the 20 full-length files of the
    main checkout (<Title>_FULL.<ext>, so nothing collides with the cut clips);
  - db/: IMDb files only, no .db; .env copied from the main checkout;
  - run_info.json with commit, parameters, stopwords and, per video, MD5,
    size, duration and fps (ffprobe).

Stage I ("Run on the whole episode", no manual deselection), II (cap 150),
III and IV are run by run_headless.py. No model call is made here.

Usage:
    python prepare_full_run.py
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from prepare_naive_run import (
    IMDB_FILES,
    MAIN_ROOT,
    PRICING_USD_PER_1M,
    create_worktree,
    git,
    model_params,
    stopwords_info,
)

RUN_ID = "FULL_pipeline"
CORPUS = "main_corpus_full"
EXPECTED_FILES = 20
OCR_LANGUAGE = "en"  # as the reference Stage I/II runs (see revision_checks/)
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
CHUNK = 1 << 24


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=avg_frame_rate,width,height,codec_name:format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(out)
    stream = data["streams"][0]
    num, den = (int(x) for x in stream["avg_frame_rate"].split("/"))
    return {
        "duration_seconds": round(float(data["format"]["duration"]), 3),
        "fps": round(num / den, 3) if den else None,
        "resolution": f"{stream['width']}x{stream['height']}",
        "codec": stream["codec_name"],
    }


def link_videos(run_dir: Path) -> dict[str, dict]:
    src_dir = MAIN_ROOT / "data" / "raw" / CORPUS
    dst_dir = run_dir / "data" / "raw" / CORPUS
    dst_dir.mkdir(parents=True)
    videos = {}
    for src in sorted(p for p in src_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS):
        if not src.stem.endswith("_FULL"):
            sys.exit(f"{src.name}: full-length files must be named <Title>_FULL.<ext>")
        os.link(src, dst_dir / src.name)
        print(f"  {src.name}: md5...", flush=True)
        videos[src.stem] = {"file": src.name, "size_bytes": src.stat().st_size, "md5": md5(src), **probe(src)}
    if len(videos) != EXPECTED_FILES:
        sys.exit(f"Found {len(videos)} full-length files, expected {EXPECTED_FILES}")
    return videos


def copy_support_files(run_dir: Path) -> None:
    import shutil

    shutil.copy2(MAIN_ROOT / ".env", run_dir / ".env")
    for name in IMDB_FILES:
        shutil.copy2(MAIN_ROOT / "db" / name, run_dir / "db" / name)


def write_run_info(run_dir: Path, videos: dict[str, dict]) -> None:
    info = {
        "run_id": RUN_ID,
        "block": "D - full-length films, end to end",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD", cwd=run_dir),
        "code_modifications": [],
        "parameters": {
            "stages": ["stage1", "stage2", "stage3", "stage4"],
            "scene_selection": "whole_episode",
            "manual_scene_deselection": False,
            "SCROLL_MAX_FRAMES_PER_SAVE": 150,
            "naive_mode": False,
            "ocr_engine": "paddleocr",
            "ocr_language": OCR_LANGUAGE,
            **model_params(run_dir),
        },
        "videos": videos,
        "stopwords": stopwords_info(run_dir),
        "stopwords_note": "July list (34 entries, no NETFLIX), as decided for block D.",
        "pricing_usd_per_1M_tokens": PRICING_USD_PER_1M,
        "expected_export_csv": f"FULL_FUZZY88_GPT_SOL_STANDARD_20products.csv",
        # Filled by run_headless.py after Stage II.
        "submitted_frames_total": None,
        "submitted_frames": {ep: None for ep in videos},
    }
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    run_dir = MAIN_ROOT / "runs" / RUN_ID
    if run_dir.exists():
        sys.exit(f"{run_dir} already exists: a run is never overwritten")
    create_worktree(run_dir)
    videos = link_videos(run_dir)
    copy_support_files(run_dir)
    write_run_info(run_dir, videos)
    total = sum(v["duration_seconds"] for v in videos.values())
    print(f"{RUN_ID}: prepared, {len(videos)} films, {total / 3600:.1f} h of video")


if __name__ == "__main__":
    main()
