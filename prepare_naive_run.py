#!/usr/bin/env python3
"""
Prepare one block-A run (naive interval curve) as an isolated git worktree.

Creates runs/NAIVE_<interval>s at the pinned commit, without the July episode
folders and databases, then fills it with:
  - data/episodes/<folder>/naive_analysis/frames/: the July naive frames whose
    sequence number (naive_XXXXX_...) is divisible by k, names unchanged;
  - data/raw/main_corpus_cut/: copies of the 25 July clips (needed only so the
    GUI lists the episodes; naive Stage III reads the frames folder);
  - db/: IMDb files only (name.basics.tsv, normalized_names.parquet), no .db;
  - .env copied from the main checkout;
  - run_info.json with commit, parameters, stopwords and the submitted frames.

No model call is made here.

Usage:
    python prepare_naive_run.py --k 2
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MAIN_ROOT = Path(__file__).resolve().parent
RUN_COMMIT = "f9c9bb6"
BASE_INTERVAL_SECONDS = 0.8
FUZZY_THRESHOLD = 88
CORPUS_DIR_NAME = "main_corpus_cut"

FOLDERS = [
    "8_e_mezzo", "Amelie", "Apocalypse_Now", "Chernobyl_S01E01_End",
    "Chernobyl_S01E01_Opening", "Dark_1x9", "El_desorden_que_dejas_S01E03_End",
    "El_desorden_que_dejas_S01E03_Opening", "Eternal_Sunshine_of_the_Spotless_Mind",
    "Fight_Club", "Hill_Street_Blues_1x13", "La_grande_bellezza", "La_piovra_1x2",
    "Maigret_S03E01_End", "Maigret_S03E01_Opening", "Planet_Earth_S01E10",
    "Prime_Suspect_1x1", "Psycho", "Romanzo_criminale_S01E01_End",
    "Romanzo_criminale_S01E01_Opening", "Se7en", "The_World_At_War_S01E03_End",
    "The_World_At_War_S01E03_Opening", "Twin_Peaks_1x3", "Yes,_Prime_Minister_1x8",
]

# Expected submitted frames per k (rule applied folder by folder, count restarts at 0).
EXPECTED_FRAMES = {2: 3298, 3: 2201, 4: 1654, 5: 1328, 6: 1106}

# Azure list prices, USD per 1M text tokens (reasoning tokens are billed as output).
PRICING_USD_PER_1M = {"input": 4.00, "cached_input": 0.40, "output": 20.00}

NAIVE_NAME = re.compile(r"^naive_(\d+)_num\d+\.jpg$")
IMDB_FILES = ["name.basics.tsv", "normalized_names.parquet"]


def git(*args: str, cwd: Path = MAIN_ROOT) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_worktree(run_dir: Path) -> None:
    rel = run_dir.relative_to(MAIN_ROOT).as_posix()
    git("worktree", "add", "--detach", "--no-checkout", rel, RUN_COMMIT)
    # Sparse checkout keeps the July episode folders and databases out of the run.
    git("sparse-checkout", "set", "--no-cone", "/*", "!/data/episodes/", "!/db/*.db", cwd=run_dir)
    git("checkout", "--detach", RUN_COMMIT, cwd=run_dir)


def copy_frames(run_dir: Path, k: int) -> dict[str, list[str]]:
    submitted = {}
    for folder in FOLDERS:
        src = MAIN_ROOT / "data" / "episodes" / folder / "naive_analysis" / "frames"
        names = sorted(p.name for p in src.glob("*.jpg"))
        seqs = [int(NAIVE_NAME.match(n).group(1)) for n in names]
        if seqs != list(range(len(seqs))):
            sys.exit(f"{folder}: July naive sequence is not contiguous from 0")
        kept = [n for n, s in zip(names, seqs) if s % k == 0]
        dst = run_dir / "data" / "episodes" / folder / "naive_analysis" / "frames"
        dst.mkdir(parents=True)
        for name in kept:
            shutil.copy2(src / name, dst / name)
        submitted[folder] = kept
    return submitted


def copy_support_files(run_dir: Path, folders: list[str] = FOLDERS, corpus: str = CORPUS_DIR_NAME) -> None:
    shutil.copy2(MAIN_ROOT / ".env", run_dir / ".env")
    for name in IMDB_FILES:
        shutil.copy2(MAIN_ROOT / "db" / name, run_dir / "db" / name)
    raw_src = MAIN_ROOT / "data" / "raw" / corpus
    raw_dst = run_dir / "data" / "raw" / corpus
    raw_dst.mkdir(parents=True)
    for folder in folders:
        (clip,) = [p for p in raw_src.iterdir() if p.stem == folder]
        shutil.copy2(clip, raw_dst / clip.name)


def read_env(path: Path, key: str) -> str | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip().strip('"')
    return None


def stopwords_info(run_dir: Path) -> dict:
    stopwords_path = run_dir / "user_ocr_stopwords.txt"
    stopwords = [w.strip() for w in stopwords_path.read_text(encoding="utf-8").splitlines() if w.strip()]
    return {
        "file": "user_ocr_stopwords.txt",
        "count": len(stopwords),
        "sha256": sha256(stopwords_path),
        "words": stopwords,
    }


def model_params(run_dir: Path) -> dict:
    """Stage III/IV settings shared by every block (same as July)."""
    env = run_dir / ".env"
    return {
        "include_previous_frame_image": False,
        "incremental_prompting": "previous frame's LLM output (credits JSON) passed as text in the prompt",
        "vlm_provider": "azure_gpt_sol_standard",
        "model_deployment": read_env(env, "GPT_SOL_AZURE_OPENAI_DEPLOYMENT_NAME"),
        "api_version": read_env(env, "GPT_TERRA_AZURE_API_VERSION"),
        "reasoning": {"mode": "standard", "effort": "medium"},
        "prompt_cache": "Azure default, no cache parameter sent (code unchanged)",
        "fuzzy_matching_enabled": True,
        "fuzzy_threshold": FUZZY_THRESHOLD,
    }


def write_run_info(run_dir: Path, run_id: str, k: int, submitted: dict[str, list[str]]) -> None:
    total = sum(len(v) for v in submitted.values())
    info = {
        "run_id": run_id,
        "block": "A - naive interval curve",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD", cwd=run_dir),
        "code_modifications": [],
        "parameters": {
            "k": k,
            "nominal_interval_seconds": round(BASE_INTERVAL_SECONDS * k, 1),
            "NAIVE_FRAME_INTERVAL_SECONDS": BASE_INTERVAL_SECONDS,
            "subsampling_rule": "keep July naive frames whose sequence number XXXXX in naive_XXXXX_numYYYYYY.jpg is divisible by k; count restarts at 0 in every folder; filenames unchanged",
            "naive_mode": True,
            **model_params(run_dir),
        },
        "source": {
            "frames_from": "July 2026 naive run (0.8 s), data/episodes/<folder>/naive_analysis/frames at commit " + RUN_COMMIT,
            "july_reference_db": "db/NAIVE_FUZZY88_GPT_SOL_STANDARD_20products_tvcredits_v3.db",
        },
        "stopwords": stopwords_info(run_dir),
        "pricing_usd_per_1M_tokens": PRICING_USD_PER_1M,
        "expected_export_csv": f"{run_id}_FUZZY88_GPT_SOL_STANDARD_20products.csv",
        "submitted_frames_expected_total": EXPECTED_FRAMES[k],
        "submitted_frames_total": total,
        "submitted_frames_per_folder": {f: len(v) for f, v in submitted.items()},
        "submitted_frames": submitted,
    }
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, required=True, choices=sorted(EXPECTED_FRAMES))
    k = parser.parse_args().k

    run_id = f"NAIVE_{BASE_INTERVAL_SECONDS * k:.1f}s"
    run_dir = MAIN_ROOT / "runs" / run_id
    if run_dir.exists():
        sys.exit(f"{run_dir} already exists: a run is never overwritten")

    create_worktree(run_dir)
    submitted = copy_frames(run_dir, k)
    copy_support_files(run_dir)
    write_run_info(run_dir, run_id, k, submitted)

    total = sum(len(v) for v in submitted.values())
    status = "OK" if total == EXPECTED_FRAMES[k] else "MISMATCH"
    print(f"{run_id}: {total} frames (expected {EXPECTED_FRAMES[k]}) {status}")


if __name__ == "__main__":
    main()
