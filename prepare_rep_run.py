#!/usr/bin/env python3
"""
Prepare one block-C run (repetitions on the hold-out) as an isolated git worktree.

Creates runs/REP_SOL_r<n> or runs/REP_GEMMA_r<n> at the pinned commit, without
the reference episode folders and databases, then fills it with:
  - data/episodes/<folder>/analysis/: the July frames/ and analysis_manifest.json
    of the 8 hold-out folders (207 frames), so every repetition sends exactly
    the same frames (Stage I and II are not rerun);
  - db/: IMDb files only, no .db; .env copied from the main checkout;
  - .env (Gemma only): absolute paths of the GGUF files in the main checkout
  - run_info.json with commit, model, parameters, stopwords and the frames.

Stage III and IV are run by run_headless.py. No model call is made here.

Usage:
    python prepare_rep_run.py --model sol --rep 1     # runs/REP_SOL_r1
    python prepare_rep_run.py --model gemma --rep 1   # runs/REP_GEMMA_r1
"""

import argparse
import hashlib
import json
import shutil
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

FOLDERS = [
    "3_Percent_S01E06_End", "3_Percent_S01E06_Opening", "Blue_Eye_Samurai_S01E01_End",
    "Honeyland_End", "Persepolis_End", "Persepolis_Opening",
    "Wild_Strawberries_End", "Wild_Strawberries_Opening",
]
EXPECTED_FRAMES = 207
GEMMA_FILES = {
    "model": MAIN_ROOT / "bin" / "gemma-4-12b-it-qat-q4_0.gguf",
    "mmproj": MAIN_ROOT / "bin" / "mmproj-gemma-4-12b-it-qat-q4_0.gguf",
}

MODELS = {
    "sol": {
        "prefix": "REP_SOL",
        "export": "FUZZY88_GPT_SOL_STANDARD_5products",
        "pricing": PRICING_USD_PER_1M,
        "params": {},
    },
    "gemma": {
        "prefix": "REP_GEMMA",
        "export": "FUZZY88_gemma_4_12B_5products",
        "pricing": {"input": 0.0, "cached_input": 0.0, "output": 0.0},
        # Same settings as the July Gemma run: provider defaults of vlm_processing.py at f9c9bb6.
        "params": {
            "vlm_provider": "gemma12b",
            "model_deployment": "gemma-4-12b-it-qat-q4_0.gguf + mmproj-gemma-4-12b-it-qat-q4_0.gguf (llama-cpp-python, local)",
            "api_version": None,
            "reasoning": None,
            "prompt_cache": None,
            "gemma_runtime": {"n_ctx": 16384, "n_gpu_layers": -1, "flash_attn": True, "temperature": 0.0},
        },
    },
}


def copy_frames(run_dir: Path) -> dict[str, list[str]]:
    submitted = {}
    for folder in FOLDERS:
        src = MAIN_ROOT / "data" / "episodes" / folder / "analysis"
        dst = run_dir / "data" / "episodes" / folder / "analysis"
        shutil.copytree(src / "frames", dst / "frames")
        shutil.copy2(src / "analysis_manifest.json", dst / "analysis_manifest.json")
        submitted[folder] = sorted(p.name for p in (dst / "frames").glob("*.jpg"))
    total = sum(map(len, submitted.values()))
    if total != EXPECTED_FRAMES:
        sys.exit(f"Hold-out has {total} frames, expected {EXPECTED_FRAMES}")
    return submitted


def copy_support_files(run_dir: Path, model: str) -> None:
    shutil.copy2(MAIN_ROOT / ".env", run_dir / ".env")
    for name in IMDB_FILES:
        shutil.copy2(MAIN_ROOT / "db" / name, run_dir / "db" / name)
    if model == "gemma":
        # Absolute paths in the run's own .env, never a link into the worktree: a forced
        # `git worktree remove` deletes through junctions (it wiped bin/ once).
        with open(run_dir / ".env", "a", encoding="utf-8") as env:
            env.write(f'\nGEMMA12B_MODEL_GGUF="{GEMMA_FILES["model"].as_posix()}"\nGEMMA12B_MMPROJ_GGUF="{GEMMA_FILES["mmproj"].as_posix()}"\n')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def write_run_info(run_dir: Path, run_id: str, model: str, rep: int, submitted: dict[str, list[str]]) -> None:
    spec = MODELS[model]
    info = {
        "run_id": run_id,
        "block": "C - repetitions on the hold-out",
        "repetition": rep,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD", cwd=run_dir),
        "code_modifications": [],
        "parameters": {
            "stages": ["stage3", "stage4"],
            "naive_mode": False,
            "stage1_stage2": "not rerun: July frames/ + analysis_manifest.json of the hold-out copied, "
                             "identical in every repetition",
            **model_params(run_dir),
            **spec["params"],
        },
        "source": {
            "frames_from": "July 2026 hold-out Stage II (2026-07-18..23, PaddleOCR lang=en, before 5fec349), "
                           "data/episodes/<folder>/analysis of the main checkout",
            "july_reference_db": "db/FUZZY88_GPT_SOL_STANDARD_25products_tvcredits_v3.db",
        },
        **({"gemma_files": {
            k: {"path": str(p), "size_bytes": p.stat().st_size, "sha256": sha256(p)} for k, p in GEMMA_FILES.items()
        }, "gemma_files_note": "Re-downloaded on 2026-10-01 after an accidental deletion; sizes identical to the "
                               "July files (6975879296 and 175115616 bytes); July hashes were not recorded."}
           if model == "gemma" else {}),
        "stopwords": stopwords_info(run_dir),
        "pricing_usd_per_1M_tokens": spec["pricing"],
        "expected_export_csv": f"{run_id}_{spec['export']}.csv",
        "submitted_frames_expected_total": EXPECTED_FRAMES,
        "submitted_frames_total": sum(map(len, submitted.values())),
        "submitted_frames_per_folder": {f: len(v) for f, v in submitted.items()},
        "submitted_frames": submitted,
    }
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--rep", type=int, required=True)
    args = parser.parse_args()

    run_id = f"{MODELS[args.model]['prefix']}_r{args.rep}"
    run_dir = MAIN_ROOT / "runs" / run_id
    if run_dir.exists():
        sys.exit(f"{run_dir} already exists: a run is never overwritten")

    create_worktree(run_dir)
    submitted = copy_frames(run_dir)
    copy_support_files(run_dir, args.model)
    write_run_info(run_dir, run_id, args.model, args.rep, submitted)
    print(f"{run_id}: prepared, {sum(map(len, submitted.values()))} frames")


if __name__ == "__main__":
    main()
