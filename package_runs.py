#!/usr/bin/env python3
"""
Package the second-revision runs (runs/<ID>/) for sharing, never with images,
.env, videos or IMDb files.

  python package_runs.py verification
      -> runs_verification_light.zip: for every run run_info.json, run.log, the
         export CSV, summary.json, calls.csv and the per-episode logs.
  python package_runs.py zenodo
      -> zenodo_second_revision/<ID>.zip, one per run, each with SHA256SUMS
         inside, plus zenodo_second_revision/SHA256SUMS of the zips.
         Every run: run_info.json, run.log, run_output/ (export, calls.csv,
         raw responses, summary, logs, timecodes), the run DB and the Stage I/II
         JSON files of every episode. FULL_pipeline: only the Stage I JSON files
         (raw_scenes_cache.json, initial_scene_analysis.json) and the list of
         selected frames with their timecodes.
"""

import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
RUN_IDS = (["NAIVE_2.4s", "NAIVE_4.8s", "PIPE_cap150", "PIPE_cap100", "PIPE_cap75", "PIPE_noscroll_cap150",
            "PIPE_noscroll_cap100", "PIPE_noscroll_cap75", "STAGE2_holdout_cap150"]
           + [f"REP_SOL_r{n}" for n in range(1, 6)] + [f"REP_GEMMA_r{n}" for n in range(1, 6)] + ["FULL_pipeline"])
FORBIDDEN_SUFFIXES = {".jpg", ".jpeg", ".png", ".mp4", ".mkv", ".mov", ".avi", ".webm", ".tsv", ".parquet", ".gguf"}
COMMIT_NOTE = ("Second-revision run of the Filmocredit pipeline, commit f9c9bb6 "
               "(github.com/guglielmopescatore/filmocredit-pipeline); analyses in analysis_2nd_revision/.\n")


def allowed(path: Path) -> bool:
    return path.name != ".env" and path.suffix.lower() not in FORBIDDEN_SUFFIXES


def verification_files(run: Path) -> list[Path]:
    out = run / "run_output"
    files = [run / "run_info.json", run / "run.log", out / "summary.json", out / "calls.csv"]
    files += sorted(out.glob("*_FUZZY88_*.csv"))
    files += sorted((out / "logs").glob("*.log")) if (out / "logs").is_dir() else []
    return [f for f in files if f.is_file()]


def zenodo_files(run: Path) -> list[Path]:
    episodes = run / "data" / "episodes"
    if run.name == "FULL_pipeline":
        files = [run / "run_output" / "stage2_frames_timecodes.csv"]
        for name in ("raw_scenes_cache.json", "initial_scene_analysis.json"):
            files += sorted(episodes.glob(f"*/analysis/{name}"))
        return [f for f in files if f.is_file()]
    files = [run / "run_info.json", run / "run.log", run / "db" / "tvcredits_v3.db"]
    files += [f for f in sorted((run / "run_output").rglob("*")) if f.is_file()]
    files += sorted(episodes.glob("*/analysis/*.json")) + sorted(episodes.glob("*/naive_analysis/*/*.json"))
    files += sorted(episodes.glob("*/analysis/*/*.json"))  # per-provider OCR JSON
    return [f for f in files if f.is_file()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_zip(target: Path, entries: list[tuple[Path, str]], readme: str) -> None:
    bad = [a for _, a in entries if not allowed(Path(a))]
    if bad:
        sys.exit(f"{target.name}: refusing forbidden files {bad[:5]}")
    sums = []
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for src, arc in entries:
            z.write(src, arc)
            sums.append(f"{sha256(src)}  {arc}")
        z.writestr("SHA256SUMS", "\n".join(sums) + "\n")
        z.writestr("README.txt", readme)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "verification":
        entries = []
        for rid in RUN_IDS:
            run = RUNS / rid
            entries += [(f, f"{rid}/{f.relative_to(run).as_posix()}") for f in verification_files(run)]
        target = ROOT / "runs_verification_light.zip"
        write_zip(target, entries, COMMIT_NOTE + "Light files of every run (no images): run_info.json, run.log, "
                                                 "export CSV, summary.json, calls.csv, per-episode logs.\n")
        print(f"{target.name}: {len(entries)} files, {target.stat().st_size / 1e6:.1f} MB")
    elif mode == "zenodo":
        out = ROOT / "zenodo_second_revision"
        out.mkdir(exist_ok=True)
        zip_sums = []
        for rid in RUN_IDS:
            run = RUNS / rid
            entries = [(f, f.relative_to(run).as_posix()) for f in zenodo_files(run)]
            target = out / f"{rid}.zip"
            scope = ("Stage I JSON files per film and the list of selected frames with timecodes only."
                     if rid == "FULL_pipeline" else
                     "run_info.json, run.log, run_output/, run DB, Stage I/II JSON per episode; no images.")
            write_zip(target, entries, f"{COMMIT_NOTE}Run {rid}. {scope}\n")
            zip_sums.append(f"{sha256(target)}  {target.name}")
            print(f"{target.name}: {len(entries)} files, {target.stat().st_size / 1e6:.1f} MB")
        (out / "SHA256SUMS").write_text("\n".join(zip_sums) + "\n", encoding="utf-8")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
