#!/usr/bin/env python3
"""
Run Stage III (VLM OCR) + Stage IV (IMDb) on a prepared run worktree without
the GUI, then export the results. When run_info.json lists "stage2" in
parameters.stages (block B), Stage II (frame selection) runs first.

It calls the same functions as the GUI STEP 2 / STEP 3 / STEP 4 buttons
(frame_analysis.analyze_candidate_scene_frames on every candidate scene,
run_azure_vlm_ocr_on_frames -> save_credits, then
IMDBBatchValidatorWithCodeAssignment.process_credits_fast), importing the code
of the worktree itself, never the code of the main checkout. Parameters come
from the run's run_info.json.

After Stage II the frames it selected are written to run_info.json
(submitted_frames). If run_info.json sets stop_if_stage2_differs_from_reference
and they differ from reference_frames, the run stops before any model call.

Outputs, all inside the run folder:
  run.log                       full log, timestamped, never rotated
  db/tvcredits_v3.db            credits + raw_response_llm_call (one row per call)
  run_output/<export>.csv       credits export (name from run_info.json)
  run_output/calls.csv          per call: timestamp, frame, tokens, cost USD
  run_output/raw_responses/     per-frame raw LLM response JSON (before dedup)
  run_output/summary.json       frame-count check, totals, cost, stage times

Step 3 resumes from its checkpoint if interrupted: rerunning the command only
sends the frames not yet analysed.

Several runs can go in parallel, one process each: Stage III runs concurrently,
while Stage II (whole scenes buffered in RAM + PaddleOCR) and Stage IV (IMDb
parquet, several GB of RAM) share one machine-wide lock (runs/.stage2.lock),
so only one heavy stage runs at a time on the machine.
The OS releases a lock if a process dies, so it never goes stale.

Usage:
    python run_headless.py runs/NAIVE_2.4s --dry-run   # checks only, no calls
    python run_headless.py runs/NAIVE_2.4s             # real run (paid calls)
    python run_headless.py runs/PIPE_cap150 --until-stage2   # block B: Stage II only, free
"""

import argparse
import csv
import json
import logging
import msvcrt
import os
import shutil
import sqlite3
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

LOCK_DIR = Path(__file__).resolve().parent / "runs"
OCR_FAILURE = "OCR final error: PaddleOCR failed"
LOCK_POLL_SECONDS = 30
# Stage II and Stage IV share one lock: together they would not fit in RAM.
# The file keeps the "stage2" name so it also serialises Stage II processes started before this change.
HEAVY_LOCK = "stage2"


def load_run(run_dir: Path) -> dict:
    info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    params = info["parameters"]
    if params.get("include_previous_frame_image") is not False:
        sys.exit("run_info.json must set include_previous_frame_image to false")
    return info


def setup_logging(run_dir: Path, log_file: Path | None = None) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    log_file = log_file or run_dir / "run.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root = logging.getLogger()
    root.handlers[:] = [file_handler, console]
    root.setLevel(logging.INFO)


def video_path(episode_id: str) -> Path:
    from scripts_v3 import config

    (clip,) = [p for p in config.RAW_VIDEO_DIR.rglob("*") if p.is_file() and p.stem == episode_id]
    return clip


def ocr_reader_for(params: dict):
    from scripts_v3 import config, utils

    lang = params["ocr_language"]
    logging.info(f"[RUN] OCR: paddleocr lang={lang}")
    reader = utils.get_paddleocr_reader(lang=config.PADDLEOCR_LANG_MAP.get(lang, lang))
    # Building PaddleOCR raises the root logger to WARNING; restore INFO so the log stays complete.
    logging.getLogger().setLevel(logging.INFO)
    return reader


def done_episodes(run_dir: Path, stage: str) -> set[str]:
    """Episodes a stage has completed: legacy <stage>_done.json plus one marker file per episode."""
    out = run_dir / "run_output"
    legacy = out / f"{stage}_done.json"
    done = set(json.loads(legacy.read_text(encoding="utf-8"))) if legacy.exists() else set()
    markers = out / f"{stage}_done"
    if markers.is_dir():
        done |= {p.stem for p in markers.glob("*.done")}
    return done


def mark_done(run_dir: Path, stage: str, episode_id: str) -> None:
    # One file per episode, so parallel workers never rewrite a shared list.
    markers = run_dir / "run_output" / f"{stage}_done"
    markers.mkdir(parents=True, exist_ok=True)
    (markers / f"{episode_id}.done").write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")


def stage1_episode(episode_id: str, ocr_reader, user_stopwords: list[str]) -> int:
    """Mirror of the GUI STEP 1 button with the selection method "Run on the whole episode"."""
    from scripts_v3 import config, scene_detection, utils

    # An interrupted Stage I can leave a partial episode folder: always start clean.
    shutil.rmtree(config.EPISODES_BASE_DIR / episode_id, ignore_errors=True)
    t0 = time.perf_counter()
    scenes, status, err = scene_detection.identify_candidate_scenes(
        video_path(episode_id), episode_id, ocr_reader, "paddleocr", user_stopwords, whole_episode=True,
    )
    utils.record_phase_time(episode_id, "step1", time.perf_counter() - t0)
    if err:
        sys.exit(f"Stage I failed on {episode_id}: {status} {err}")
    logging.info(f"[RUN] Stage I done {episode_id}: {len(scenes)} candidate scenes ({status})")
    return len(scenes)


def stage1(run_dir: Path, episodes: list[str], params: dict) -> None:
    from scripts_v3 import utils

    if params.get("scene_selection") != "whole_episode":
        sys.exit("Stage I is only supported with scene_selection = whole_episode")
    done = done_episodes(run_dir, "stage1")
    ocr_reader = ocr_reader_for(params)
    user_stopwords = utils.load_user_stopwords()
    for i, episode_id in enumerate(episodes, 1):
        if episode_id in done:
            logging.info(f"[RUN] Stage I {i}/{len(episodes)}: {episode_id} already done")
            continue
        logging.info(f"[RUN] Stage I {i}/{len(episodes)}: {episode_id}")
        stage1_episode(episode_id, ocr_reader, user_stopwords)
        mark_done(run_dir, "stage1", episode_id)


def stage2_peak_bytes(episode_id: str) -> int:
    """Stage II holds every frame of a scene in RAM: bytes of the longest candidate scene."""
    import cv2

    from scripts_v3 import config

    scenes = json.loads(
        (config.EPISODES_BASE_DIR / episode_id / "analysis" / "initial_scene_analysis.json").read_text(encoding="utf-8")
    ).get("candidate_scenes", [])
    capture = cv2.VideoCapture(str(video_path(episode_id)))
    width, height = capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    capture.release()
    longest = max((sc["original_end_frame"] - sc["original_start_frame"] for sc in scenes), default=0)
    return int(longest * width * height * 3)


def run_parallel(run_dir: Path, stage: str, episodes: list[str], workers: int) -> list[str]:
    """Run one stage on independent episodes in up to `workers` child processes.

    Each child is this script in --worker mode on one episode, logging to
    run_output/logs/<stage>_<episode>.log, so a crash (e.g. out of memory)
    only loses that episode and its memory is freed when the child exits.
    Stage II episodes go from the lightest to the heaviest scene in RAM.
    The parent holds the machine lock. Returns the episodes that failed.
    """
    import subprocess

    pending = [ep for ep in episodes if ep not in done_episodes(run_dir, stage)]
    if stage == "stage2":
        peaks = {ep: stage2_peak_bytes(ep) for ep in pending}
        pending.sort(key=peaks.get)
        logging.info("[RUN] stage2 order (longest candidate scene in RAM): "
                     + ", ".join(f"{ep} {peaks[ep] / 1e9:.1f} GB" for ep in pending))
    logging.info(f"[RUN] {stage}: {len(pending)} episodes to do with {workers} parallel workers")
    running = {}
    failed = []
    while pending or running:
        while pending and len(running) < workers:
            ep = pending.pop(0)
            running[ep] = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), str(run_dir),
                 "--worker-stage", stage, "--worker-episode", ep],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            logging.info(f"[RUN] {stage} started {ep} (PID {running[ep].pid})")
        for ep, proc in list(running.items()):
            if proc.poll() is not None:
                del running[ep]
                # The done marker is written only after the work is saved; PaddleOCR can still
                # crash the interpreter at exit (0xC0000409), so the exit code alone is not the verdict.
                ok = ep in done_episodes(run_dir, stage)
                if ok and proc.returncode != 0:
                    logging.warning(f"[RUN] {stage} {ep}: work saved, process crashed at exit (code {proc.returncode})")
                logging.info(f"[RUN] {stage} {'done' if ok else 'FAILED'} {ep} (exit {proc.returncode})")
                if not ok:
                    failed.append(ep)
        time.sleep(5)
    if failed:
        logging.error(f"[RUN] {stage} failed on {failed}: see run_output/logs/{stage}_<episode>.log")
    return failed


def stage2_episode(episode_id: str, ocr_reader, user_stopwords: list[str]) -> int:
    """Mirror of the GUI STEP 2 button with every candidate scene selected."""
    from scenedetect import open_video

    from scripts_v3 import config, frame_analysis

    analysis_dir = config.EPISODES_BASE_DIR / episode_id / "analysis"
    # A Stage II interrupted mid-episode leaves partial output: always start clean.
    for sub in ("frames", "skipped_frames"):
        shutil.rmtree(analysis_dir / sub, ignore_errors=True)
    manifest_path = analysis_dir / "analysis_manifest.json"
    manifest_path.unlink(missing_ok=True)

    step1 = json.loads((analysis_dir / "initial_scene_analysis.json").read_text(encoding="utf-8"))
    scenes = step1.get("candidate_scenes", [])
    video_file = video_path(episode_id)
    video_stream = open_video(str(video_file))
    fps = video_stream.frame_rate
    frame_width, frame_height = video_stream.frame_size
    texts_cache, files_cache = [], []
    last_text = last_hash = last_bbox = None
    manifest = {"scenes": {}}
    for n, scene in enumerate(scenes):
        if not all(k in scene for k in ("original_start_frame", "original_end_frame")):
            logging.warning(f"[RUN] {episode_id}: skipping scene with missing keys: {scene}")
            continue
        compatible = {**scene, "start_frame": scene["original_start_frame"], "end_frame": scene["original_end_frame"]}
        result, last_text, last_hash, last_bbox = frame_analysis.analyze_candidate_scene_frames(
            video_path=video_file,
            episode_id=episode_id,
            scene_info=compatible,
            fps=fps,
            frame_height=frame_height,
            frame_width=frame_width,
            ocr_reader=ocr_reader,
            ocr_engine_type="paddleocr",
            user_stopwords=user_stopwords,
            global_last_saved_ocr_text_input=last_text,
            global_last_saved_frame_hash_input=last_hash,
            global_last_saved_ocr_bbox_input=last_bbox,
            episode_saved_texts_cache=texts_cache,
            episode_saved_files_cache=files_cache,
        )
        index = scene.get("scene_index")
        manifest["scenes"][f"scene_{index}" if index is not None else f"scene_{episode_id}_{n}"] = result
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return len(scenes)


def stage2(run_dir: Path, episodes: list[str], params: dict) -> None:
    from scripts_v3 import utils

    done = done_episodes(run_dir, "stage2")
    ocr_reader = ocr_reader_for(params)
    user_stopwords = utils.load_user_stopwords()
    for i, episode_id in enumerate(episodes, 1):
        if episode_id in done:
            logging.info(f"[RUN] Stage II {i}/{len(episodes)}: {episode_id} already done")
            continue
        logging.info(f"[RUN] Stage II {i}/{len(episodes)}: {episode_id}")
        t0 = time.perf_counter()
        n_scenes = stage2_episode(episode_id, ocr_reader, user_stopwords)
        utils.record_phase_time(episode_id, "step2", time.perf_counter() - t0)
        mark_done(run_dir, "stage2", episode_id)
        logging.info(f"[RUN] Stage II done {episode_id}: {n_scenes} scenes, {len(pipeline_frames(episode_id))} frames")


def pipeline_frames(episode_id: str) -> list[str]:
    """Frames Stage III will submit: manifest output_files whose image exists."""
    from scripts_v3 import config

    analysis_dir = config.EPISODES_BASE_DIR / episode_id / "analysis"
    manifest = json.loads((analysis_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
    names = {
        Path(f["path"]).name
        for scene in manifest.get("scenes", {}).values()
        for f in scene.get("output_files", [])
        if f.get("path") and (analysis_dir / "frames" / Path(f["path"]).name).is_file()
    }
    return sorted(names)


def write_stage2_timecodes(run_dir: Path, episodes: list[str]) -> None:
    """run_output/stage2_frames_timecodes.csv: every selected frame with its position in the video."""
    import cv2

    from scripts_v3 import config

    out = run_dir / "run_output" / "stage2_frames_timecodes.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode_id", "scene", "frame_file", "frame_num", "fps", "seconds", "timecode"])
        for ep in episodes:
            capture = cv2.VideoCapture(str(video_path(ep)))
            fps = capture.get(cv2.CAP_PROP_FPS)
            capture.release()
            analysis_dir = config.EPISODES_BASE_DIR / ep / "analysis"
            manifest = json.loads((analysis_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
            for scene_key, scene in manifest.get("scenes", {}).items():
                for frame in scene.get("output_files", []):
                    name = Path(frame.get("path") or "").name
                    if not name or not (analysis_dir / "frames" / name).is_file():
                        continue
                    seconds = frame["frame_num"] / fps
                    h, rem = divmod(seconds, 3600)
                    m, sec = divmod(rem, 60)
                    writer.writerow([ep, scene_key, name, frame["frame_num"], round(fps, 3), round(seconds, 3),
                                     f"{int(h):02d}:{int(m):02d}:{sec:06.3f}"])


def record_stage2_frames(run_dir: Path, info: dict, episodes: list[str]) -> None:
    """Write the Stage II selection into run_info.json; stop if it must match July and does not."""
    write_stage2_timecodes(run_dir, episodes)
    submitted = {ep: pipeline_frames(ep) for ep in episodes}
    info["submitted_frames"] = submitted
    info["submitted_frames_total"] = sum(map(len, submitted.values()))
    info["submitted_frames_per_folder"] = {ep: len(v) for ep, v in submitted.items()}
    reference = info.get("reference_frames") or {}
    diff = {
        ep: {"only_new": sorted(set(submitted[ep]) - set(reference[ep])),
             "only_reference": sorted(set(reference[ep]) - set(submitted[ep]))}
        for ep in episodes
        if ep in reference and set(submitted[ep]) != set(reference[ep])
    }
    info["stage2_vs_reference"] = {"identical": not diff, "differences": diff} if reference else None
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info(
        f"[RUN] Stage II selected {info['submitted_frames_total']} frames "
        f"(July reference {info.get('reference_frames_total')}, identical={not diff})"
    )
    if info.get("stop_if_stage2_differs_from_reference") and diff:
        sys.exit("Stage II frames differ from the July reference: stopped before any model call (see run_info.json)")


def redo_episodes(run_dir: Path, info: dict, redo: list[str], reason: str | None) -> None:
    """Clear Stage I/II completion of some episodes so the next stages redo them, keeping a trace."""
    from scripts_v3 import config

    unknown = [ep for ep in redo if ep not in info["submitted_frames"]]
    if unknown:
        sys.exit(f"Unknown episodes: {unknown}")
    conn = sqlite3.connect(config.DB_PATH)
    q = ",".join("?" * len(redo))
    old_times = conn.execute(
        f'SELECT episode_id, step, seconds, recorded_at FROM "{config.DB_TABLE_TIMING}" '
        f"WHERE step IN ('step1', 'step2') AND episode_id IN ({q})", redo,
    ).fetchall()
    conn.execute(f"DELETE FROM \"{config.DB_TABLE_TIMING}\" WHERE step IN ('step1', 'step2') AND episode_id IN ({q})", redo)
    conn.commit()
    conn.close()
    for stage in ("stage1", "stage2"):
        for ep in redo:
            (run_dir / "run_output" / f"{stage}_done" / f"{ep}.done").unlink(missing_ok=True)
            for suffix in (".log",):
                log = run_dir / "run_output" / "logs" / f"{stage}_{ep}{suffix}"
                if log.exists():
                    log.rename(log.with_name(f"{log.stem}.failed_{datetime.now():%Y%m%d_%H%M%S}.log"))
    info.setdefault("redone_episodes", []).append({
        "episodes": redo,
        "redone_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "discarded_timings": [dict(zip(("episode_id", "step", "seconds", "recorded_at"), row)) for row in old_times],
        "discarded_logs": "run_output/logs/<stage>_<episode>.failed_<timestamp>.log",
    })
    info["submitted_frames_total"] = None  # Stage II selection is recomputed for the whole run
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info(f"[RUN] redo of Stage I/II for {redo}: {reason}")


def stage3(episodes: list[str], params: dict) -> None:
    from scripts_v3 import constants, utils, vlm_processing

    provider = vlm_processing.resolve_vlm_provider(params["vlm_provider"])
    naive = params["naive_mode"]
    for i, episode_id in enumerate(episodes, 1):
        logging.info(f"[RUN] Stage III {i}/{len(episodes)}: {episode_id}")
        t0 = time.perf_counter()
        count, status, err = vlm_processing.run_azure_vlm_ocr_on_frames(
            episode_id,
            constants.DEFAULT_VLM_MAX_NEW_TOKENS,
            params["vlm_provider"],
            False,  # enable_role_correction: disabled in the GUI
            False,  # include_previous_frame (image)
            naive_mode=naive,
        )
        if err or status not in ("completed", "completed_no_new_frames"):
            sys.exit(f"Stage III failed on {episode_id}: {status} {err}")
        ocr_dir = vlm_processing.get_vlm_ocr_dir(episode_id, provider, naive_mode=naive)
        credits = utils.load_vlm_results_from_jsonl(ocr_dir / f"{episode_id}_credits_azure_vlm.json")
        if credits:
            utils.save_credits(episode_id, credits)
        utils.record_phase_time(episode_id, "step3", time.perf_counter() - t0, provider=provider)
        logging.info(f"[RUN] Stage III done {episode_id}: {len(credits or [])} credits")


@contextmanager
def machine_lock(stage: str):
    """Hold the machine-wide lock for a stage (Windows byte-range lock on byte 0)."""
    with open(LOCK_DIR / f".{stage}.lock", "a+") as f:
        f.seek(0)
        waited = False
        while True:
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if not waited:
                    logging.info(f"[RUN] {stage} lock held by another run, waiting...")
                    waited = True
                time.sleep(LOCK_POLL_SECONDS)
        logging.info(f"[RUN] {stage} lock acquired")
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            logging.info(f"[RUN] {stage} lock released")


def stage4(episodes: list[str], params: dict) -> None:
    from scripts_v3 import utils
    from scripts_v3.imdb_batch_validation import IMDBBatchValidatorWithCodeAssignment

    validator = IMDBBatchValidatorWithCodeAssignment(
        fuzzy_enabled=params["fuzzy_matching_enabled"], fuzzy_threshold=params["fuzzy_threshold"]
    )
    for i, episode_id in enumerate(episodes, 1):
        logging.info(f"[RUN] Stage IV {i}/{len(episodes)}: {episode_id}")
        t0 = time.perf_counter()
        credits = validator.get_unprocessed_credits(episode_id=episode_id)
        if credits:
            validator.process_credits_fast(credits)
        utils.record_phase_time(episode_id, "step4", time.perf_counter() - t0)
    validator.save_fuzzy_corrections_to_csv()


CALL_FIELDS = ["input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens", "reasoning_tokens", "cost_usd"]


def call_metrics(response: dict, pricing: dict) -> dict:
    """Token counts and cost of one call. Cache writes are not priced: the
    list prices we use have no separate rate for them."""
    usage = response.get("usage") or {}
    input_details = usage.get("input_tokens_details") or {}
    m = {
        "input_tokens": usage.get("input_tokens", 0),
        "cache_read_tokens": input_details.get("cached_tokens", 0),
        "cache_write_tokens": input_details.get("cache_write_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "reasoning_tokens": (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0),
    }
    m["cost_usd"] = (
        (m["input_tokens"] - m["cache_read_tokens"]) * pricing["input"]
        + m["cache_read_tokens"] * pricing["cached_input"]
        + m["output_tokens"] * pricing["output"]
    ) / 1_000_000
    return m


def finalize(run_dir: Path, info: dict, stage_seconds: dict) -> dict:
    from export_db_to_csv import CREDITS_COLUMNS, export_table_to_csv
    from scripts_v3 import config

    out = run_dir / "run_output"
    raw_dir = out / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_name = info["expected_export_csv"]
    export_table_to_csv(
        config.DB_PATH, config.DB_TABLE_CREDITS, out,
        columns=CREDITS_COLUMNS, output_basename=Path(csv_name).stem,
    )

    pricing = info["pricing_usd_per_1M_tokens"]
    conn = sqlite3.connect(config.DB_PATH)
    rows = conn.execute(
        f'SELECT episode_id, source_frame, recorded_at, raw_response FROM "{config.DB_TABLE_RAW_RESPONSE}" ORDER BY id'
    ).fetchall()
    conn.close()

    totals = {"calls": 0, **{k: 0 for k in CALL_FIELDS}}
    retention_counts: dict[str, int] = {}
    called: dict[str, list[str]] = {}
    with open(out / "calls.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["recorded_at_utc", "episode_id", "source_frame", "prompt_cache_retention", *CALL_FIELDS])
        for episode_id, frame, recorded_at, raw in rows:
            response = json.loads(raw)
            m = call_metrics(response, pricing)
            retention = str(response.get("prompt_cache_retention"))
            writer.writerow([recorded_at, episode_id, frame, retention, *(m[k] for k in CALL_FIELDS[:-1]), f"{m['cost_usd']:.6f}"])
            (raw_dir / episode_id).mkdir(exist_ok=True)
            (raw_dir / episode_id / f"{Path(frame).stem}.json").write_text(
                json.dumps(response, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            called.setdefault(episode_id, []).append(frame)
            retention_counts[retention] = retention_counts.get(retention, 0) + 1
            totals["calls"] += 1
            for k in CALL_FIELDS:
                totals[k] += m[k]

    expected = info["submitted_frames"]
    mismatches = {
        ep: {"missing": sorted(set(frames) - set(called.get(ep, []))),
             "extra_or_duplicate": len(called.get(ep, [])) - len(set(called.get(ep, [])) & set(frames))}
        for ep, frames in expected.items()
        if sorted(called.get(ep, [])) != sorted(frames)
    }
    summary = {
        "run_id": info["run_id"],
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "frames_expected": info["submitted_frames_total"],
        "frames_called": totals["calls"],
        "frame_check": "OK" if not mismatches and totals["calls"] == info["submitted_frames_total"] else "MISMATCH",
        "mismatches": mismatches,
        "totals": {**totals, "cost_usd": round(totals["cost_usd"], 4)},
        "stage_seconds": stage_seconds,
        "export_csv": f"run_output/{csv_name}",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Prompt caching is left at the Azure default (no cache parameter is sent,
    # so the code stays at the pinned commit); record what Azure actually did.
    info["prompt_cache"] = {
        "requested_by_code": None,
        "prompt_cache_retention_effective": retention_counts,
        "cache_read_tokens": totals["cache_read_tokens"],
        "cache_write_tokens": totals["cache_write_tokens"],
        "input_tokens": totals["input_tokens"],
        "cache_read_fraction": round(totals["cache_read_tokens"] / totals["input_tokens"], 4) if totals["input_tokens"] else None,
    }
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="check the run without calling the model")
    parser.add_argument("--finalize-only", action="store_true", help="redo exports/summary from the run DB, no calls")
    parser.add_argument("--until-stage2", action="store_true", help="run Stage II only (free), stop before any model call")
    parser.add_argument("--until-stage1", action="store_true", help="run Stage I only (free), stop before Stage II")
    parser.add_argument("--stage1-workers", type=int, help="child processes for Stage I, one episode each (default: in-process)")
    parser.add_argument("--stage2-workers", type=int, help="child processes for Stage II, one episode each, lightest first; "
                                                          "an episode that fails is recorded and skipped (default: in-process)")
    parser.add_argument("--redo-episodes", nargs="+", metavar="EPISODE",
                        help="redo Stage I and II of these episodes (done markers cleared, old timings kept in run_info.json)")
    parser.add_argument("--only-episodes", nargs="+", metavar="EPISODE",
                        help="run Stages I-IV on these episodes only; the others keep their results")
    parser.add_argument("--redo-reason", help="why the episodes are redone (written to run_info.json)")
    parser.add_argument("--worker-stage", choices=["stage1", "stage2"], help=argparse.SUPPRESS)
    parser.add_argument("--worker-episode", help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Pipeline code prints emoji; a redirected stdout on Windows is cp1252.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run_dir = args.run_dir.resolve()
    info = load_run(run_dir)
    params = info["parameters"]
    episodes = list(info["submitted_frames"])
    all_episodes = episodes
    if args.only_episodes:
        unknown = [ep for ep in args.only_episodes if ep not in episodes]
        if unknown:
            sys.exit(f"Unknown episodes: {unknown}")
        episodes = args.only_episodes

    # Import the worktree's own code, with the worktree as project root.
    os.chdir(run_dir)
    sys.path.insert(0, str(run_dir))
    if args.worker_stage:
        setup_logging(run_dir, run_dir / "run_output" / "logs" / f"{args.worker_stage}_{args.worker_episode}.log")
    else:
        setup_logging(run_dir)
    from scripts_v3 import config, utils, vlm_processing

    if args.worker_stage:
        # Child of run_parallel: one stage on one episode; the parent holds the machine lock.
        if Path(config.PROJECT_ROOT).resolve() != run_dir:
            sys.exit(f"Wrong code imported: PROJECT_ROOT={config.PROJECT_ROOT}")
        utils.init_db()
        reader = ocr_reader_for(params)
        stopwords = utils.load_user_stopwords()
        if args.worker_stage == "stage1":
            stage1_episode(args.worker_episode, reader, stopwords)
        else:
            t0 = time.perf_counter()
            n_scenes = stage2_episode(args.worker_episode, reader, stopwords)
            utils.record_phase_time(args.worker_episode, "step2", time.perf_counter() - t0)
            logging.info(f"[RUN] Stage II done {args.worker_episode}: {n_scenes} scenes")
        # The pipeline logs a failed OCR call and carries on as if the frame had no text
        # (e.g. CUDA out of memory): such an episode is not complete, whatever the exit code.
        worker_log = run_dir / "run_output" / "logs" / f"{args.worker_stage}_{args.worker_episode}.log"
        ocr_failures = sum(1 for line in open(worker_log, encoding="utf-8") if OCR_FAILURE in line)
        if ocr_failures:
            sys.exit(f"{args.worker_episode}: {ocr_failures} OCR calls failed (see {worker_log.name}); not marked done")
        mark_done(run_dir, args.worker_stage, args.worker_episode)
        return

    if Path(config.PROJECT_ROOT).resolve() != run_dir:
        sys.exit(f"Wrong code imported: PROJECT_ROOT={config.PROJECT_ROOT}")
    provider = vlm_processing.resolve_vlm_provider(params["vlm_provider"])
    if args.redo_episodes and not args.dry_run:
        redo_episodes(run_dir, info, args.redo_episodes, args.redo_reason)
    stage2_pending = "stage2" in params.get("stages", []) and info.get("submitted_frames_total") is None
    stage1_pending = "stage1" in params.get("stages", []) and stage2_pending
    if stage2_pending:
        missing = [] if stage1_pending else [
            ep for ep in episodes if not (config.EPISODES_BASE_DIR / ep / "analysis" / "initial_scene_analysis.json").is_file()
        ]
        clips = [video_path(ep).name for ep in episodes]
        logging.info(
            f"[RUN] {info['run_id']} commit={info['git_commit'][:7]} provider={provider} "
            f"naive={params['naive_mode']} cap={params.get('SCROLL_MAX_FRAMES_PER_SAVE')} fuzzy={params['fuzzy_threshold']} "
            f"episodes={len(episodes)} clips={len(clips)} stage1={'to run' if stage1_pending else 'given'} "
            f"stage1_missing={missing} db={config.DB_PATH}"
        )
        if missing:
            sys.exit("Stage I outputs missing")
    else:
        frames = {ep: sorted(p.name for p in config.get_frames_dir(ep, naive_mode=params["naive_mode"]).glob("*.jpg")) for ep in episodes}
        frames_ok = frames == {ep: sorted(info["submitted_frames"][ep]) for ep in episodes}
        logging.info(
            f"[RUN] {info['run_id']} commit={info['git_commit'][:7]} provider={provider} "
            f"naive={params['naive_mode']} fuzzy={params['fuzzy_threshold']} episodes={len(episodes)} "
            f"frames={sum(map(len, frames.values()))} frames_match_run_info={frames_ok} db={config.DB_PATH}"
        )
        if not frames_ok:
            sys.exit("Frames on disk differ from run_info.json")
    if args.dry_run:
        return
    if args.finalize_only:
        summary_path = run_dir / "run_output" / "summary.json"
        previous = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        summary = finalize(run_dir, info, previous.get("stage_seconds", {}))
        logging.info(f"[RUN] finalize-only {summary['frame_check']}: {summary['frames_called']}/{summary['frames_expected']} frames")
        return

    utils.init_db()
    stage_seconds = {}
    if stage1_pending:
        with machine_lock(HEAVY_LOCK):
            t0 = time.perf_counter()
            if args.stage1_workers is not None:
                if run_parallel(run_dir, "stage1", episodes, args.stage1_workers):
                    sys.exit("Stage I failed on some episodes: rerun to retry them")
            else:
                stage1(run_dir, episodes, params)
            stage_seconds["stage1"] = round(time.perf_counter() - t0, 1)
        if args.until_stage1:
            info["stage_seconds_stage1"] = stage_seconds["stage1"]
            (run_dir / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
            logging.info("[RUN] --until-stage1: stopping before Stage II (no model call made)")
            return
    if stage2_pending:
        with machine_lock(HEAVY_LOCK):
            t0 = time.perf_counter()
            if args.stage2_workers is not None:
                stage2_failed = run_parallel(run_dir, "stage2", episodes, args.stage2_workers)
                if stage2_failed:
                    # Keep going with the episodes Stage II could process; record the others.
                    info["stage2_failed"] = {
                        ep: f"Stage II did not complete (see run_output/logs/stage2_{ep}.log); "
                            f"longest candidate scene needs {stage2_peak_bytes(ep) / 1e9:.1f} GB of RAM"
                        for ep in stage2_failed
                    }
                    episodes = [ep for ep in episodes if ep not in stage2_failed]
            else:
                stage2(run_dir, episodes, params)
            stage_seconds["stage2"] = round(time.perf_counter() - t0, 1)
        record_stage2_frames(run_dir, info, [ep for ep in all_episodes if ep not in info.get("stage2_failed", {})])
        info["stage_seconds_before_stage3"] = stage_seconds
        (run_dir / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.until_stage2:
        logging.info("[RUN] --until-stage2: stopping before Stage III (no model call made)")
        return
    t0 = time.perf_counter()
    stage3(episodes, params)
    stage_seconds["stage3"] = round(time.perf_counter() - t0, 1)
    with machine_lock(HEAVY_LOCK):
        t0 = time.perf_counter()
        stage4(episodes, params)
        stage_seconds["stage4"] = round(time.perf_counter() - t0, 1)

    if args.only_episodes:
        # Keep the stage times of the full run; record those of the partial redo next to them.
        summary_path = run_dir / "run_output" / "summary.json"
        previous = json.loads(summary_path.read_text(encoding="utf-8")).get("stage_seconds", {}) if summary_path.exists() else {}
        stage_seconds = {**previous, f"redo_{'+'.join(episodes)}": stage_seconds}
    summary = finalize(run_dir, info, stage_seconds)
    logging.info(f"[RUN] {summary['frame_check']}: {summary['frames_called']}/{summary['frames_expected']} frames, cost {summary['totals']['cost_usd']} USD")


if __name__ == "__main__":
    main()
