"""
Naive frame extraction: Steps 1+2 collapsed into a single fixed-interval pass.

No scene detection, no OCR, no similarity/dedup filtering - one frame every
`config.NAIVE_FRAME_INTERVAL_SECONDS` seconds, plus the very first and the very
last frame of the file. Output goes to
data/episodes/<episode>/naive_analysis/frames so it never mixes with the
regular analysis/frames output of the normal pipeline.

Frames are read with grab()/retrieve(): grab() advances the decoder without
paying for a full decode, and only the frames we actually keep get retrieved.
That is both faster than decoding everything and more accurate than seeking by
frame index, which many codecs answer only approximately.
"""

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2

from . import config

# Filename prefix owned by this module. Stale frames from a previous naive run
# are cleared by matching this exact prefix, so nothing outside what this
# function itself wrote is ever removed.
FRAME_PREFIX = "naive_"
FRAME_GLOB = f"{FRAME_PREFIX}*.jpg"

# Fallback when the container reports a nonsensical frame rate.
FALLBACK_FPS = 25.0


def _target_frame_indices(total_frames: int, fps: float, interval_seconds: float) -> List[int]:
    """Frame indices to keep: one every `interval_seconds`, plus first and last."""
    step = max(1, int(round(fps * interval_seconds)))
    targets = set(range(0, total_frames, step))
    targets.add(0)
    targets.add(total_frames - 1)
    return sorted(t for t in targets if 0 <= t < total_frames)


def _save_frame(img, frames_dir: Path, seq: int, frame_num: int) -> bool:
    """Write one frame. `seq` keeps the lexicographic order of the filenames
    chronological, which is what Step 3 relies on (it just sorts the *.jpg)."""
    out_path = frames_dir / f"{FRAME_PREFIX}{seq:05d}_num{frame_num:06d}.jpg"
    try:
        return bool(cv2.imwrite(str(out_path), img))
    except Exception as e:
        logging.error(f"Failed to write naive frame {frame_num} to {out_path}: {e}")
        return False


def _clear_previous_frames(frames_dir: Path, episode_id: str) -> None:
    """Remove frames written by a previous naive run for this episode.

    Only files matching this module's own `naive_*.jpg` prefix inside the
    dedicated naive_analysis/frames folder are touched - a re-run with a
    different interval would otherwise leave stale frames behind and feed Step 3
    a mix of two extractions.
    """
    stale = list(frames_dir.glob(FRAME_GLOB))
    if not stale:
        return
    logging.info(f"[{episode_id}] Removing {len(stale)} frame(s) from a previous naive extraction in {frames_dir}")
    for path in stale:
        try:
            path.unlink()
        except OSError as e:
            logging.warning(f"[{episode_id}] Could not remove stale naive frame {path.name}: {e}")


def extract_frames_naive(
    video_path: Path,
    episode_id: str,
    interval_seconds: Optional[float] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[int, str, Optional[str]]:
    """Extract one frame every `interval_seconds`, plus the first and last frame.

    Args:
        video_path: Source video file.
        episode_id: Episode identifier (the video filename stem).
        interval_seconds: Seconds between kept frames. Defaults to
            config.NAIVE_FRAME_INTERVAL_SECONDS.
        progress_callback: Optional callable(frames_read, total_frames) for UI
            progress; called periodically, not on every frame.

    Returns:
        (saved_count, status, error_message) - status is 'completed',
        'completed_no_frames', or an 'error_*' string.
    """
    if interval_seconds is None:
        interval_seconds = config.NAIVE_FRAME_INTERVAL_SECONDS

    frames_dir = config.get_frames_dir(episode_id, naive_mode=True)
    try:
        frames_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        msg = f"Failed to create naive frames directory {frames_dir}: {e}"
        logging.error(f"[{episode_id}] {msg}", exc_info=True)
        return 0, "error_creating_output_dir", msg

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        msg = f"Could not open video: {video_path}"
        logging.error(f"[{episode_id}] {msg}")
        return 0, "error_opening_video", msg

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not fps or fps <= 0:
            logging.warning(f"[{episode_id}] Video reports invalid FPS ({fps}); falling back to {FALLBACK_FPS}")
            fps = FALLBACK_FPS
        if total_frames <= 0:
            msg = f"Video reports no frames (CAP_PROP_FRAME_COUNT={total_frames})"
            logging.error(f"[{episode_id}] {msg}")
            return 0, "error_no_frames", msg

        _clear_previous_frames(frames_dir, episode_id)

        targets = _target_frame_indices(total_frames, fps, interval_seconds)
        remaining = set(targets)
        saved_indices = set()
        logging.info(
            f"[{episode_id}] Naive extraction: {total_frames} frames @ {fps:.3f} fps, "
            f"one every {interval_seconds}s -> {len(targets)} target frame(s) into {frames_dir}"
        )

        last_grabbed = -1
        idx = 0
        while True:
            if not cap.grab():
                break
            last_grabbed = idx

            if idx in remaining:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    if _save_frame(frame, frames_dir, len(saved_indices), idx):
                        saved_indices.add(idx)
                else:
                    logging.warning(f"[{episode_id}] Could not decode target frame {idx}; skipping it")
                remaining.discard(idx)

            idx += 1
            if progress_callback and idx % 500 == 0:
                progress_callback(idx, total_frames)

        # CAP_PROP_FRAME_COUNT is frequently an estimate, so the decoder can stop
        # before total_frames-1 and the "last frame" target would never be hit.
        # Go back for the last frame we actually decoded so the caller's
        # "include the last frame of the file" guarantee always holds.
        if last_grabbed >= 0 and last_grabbed not in saved_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, last_grabbed)
            ok, frame = cap.read()
            if ok and frame is not None:
                if _save_frame(frame, frames_dir, len(saved_indices), last_grabbed):
                    saved_indices.add(last_grabbed)
                    logging.info(
                        f"[{episode_id}] Saved real last frame {last_grabbed} "
                        f"(file ended before the reported frame {total_frames - 1})"
                    )
            else:
                logging.warning(f"[{episode_id}] Could not re-read last frame {last_grabbed}")

        if progress_callback:
            progress_callback(idx, total_frames)
    finally:
        cap.release()

    saved = len(saved_indices)
    if saved == 0:
        msg = f"No frames could be extracted from {video_path}"
        logging.warning(f"[{episode_id}] {msg}")
        return 0, "completed_no_frames", msg

    logging.info(f"[{episode_id}] Naive extraction completed: {saved} frame(s) saved to {frames_dir}")
    return saved, "completed", None
