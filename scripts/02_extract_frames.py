#!/usr/bin/env python3
"""
02_extract_frames.py: Extract frames containing text from video segments

Usage:
    python 02_extract_frames.py --episode-id tt1234567 \
                               --fps 2 \
                               --text-threshold 0.05
"""
import argparse
import os
import json
import logging
import sys

# Ensure OpenCV is available
try:
    import cv2
except ImportError:
    print(
        "Error: 'cv2' (OpenCV) is not installed.\n"
        "Please install it by running: pip install opencv-python\n"
        "or add 'opencv-python' to requirements.txt and run pip install -r requirements.txt"
    )
    sys.exit(1)

# Ensure numpy is available
try:
    import numpy as np
except ImportError:
    print(
        "Error: 'numpy' is not installed.\n"
        "Please install it by running: pip install numpy\n"
        "or add 'numpy' to requirements.txt and run pip install -r requirements.txt"
    )
    sys.exit(1)

# Ensure tqdm is available
try:
    from tqdm import tqdm
except ImportError:
    print(
        "Error: 'tqdm' is not installed.\n"
        "Please install it by running: pip install tqdm\n"
        "or add 'tqdm' to requirements.txt and run pip install -r requirements.txt"
    )
    sys.exit(1)


def is_text(frame, thresh_ratio):
    """
    Determine if a frame contains text:
    - Convert to grayscale
    - Otsu binarize
    - Compute ratio of white pixels
    - Return True if ratio > thresh_ratio
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white = np.count_nonzero(binary == 255)
    total = binary.size
    return (white / total) > thresh_ratio


def process_segment(video_path, episode_id, fps, threshold, manifest, stats):
    """Process a single video segment, extract text frames."""
    segment_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join('data', 'frames', episode_id, segment_name)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video: {video_path}")
        return

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_step = orig_fps / fps
    next_frame = 0.0
    frame_idx = 0
    saved = []

    try:
        with tqdm(desc=f"Segment {segment_name}", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx >= next_frame:
                    # Test text
                    if is_text(frame, threshold):
                        fname = f"{frame_idx:05d}.jpg"
                        path = os.path.join(out_dir, fname)
                        cv2.imwrite(path, frame)
                        saved.append(fname)
                    next_frame += frame_step
                frame_idx += 1
                pbar.update(1)
    except KeyboardInterrupt:
        logging.info("Interrupted by user, saving progress...")
    finally:
        cap.release()

    # Update manifest and stats
    manifest[segment_name] = saved
    stats[segment_name] = {
        "processed": frame_idx,
        "saved": len(saved)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract text frames from video segments"
    )
    parser.add_argument('--episode-id', required=True,
                        help='Episode identifier, e.g., tt1234567')
    parser.add_argument('--fps', type=float, default=2.0,
                        help='Frames per second to sample')
    parser.add_argument('--text-threshold', type=float, default=0.05,
                        help='White pixel ratio threshold for text detection')
    args = parser.parse_args()

    episode_id = args.episode_id
    fps = args.fps
    threshold = args.text_threshold

    # Prepare manifest file
    frames_dir = os.path.join('data', 'frames', episode_id)
    os.makedirs(frames_dir, exist_ok=True)
    manifest_path = os.path.join(frames_dir, 'frames.json')
    if os.path.isfile(manifest_path):
        with open(manifest_path, 'r') as mf:
            manifest = json.load(mf)
    else:
        manifest = {}

    stats = {}

    # Find segment files
    clips_dir = os.path.join('data', 'clips')
    pattern = f"{episode_id}_"
    files = [os.path.join(clips_dir, f) for f in os.listdir(clips_dir)
             if f.startswith(pattern) and f.endswith('.mp4')]

    if not files:
        logging.error(f"No clips found for episode {episode_id} in {clips_dir}")
        sys.exit(1)

    for video in files:
        process_segment(video, episode_id, fps, threshold, manifest, stats)

    # Write manifest
    with open(manifest_path, 'w') as mf:
        json.dump(manifest, mf, indent=2)
    logging.info(f"Frames manifest written to {manifest_path}")

    # Write stats summary
    stats_path = os.path.join(frames_dir, 'stats.json')
    with open(stats_path, 'w') as sf:
        json.dump(stats, sf, indent=2)
    logging.info(f"Stats written to {stats_path}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
