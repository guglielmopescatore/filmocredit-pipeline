#!/usr/bin/env python3
"""
01_segment_video.py: Segment a video into clips losslessly using ffmpeg

Usage:
    python 01_segment_video.py --input data/raw/VID.mp4 \
                               --segments "00:00:00-00:01:30,00:42:10-00:44:00" \
                               --episode-id tt1234567 [--dry-run]
"""
import argparse
import subprocess
import os
import json
import logging
import sys

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


def parse_time(ts):
    """Convert hh:mm:ss to total seconds."""
    h, m, s = ts.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


def main():
    parser = argparse.ArgumentParser(
        description="Segment video losslessly into clips using ffmpeg"
    )
    parser.add_argument('--input', required=True,
                        help='Path to input video file')
    parser.add_argument('--segments', required=True,
                        help='Comma-separated list of start-end pairs (hh:mm:ss-hh:mm:ss)')
    parser.add_argument('--episode-id', required=True,
                        help='Episode identifier, e.g., tt1234567')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing')
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        logging.error(f"Input file not found: {input_path}")
        return

    # Prepare output directories
    clips_dir = os.path.join('data', 'clips')
    os.makedirs(clips_dir, exist_ok=True)
    ocr_dir = os.path.join('data', 'ocr', args.episode_id)
    os.makedirs(ocr_dir, exist_ok=True)

    # Parse segments
    raw_segs = args.segments.split(',')
    segments = []
    for seg in raw_segs:
        try:
            start, end = seg.split('-', 1)
            segments.append((start, end))
        except ValueError:
            logging.error(f"Invalid segment format: {seg}")
            return

    manifest_path = os.path.join(ocr_dir, 'segments.json')
    # Load or initialize manifest
    if os.path.isfile(manifest_path):
        with open(manifest_path, 'r') as mf:
            manifest = json.load(mf)
    else:
        manifest = []

    total_bytes = 0

    # Process segments
    for idx, (start, end) in enumerate(tqdm(segments, desc="Segmenting"), start=1):
        out_name = f"{args.episode_id}_{idx}.mp4"
        out_path = os.path.join(clips_dir, out_name)
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-ss', start, '-to', end,
            '-c', 'copy', out_path
        ]

        if args.dry_run:
            print(f"[DRY RUN] Would run: {' '.join(cmd)}")
            continue

        # Run ffmpeg
        subprocess.run(cmd, check=True)

        # Compute size
        size_bytes = os.path.getsize(out_path)
        total_bytes += size_bytes

        # Compute duration saved
        duration_sec = parse_time(end) - parse_time(start)
        logging.info(
            f"Segment {idx}: {start} -> {end} | duration {duration_sec}s | "
            f"output {out_path} | size {size_bytes/1024/1024:.2f} MB"
        )

        # Update manifest entry
        entry = {
            "segment": out_name,
            "start": start,
            "end": end
        }
        if entry not in manifest:
            manifest.append(entry)

    # Write manifest if not dry-run
    if not args.dry_run:
        with open(manifest_path, 'w') as mf:
            json.dump(manifest, mf, indent=2)
        logging.info(f"Manifest updated at {manifest_path}")

        total_mb = total_bytes / (1024 * 1024)
        logging.info(f"Total clips size: {total_mb:.2f} MB")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
