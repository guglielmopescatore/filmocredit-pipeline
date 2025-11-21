#!/usr/bin/env python3
"""
Creates a CSV with the episode name and number of images present in
`data/episodes/<episode>/analysis/frames/`.

Usage:
  python count_frames_per_episode.py
  python count_frames_per_episode.py --output-dir exports
  python count_frames_per_episode.py --include-extensions .jpg .png
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable

try:
    # Use project config paths for consistency
    from scripts_v3 import config
    PROJECT_ROOT: Path = config.PathConfig.PROJECT_ROOT
    EPISODES_BASE_DIR: Path = config.PathConfig.EPISODES_BASE_DIR
except Exception:
    # Fallback: assume script is in project root
    PROJECT_ROOT = Path(__file__).resolve().parent
    EPISODES_BASE_DIR = PROJECT_ROOT / 'data' / 'episodes'


def iter_image_files(frames_dir: Path, exts: Iterable[str]) -> Iterable[Path]:
    if not frames_dir.is_dir():
        return []
    normalized_exts = {e.lower() if e.startswith('.') else f'.{e.lower()}' for e in exts}
    return [
        p for p in frames_dir.iterdir()
        if p.is_file() and p.suffix.lower() in normalized_exts
    ]


essential_exts = ['.jpg', '.jpeg', '.png']


def main():
    parser = argparse.ArgumentParser(description='Count frames per episode and export to CSV')
    parser.add_argument('--output-dir', type=str, default='exports', help='Directory to save the CSV (default: exports)')
    parser.add_argument('--include-extensions', nargs='*', default=essential_exts,
                        help='Image extensions to count (default: .jpg .jpeg .png)')
    parser.add_argument('--filename', type=str, default=None,
                        help='Custom CSV filename (default: frames_count_YYYYmmdd_HHMMSS.csv)')

    args = parser.parse_args()

    episodes_dir = EPISODES_BASE_DIR
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = args.filename or f'frames_count_{timestamp}.csv'
    out_path = output_dir / filename

    rows = []
    total_images = 0
    episodes_scanned = 0

    if not episodes_dir.exists():
        print(f"❌ Episodes directory not found: {episodes_dir}")
        return 1

    for ep_dir in sorted([p for p in episodes_dir.iterdir() if p.is_dir()]):
        episode_id = ep_dir.name
        frames_dir = ep_dir / 'analysis' / 'frames'
        images = iter_image_files(frames_dir, args.include_extensions)
        count = len(images)
        rows.append({'episode_id': episode_id, 'images_count': count})
        total_images += count
        episodes_scanned += 1

    # Write CSV
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['episode_id', 'images_count'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ CSV created: {out_path}")
    print(f"   Episodes scanned: {episodes_scanned}")
    print(f"   Total images:    {total_images}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
