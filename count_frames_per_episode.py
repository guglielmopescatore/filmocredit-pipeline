#!/usr/bin/env python3
"""
Creates a CSV with the episode name, number of images present in
`data/episodes/<episode>/analysis/frames/`, and theoretical frames
if extracted at 0.8 second intervals from the source video.

Usage:
  python count_frames_per_episode.py
  python count_frames_per_episode.py --output-dir exports
  python count_frames_per_episode.py --include-extensions .jpg .png
  python count_frames_per_episode.py --interval 0.5
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

try:
    from scenedetect import open_video
except ImportError:
    open_video = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    # Use project config paths for consistency
    from scripts_v3 import config
    PROJECT_ROOT: Path = config.PathConfig.PROJECT_ROOT
    EPISODES_BASE_DIR: Path = config.PathConfig.EPISODES_BASE_DIR
    RAW_VIDEO_DIR: Path = config.PathConfig.RAW_VIDEO_DIR
except Exception:
    # Fallback: assume script is in project root
    PROJECT_ROOT = Path(__file__).resolve().parent
    EPISODES_BASE_DIR = PROJECT_ROOT / 'data' / 'episodes'
    RAW_VIDEO_DIR = PROJECT_ROOT / 'data' / 'raw'


def iter_image_files(frames_dir: Path, exts: Iterable[str]) -> Iterable[Path]:
    if not frames_dir.is_dir():
        return []
    normalized_exts = {e.lower() if e.startswith('.') else f'.{e.lower()}' for e in exts}
    return [
        p for p in frames_dir.iterdir()
        if p.is_file() and p.suffix.lower() in normalized_exts
    ]


def get_video_duration_seconds(video_path: Path) -> float:
    """Get video duration in seconds using scenedetect."""
    if not open_video:
        return 0.0
    
    try:
        video = open_video(str(video_path))
        duration_seconds = video.duration.get_seconds()
        return duration_seconds
    except Exception as e:
        print(f"⚠️  Could not read video duration for {video_path.name}: {e}")
        return 0.0


def get_frame_dimensions(frame_path: Path) -> tuple[int, int]:
    """Get width and height of an image frame."""
    if not Image:
        return 0, 0
    
    try:
        with Image.open(frame_path) as img:
            return img.size  # Returns (width, height)
    except Exception as e:
        print(f"⚠️  Could not read frame dimensions for {frame_path.name}: {e}")
        return 0, 0


def calculate_tokens_from_dimensions(width: int, height: int) -> int:
    """Calculate Claude tokens using formula: tokens = (width * height) / 750"""
    if width == 0 or height == 0:
        return 0
    return int((width * height) / 750)


def calculate_theoretical_frames(episode_dir: Path, interval_seconds: float) -> tuple[int, float]:
    """
    Calculate theoretical frames if extracted at interval_seconds from source video.
    
    Returns:
        Tuple of (theoretical_frame_count, video_duration_seconds)
    """
    # Try to find video file in raw directory or episode directory
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv']
    
    # Search in episode directory first
    for ext in video_extensions:
        video_path = episode_dir / f"{episode_dir.name}{ext}"
        if video_path.exists():
            duration = get_video_duration_seconds(video_path)
            if duration > 0:
                theoretical_count = int(duration / interval_seconds)
                return theoretical_count, duration
    
    # Search in raw video directory
    if RAW_VIDEO_DIR.exists():
        for video_file in RAW_VIDEO_DIR.iterdir():
            if video_file.is_file() and video_file.suffix.lower() in video_extensions:
                # Match by episode name
                if episode_dir.name.lower() in video_file.stem.lower() or video_file.stem.lower() in episode_dir.name.lower():
                    duration = get_video_duration_seconds(video_file)
                    if duration > 0:
                        theoretical_count = int(duration / interval_seconds)
                        return theoretical_count, duration
    
    return 0, 0.0


essential_exts = ['.jpg', '.jpeg', '.png']


def main():
    parser = argparse.ArgumentParser(description='Count frames per episode and calculate theoretical frames from video')
    parser.add_argument('--output-dir', type=str, default='exports', help='Directory to save the CSV (default: exports)')
    parser.add_argument('--include-extensions', nargs='*', default=essential_exts,
                        help='Image extensions to count (default: .jpg .jpeg .png)')
    parser.add_argument('--filename', type=str, default=None,
                        help='Custom CSV filename (default: frames_count_YYYYmmdd_HHMMSS.csv)')
    parser.add_argument('--interval', type=float, default=0.8,
                        help='Interval in seconds for theoretical frame extraction (default: 0.8)')

    args = parser.parse_args()

    episodes_dir = EPISODES_BASE_DIR
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = args.filename or f'frames_count_{timestamp}.csv'
    out_path = output_dir / filename

    rows = []
    total_images = 0
    total_theoretical = 0
    total_tokens = 0
    total_cost = 0.0
    total_theoretical_tokens = 0
    total_theoretical_cost = 0.0
    total_theoretical_cost_with_caching = 0.0
    total_cached_cost = 0.0
    episodes_scanned = 0
    claude_cost_per_mtok = 3.0  # $3 per million tokens
    claude_cached_cost_per_mtok = 0.30  # $0.30 per million cached tokens
    prompt_tokens_per_call = 11303  # Fixed prompt tokens for each API call

    if not episodes_dir.exists():
        print(f"❌ Episodes directory not found: {episodes_dir}")
        return 1

    print(f"📊 Scanning episodes with interval={args.interval}s...\n")

    for ep_dir in sorted([p for p in episodes_dir.iterdir() if p.is_dir()]):
        episode_id = ep_dir.name
        frames_dir = ep_dir / 'analysis' / 'frames'
        images = list(iter_image_files(frames_dir, args.include_extensions))
        actual_count = len(images)
        
        # Calculate theoretical frames from video
        theoretical_count, duration = calculate_theoretical_frames(ep_dir, args.interval)
        
        # Calculate tokens from first frame (assuming all frames have same dimensions)
        width, height = 0, 0
        image_tokens_per_frame = 0
        tokens_per_frame = 0
        episode_tokens = 0
        episode_cost = 0.0
        episode_cost_with_caching = 0.0
        theoretical_tokens = 0
        theoretical_cost = 0.0
        theoretical_cost_with_caching = 0.0
        savings = 0.0
        savings_pct = 0.0
        savings_cached = 0.0
        savings_cached_pct = 0.0
        
        if images and actual_count > 0:
            width, height = get_frame_dimensions(images[0])
            image_tokens_per_frame = calculate_tokens_from_dimensions(width, height)
            # Total tokens per frame = image tokens + fixed prompt tokens
            tokens_per_frame = image_tokens_per_frame + prompt_tokens_per_call
            episode_tokens = tokens_per_frame * actual_count
            episode_cost = (episode_tokens / 1_000_000) * claude_cost_per_mtok
            
            # Calculate cost with prompt caching (first frame full cost, rest cached)
            if actual_count > 0:
                # First frame: full cost for both image and prompt
                first_frame_cost = (tokens_per_frame / 1_000_000) * claude_cost_per_mtok
                # Remaining frames: full cost for image, cached cost for prompt
                if actual_count > 1:
                    remaining_frames = actual_count - 1
                    remaining_image_cost = (image_tokens_per_frame * remaining_frames / 1_000_000) * claude_cost_per_mtok
                    remaining_prompt_cost = (prompt_tokens_per_call * remaining_frames / 1_000_000) * claude_cached_cost_per_mtok
                    episode_cost_with_caching = first_frame_cost + remaining_image_cost + remaining_prompt_cost
                else:
                    episode_cost_with_caching = first_frame_cost
            
            # Calculate theoretical cost if all frames were extracted
            if theoretical_count > 0:
                theoretical_tokens = tokens_per_frame * theoretical_count
                theoretical_cost = (theoretical_tokens / 1_000_000) * claude_cost_per_mtok
                savings = theoretical_cost - episode_cost
                savings_pct = (savings / theoretical_cost * 100) if theoretical_cost > 0 else 0
                
                # Calculate theoretical cost with caching
                first_frame_cost_theo = (tokens_per_frame / 1_000_000) * claude_cost_per_mtok
                if theoretical_count > 1:
                    remaining_frames_theo = theoretical_count - 1
                    remaining_image_cost_theo = (image_tokens_per_frame * remaining_frames_theo / 1_000_000) * claude_cost_per_mtok
                    remaining_prompt_cost_theo = (prompt_tokens_per_call * remaining_frames_theo / 1_000_000) * claude_cached_cost_per_mtok
                    theoretical_cost_with_caching = first_frame_cost_theo + remaining_image_cost_theo + remaining_prompt_cost_theo
                else:
                    theoretical_cost_with_caching = first_frame_cost_theo
                
                savings_cached = theoretical_cost_with_caching - episode_cost_with_caching
                savings_cached_pct = (savings_cached / theoretical_cost_with_caching * 100) if theoretical_cost_with_caching > 0 else 0
        
        rows.append({
            'episode_id': episode_id,
            'actual_frames': actual_count,
            'video_duration_seconds': f"{duration:.2f}" if duration > 0 else "N/A",
            'theoretical_frames_at_interval': theoretical_count if theoretical_count > 0 else "N/A",
            'frame_width': width if width > 0 else "N/A",
            'frame_height': height if height > 0 else "N/A",
            'image_tokens_per_frame': image_tokens_per_frame if image_tokens_per_frame > 0 else "N/A",
            'prompt_tokens_per_frame': prompt_tokens_per_call,
            'total_tokens_per_frame': tokens_per_frame if tokens_per_frame > 0 else "N/A",
            'actual_tokens': episode_tokens if episode_tokens > 0 else "N/A",
            'actual_cost_no_cache_usd': f"{episode_cost:.4f}" if episode_cost > 0 else "N/A",
            'actual_cost_with_cache_usd': f"{episode_cost_with_caching:.4f}" if episode_cost_with_caching > 0 else "N/A",
            'theoretical_tokens': theoretical_tokens if theoretical_tokens > 0 else "N/A",
            'theoretical_cost_no_cache_usd': f"{theoretical_cost:.4f}" if theoretical_cost > 0 else "N/A",
            'theoretical_cost_with_cache_usd': f"{theoretical_cost_with_caching:.4f}" if theoretical_cost_with_caching > 0 else "N/A",
            'savings_no_cache_usd': f"{savings:.4f}" if savings > 0 else "N/A",
            'savings_no_cache_percent': f"{savings_pct:.1f}" if savings_pct > 0 else "N/A",
            'savings_with_cache_usd': f"{savings_cached:.4f}" if savings_cached > 0 else "N/A",
            'savings_with_cache_percent': f"{savings_cached_pct:.1f}" if savings_cached_pct > 0 else "N/A"
        })
        
        total_images += actual_count
        if theoretical_count > 0:
            total_theoretical += theoretical_count
        if episode_tokens > 0:
            total_tokens += episode_tokens
            total_cost += episode_cost
        if episode_cost_with_caching > 0:
            total_cached_cost += episode_cost_with_caching
        if theoretical_tokens > 0:
            total_theoretical_tokens += theoretical_tokens
            total_theoretical_cost += theoretical_cost
        if theoretical_cost_with_caching > 0:
            total_theoretical_cost_with_caching += theoretical_cost_with_caching
        episodes_scanned += 1
        
        # Print progress
        if theoretical_count > 0:
            percentage = (actual_count / theoretical_count * 100) if theoretical_count > 0 else 0
            savings_str = f"💰 saved ${savings:.2f} ({savings_pct:.1f}%)" if savings > 0 else ""
            print(f"  {episode_id}: {actual_count}/{theoretical_count} frames ({percentage:.1f}%) | ${episode_cost:.2f} vs ${theoretical_cost:.2f} | {savings_str}")
        else:
            cost_str = f"${episode_cost:.2f}" if episode_cost > 0 else "N/A"
            print(f"  {episode_id}: {actual_count} frames | ⚠️ video not found | {cost_str}")

    # Write CSV
    with out_path.open('w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'episode_id', 
            'actual_frames', 
            'video_duration_seconds', 
            'theoretical_frames_at_interval',
            'frame_width',
            'frame_height',
            'image_tokens_per_frame',
            'prompt_tokens_per_frame',
            'total_tokens_per_frame',
            'actual_tokens',
            'actual_cost_no_cache_usd',
            'actual_cost_with_cache_usd',
            'theoretical_tokens',
            'theoretical_cost_no_cache_usd',
            'theoretical_cost_with_cache_usd',
            'savings_no_cache_usd',
            'savings_no_cache_percent',
            'savings_with_cache_usd',
            'savings_with_cache_percent'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Calculate total savings
    total_savings_no_cache = total_theoretical_cost - total_cost
    total_savings_pct_no_cache = (total_savings_no_cache / total_theoretical_cost * 100) if total_theoretical_cost > 0 else 0
    
    total_savings_with_cache = total_theoretical_cost_with_caching - total_cached_cost
    total_savings_pct_with_cache = (total_savings_with_cache / total_theoretical_cost_with_caching * 100) if total_theoretical_cost_with_caching > 0 else 0

    # Print summary
    print(f"\n{'='*100}")
    print(f"✅ CSV created: {out_path}")
    print(f"\n📈 FRAME ANALYSIS:")
    print(f"   Episodes scanned:      {episodes_scanned}")
    print(f"   Total actual frames:   {total_images:,}")
    if total_theoretical > 0:
        print(f"   Total theoretical:     {total_theoretical:,} (at {args.interval}s intervals)")
        print(f"   Frame coverage:        {total_images / total_theoretical * 100:.1f}%")
    
    if total_tokens > 0:
        print(f"\n💰 COST ANALYSIS:")
        print(f"\n   📊 SCENARIO 1: Without Prompt Caching (${claude_cost_per_mtok}/MTok for all tokens)")
        print(f"   ─────────────────────────────────────────────────────────")
        print(f"   ACTUAL (optimized approach):")
        print(f"     • Tokens:                 {total_tokens:,}")
        print(f"     • Cost:                   ${total_cost:.2f}")
        print(f"     • Avg/episode:            ${total_cost / episodes_scanned:.2f}" if episodes_scanned > 0 else "")
        
        if total_theoretical_tokens > 0:
            print(f"\n   THEORETICAL (naive 1 frame every {args.interval}s):")
            print(f"     • Tokens:                 {total_theoretical_tokens:,}")
            print(f"     • Cost:                   ${total_theoretical_cost:.2f}")
            print(f"     • Avg/episode:            ${total_theoretical_cost / episodes_scanned:.2f}" if episodes_scanned > 0 else "")
            
            print(f"\n   💵 SAVINGS (frame deduplication only):")
            print(f"     • Total saved:            ${total_savings_no_cache:.2f} ({total_savings_pct_no_cache:.1f}%)")
            print(f"     • Avg saved/episode:      ${total_savings_no_cache / episodes_scanned:.2f}" if episodes_scanned > 0 else "")
        
        if total_cached_cost > 0 and total_theoretical_cost_with_caching > 0:
            print(f"\n   📊 SCENARIO 2: With Prompt Caching (prompt @ ${claude_cached_cost_per_mtok}/MTok after 1st frame)")
            print(f"   ─────────────────────────────────────────────────────────")
            print(f"   ACTUAL (optimized with caching):")
            print(f"     • Cost:                   ${total_cached_cost:.2f}")
            print(f"     • Avg/episode:            ${total_cached_cost / episodes_scanned:.2f}" if episodes_scanned > 0 else "")
            
            actual_cache_benefit = total_cost - total_cached_cost
            actual_cache_benefit_pct = (actual_cache_benefit / total_cost * 100) if total_cost > 0 else 0
            print(f"     • Cache benefit:          ${actual_cache_benefit:.2f} saved vs no-cache ({actual_cache_benefit_pct:.1f}%)")
            
            print(f"\n   THEORETICAL (naive 1 frame every {args.interval}s with caching):")
            print(f"     • Cost:                   ${total_theoretical_cost_with_caching:.2f}")
            print(f"     • Avg/episode:            ${total_theoretical_cost_with_caching / episodes_scanned:.2f}" if episodes_scanned > 0 else "")
            
            theo_cache_benefit = total_theoretical_cost - total_theoretical_cost_with_caching
            theo_cache_benefit_pct = (theo_cache_benefit / total_theoretical_cost * 100) if total_theoretical_cost > 0 else 0
            print(f"     • Cache benefit:          ${theo_cache_benefit:.2f} saved vs no-cache ({theo_cache_benefit_pct:.1f}%)")
            
            print(f"\n   💵 SAVINGS (frame deduplication + caching):")
            print(f"     • Total saved:            ${total_savings_with_cache:.2f} ({total_savings_pct_with_cache:.1f}%)")
            print(f"     • Avg saved/episode:      ${total_savings_with_cache / episodes_scanned:.2f}" if episodes_scanned > 0 else "")
    
    print(f"{'='*100}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
