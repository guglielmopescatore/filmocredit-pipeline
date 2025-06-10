#!/usr/bin/env python3
"""
manual_segmentation.py: Manual video segmentation module for scripts_v3

This module provides the same functionality as the original 01_segment_video.py
but integrated with the scripts_v3 architecture.
"""
import argparse
import subprocess
import os
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm

from . import config, utils

logger = logging.getLogger(__name__)


def parse_time(ts: str) -> int:
    """Convert hh:mm:ss to total seconds."""
    try:
        h, m, s = ts.split(':')
        return int(h) * 3600 + int(m) * 60 + int(s)
    except (ValueError, AttributeError):
        raise ValueError(f"Invalid time format: {ts}. Expected hh:mm:ss")


def validate_ffmpeg() -> bool:
    """Check if ffmpeg is available in the system."""
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def segment_video_manually(
    input_path: str,
    segments: List[Tuple[str, str]],
    episode_id: str,
    output_dir: Optional[str] = None,
    dry_run: bool = False
) -> List[str]:
    """
    Segment video into clips using ffmpeg with manual time ranges.
    
    Args:
        input_path: Path to input video file
        segments: List of (start_time, end_time) tuples in hh:mm:ss format
        episode_id: Episode identifier
        output_dir: Output directory (defaults to config)
        dry_run: If True, only print commands without executing
        
    Returns:
        List of output clip paths
    """
    input_path = os.path.abspath(input_path)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if not validate_ffmpeg():
        raise RuntimeError("ffmpeg not found. Please install ffmpeg and ensure it's in your PATH")
    
    # Use config or default output directory
    if output_dir is None:
        output_dir = config.get_processed_dir()
    
    # Prepare output directories
    clips_dir = Path(output_dir) / 'clips'
    clips_dir.mkdir(parents=True, exist_ok=True)
    
    episode_dir = Path(output_dir) / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate segments
    validated_segments = []
    for start, end in segments:
        try:
            start_sec = parse_time(start)
            end_sec = parse_time(end)
            if start_sec >= end_sec:
                logger.warning(f"Invalid segment: {start}-{end} (start >= end)")
                continue
            validated_segments.append((start, end))
        except ValueError as e:
            logger.error(f"Invalid segment format {start}-{end}: {e}")
            continue
    
    if not validated_segments:
        raise ValueError("No valid segments provided")
    
    # Load or initialize manifest
    manifest_path = episode_dir / 'segments.json'
    if manifest_path.exists():
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
    else:
        manifest = []
    
    output_clips = []
    total_bytes = 0
    
    # Process segments
    for idx, (start, end) in enumerate(tqdm(validated_segments, desc="Segmenting video"), start=1):
        out_name = f"{episode_id}_{idx:03d}.mp4"
        out_path = clips_dir / out_name
        
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-ss', start, '-to', end,
            '-c', 'copy', str(out_path)
        ]
        
        if dry_run:
            logger.info(f"[DRY RUN] Would run: {' '.join(cmd)}")
            output_clips.append(str(out_path))
            continue
        
        try:
            # Run ffmpeg
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Compute size and duration
            size_bytes = out_path.stat().st_size
            total_bytes += size_bytes
            duration_sec = parse_time(end) - parse_time(start)
            
            logger.info(
                f"Segment {idx}: {start} -> {end} | duration {duration_sec}s | "
                f"output {out_path.name} | size {size_bytes/1024/1024:.2f} MB"
            )
            
            # Update manifest entry
            entry = {
                "segment": out_name,
                "start": start,
                "end": end,
                "duration_sec": duration_sec,
                "size_bytes": size_bytes,
                "clip_path": str(out_path)
            }
            
            # Check if entry already exists
            existing_entry = next((e for e in manifest if e.get("segment") == out_name), None)
            if existing_entry:
                manifest[manifest.index(existing_entry)] = entry
            else:
                manifest.append(entry)
            
            output_clips.append(str(out_path))
            
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg failed for segment {idx} ({start}-{end}): {e}")
            logger.error(f"ffmpeg stderr: {e.stderr}")
            continue
    
    # Write manifest if not dry-run
    if not dry_run and manifest:
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        logger.info(f"Manifest updated at {manifest_path}")
        
        total_mb = total_bytes / (1024 * 1024)
        logger.info(f"Total clips size: {total_mb:.2f} MB")
    
    return output_clips


def parse_segments_string(segments_str: str) -> List[Tuple[str, str]]:
    """
    Parse comma-separated segments string into list of tuples.
    
    Args:
        segments_str: Comma-separated list of start-end pairs (hh:mm:ss-hh:mm:ss)
        
    Returns:
        List of (start_time, end_time) tuples
    """
    if not segments_str:
        return []
    
    segments = []
    raw_segs = segments_str.split(',')
    
    for seg in raw_segs:
        seg = seg.strip()
        if not seg:
            continue
            
        try:
            start, end = seg.split('-', 1)
            segments.append((start.strip(), end.strip()))
        except ValueError:
            logger.error(f"Invalid segment format: {seg}. Expected format: hh:mm:ss-hh:mm:ss")
            continue
    
    return segments


def main():
    """Command-line interface for manual video segmentation."""
    parser = argparse.ArgumentParser(
        description="Segment video losslessly into clips using ffmpeg"
    )
    parser.add_argument('--input', required=True,
                        help='Path to input video file')
    parser.add_argument('--segments', required=True,
                        help='Comma-separated list of start-end pairs (hh:mm:ss-hh:mm:ss)')
    parser.add_argument('--episode-id', required=True,
                        help='Episode identifier, e.g., tt1234567')
    parser.add_argument('--output-dir', 
                        help='Output directory (defaults to config)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing')
    
    args = parser.parse_args()
    
    # Parse segments
    segments = parse_segments_string(args.segments)
    if not segments:
        logger.error("No valid segments provided")
        return 1
    
    try:
        output_clips = segment_video_manually(
            input_path=args.input,
            segments=segments,
            episode_id=args.episode_id,
            output_dir=args.output_dir,
            dry_run=args.dry_run
        )
        
        logger.info(f"Successfully processed {len(output_clips)} segments")
        for clip in output_clips:
            logger.info(f"  -> {clip}")
            
        return 0
        
    except Exception as e:
        logger.error(f"Error during segmentation: {e}")
        return 1


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    sys.exit(main())
