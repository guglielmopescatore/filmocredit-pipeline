#!/usr/bin/env python3
"""
Delete OCR folders from all episodes.

This script safely deletes all /ocr folders inside each episode directory
under data/episodes/, allowing you to re-run Step 3 (VLM processing) from scratch.

Usage:
    python delete_ocr_folders.py
    python delete_ocr_folders.py --dry-run
    python delete_ocr_folders.py --episode "Episode_Name"
"""

import argparse
import shutil
import sys
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent


def get_episodes_dir() -> Path:
    """Get the episodes directory."""
    return get_project_root() / 'data' / 'episodes'


def delete_ocr_folder(episode_path: Path, dry_run: bool = False) -> tuple[bool, str]:
    """
    Delete the OCR folder for a specific episode.
    
    Args:
        episode_path: Path to the episode directory
        dry_run: If True, only show what would be deleted without actually deleting
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    ocr_folder = episode_path / 'ocr'
    
    if not ocr_folder.exists():
        return True, f"No OCR folder found in {episode_path.name}"
    
    if not ocr_folder.is_dir():
        return False, f"OCR path exists but is not a directory: {ocr_folder}"
    
    try:
        if dry_run:
            # Count files that would be deleted
            file_count = sum(1 for _ in ocr_folder.rglob('*') if _.is_file())
            return True, f"[DRY RUN] Would delete {ocr_folder} ({file_count} files)"
        else:
            shutil.rmtree(ocr_folder)
            return True, f"✅ Deleted {ocr_folder}"
    except Exception as e:
        return False, f"❌ Error deleting {ocr_folder}: {e}"


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Delete OCR folders from all episodes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Delete all OCR folders:
    python delete_ocr_folders.py
    
  Preview what would be deleted (dry run):
    python delete_ocr_folders.py --dry-run
    
  Delete OCR folder for specific episode:
    python delete_ocr_folders.py --episode "Amelie"
    
  Dry run for specific episode:
    python delete_ocr_folders.py --episode "Amelie" --dry-run
        """
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    
    parser.add_argument(
        '--episode',
        type=str,
        help='Delete OCR folder only for specific episode (default: all episodes)'
    )
    
    args = parser.parse_args()
    
    # Get episodes directory
    episodes_dir = get_episodes_dir()
    
    if not episodes_dir.exists():
        print(f"❌ Error: Episodes directory not found at {episodes_dir}")
        sys.exit(1)
    
    # Get list of episodes to process
    if args.episode:
        episode_path = episodes_dir / args.episode
        if not episode_path.exists():
            print(f"❌ Error: Episode '{args.episode}' not found at {episode_path}")
            sys.exit(1)
        episodes = [episode_path]
    else:
        # Get all episode directories
        episodes = [d for d in episodes_dir.iterdir() if d.is_dir()]
    
    if not episodes:
        print(f"❌ No episodes found in {episodes_dir}")
        sys.exit(1)
    
    # Display header
    mode_text = "DRY RUN - " if args.dry_run else ""
    print(f"\n{mode_text}Deleting OCR folders...")
    print(f"Episodes directory: {episodes_dir}")
    print(f"Episodes to process: {len(episodes)}\n")
    
    # Process each episode
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for episode_path in sorted(episodes):
        success, message = delete_ocr_folder(episode_path, dry_run=args.dry_run)
        
        if success:
            if "No OCR folder" in message:
                skip_count += 1
                print(f"⏭️  {message}")
            else:
                success_count += 1
                print(message)
        else:
            error_count += 1
            print(message)
    
    # Summary
    print(f"\n{'='*60}")
    if args.dry_run:
        print("DRY RUN SUMMARY:")
        print(f"  Would delete: {success_count}")
    else:
        print("SUMMARY:")
        print(f"  Deleted: {success_count}")
    print(f"  Skipped (no OCR folder): {skip_count}")
    print(f"  Errors: {error_count}")
    print(f"{'='*60}\n")
    
    if args.dry_run and success_count > 0:
        print("ℹ️  This was a dry run. Run without --dry-run to actually delete folders.")
    
    sys.exit(0 if error_count == 0 else 1)


if __name__ == '__main__':
    main()
