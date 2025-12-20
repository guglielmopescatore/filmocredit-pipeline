#!/usr/bin/env python3
"""
Rename OCR folders from /ocr to /claude_ocr for all episodes.

This mirrors delete_ocr_folders.py but renames instead of deleting, so
existing OCR outputs are preserved under a different folder.

Usage:
    python rename_ocr_to_claude.py
    python rename_ocr_to_claude.py --dry-run
    python rename_ocr_to_claude.py --episode "Episode_Name"
"""

import argparse
import sys
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent


def get_episodes_dir() -> Path:
    return get_project_root() / "data" / "episodes"


def rename_ocr_folder(episode_path: Path, dry_run: bool = False) -> tuple[bool, str]:
    """Rename the OCR folder to claude_ocr for a specific episode."""
    ocr_folder = episode_path / "ocr"
    target_folder = episode_path / "claude_ocr"

    if not ocr_folder.exists():
        return True, f"No OCR folder found in {episode_path.name}"

    if not ocr_folder.is_dir():
        return False, f"OCR path exists but is not a directory: {ocr_folder}"

    if target_folder.exists():
        return False, f"Target folder already exists: {target_folder}"

    if dry_run:
        file_count = sum(1 for _ in ocr_folder.rglob("*") if _.is_file())
        return True, f"[DRY RUN] Would rename {ocr_folder} -> {target_folder} ({file_count} files)"

    try:
        ocr_folder.rename(target_folder)
        return True, f"✅ Renamed {ocr_folder} -> {target_folder}"
    except Exception as e:
        return False, f"❌ Error renaming {ocr_folder}: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Rename OCR folders to claude_ocr",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Rename all OCR folders:
    python rename_ocr_to_claude.py

  Preview what would be renamed (dry run):
    python rename_ocr_to_claude.py --dry-run

  Rename OCR folder for specific episode:
    python rename_ocr_to_claude.py --episode "Amelie"

  Dry run for specific episode:
    python rename_ocr_to_claude.py --episode "Amelie" --dry-run
        """,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be renamed without actually renaming",
    )

    parser.add_argument(
        "--episode",
        type=str,
        help="Rename OCR folder only for specific episode (default: all episodes)",
    )

    args = parser.parse_args()

    episodes_dir = get_episodes_dir()
    if not episodes_dir.exists():
        print(f"❌ Error: Episodes directory not found at {episodes_dir}")
        sys.exit(1)

    if args.episode:
        episode_path = episodes_dir / args.episode
        if not episode_path.exists():
            print(f"❌ Error: Episode '{args.episode}' not found at {episode_path}")
            sys.exit(1)
        episodes = [episode_path]
    else:
        episodes = [d for d in episodes_dir.iterdir() if d.is_dir()]

    if not episodes:
        print(f"❌ No episodes found in {episodes_dir}")
        sys.exit(1)

    mode_text = "DRY RUN - " if args.dry_run else ""
    print(f"\n{mode_text}Renaming OCR folders to claude_ocr...")
    print(f"Episodes directory: {episodes_dir}")
    print(f"Episodes to process: {len(episodes)}\n")

    success_count = 0
    skip_count = 0
    error_count = 0

    for episode_path in sorted(episodes):
        success, message = rename_ocr_folder(episode_path, dry_run=args.dry_run)

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

    print(f"\n{'='*60}")
    if args.dry_run:
        print("DRY RUN SUMMARY:")
        print(f"  Would rename: {success_count}")
    else:
        print("SUMMARY:")
        print(f"  Renamed: {success_count}")
    print(f"  Skipped (no OCR folder): {skip_count}")
    print(f"  Errors: {error_count}")
    print(f"{'='*60}\n")

    if args.dry_run and success_count > 0:
        print("ℹ️  This was a dry run. Run without --dry-run to actually rename folders.")

    sys.exit(0 if error_count == 0 else 1)


if __name__ == "__main__":
    main()
