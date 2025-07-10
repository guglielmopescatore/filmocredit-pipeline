#!/usr/bin/env python3
"""
Video File Name Sanitizer
Checks and sanitizes video file names in data/raw folder before program startup.
Replaces spaces with underscores to ensure compatibility.
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple
from .video_formats import get_supported_extensions_set

def sanitize_video_filenames(data_raw_path: Path) -> Tuple[int, List[str]]:
    """
    Sanitize video file names by replacing spaces with underscores.
    
    Args:
        data_raw_path: Path to the data/raw directory
        
    Returns:
        Tuple of (number of files renamed, list of rename operations)
    """
    if not data_raw_path.exists():
        logging.warning(f"Data/raw directory not found: {data_raw_path}")
        return 0, []
    
    # Video file extensions supported by the main program
    # Synchronized with app.py allowed_extensions
    video_extensions = {'.mp4', '.mkv', '.avi', '.mov'}
    
    renamed_files = []
    rename_operations = []
    
    # Find all video files with spaces in their names
    for file_path in data_raw_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            original_name = file_path.name
            
            # Check if filename contains spaces
            if ' ' in original_name:
                # Create new name with underscores instead of spaces
                new_name = original_name.replace(' ', '_')
                new_path = file_path.parent / new_name
                
                # Check if target file already exists
                if new_path.exists():
                    logging.warning(f"Cannot rename '{original_name}' to '{new_name}' - target file already exists")
                    continue
                
                try:
                    # Rename the file
                    file_path.rename(new_path)
                    renamed_files.append(original_name)
                    rename_operations.append(f"'{original_name}' → '{new_name}'")
                    logging.info(f"Renamed video file: '{original_name}' → '{new_name}'")
                    
                except OSError as e:
                    logging.error(f"Failed to rename '{original_name}': {e}")
                    continue
    
    return len(renamed_files), rename_operations


def check_and_sanitize_video_files() -> bool:
    """
    Main function to check and sanitize video files at program startup.
    
    Returns:
        True if operation completed successfully, False if critical errors occurred
    """
    try:
        # Get the current working directory and construct path to data/raw
        current_dir = Path.cwd()
        data_raw_path = current_dir / "data" / "raw"
        
        logging.info("🎬 Starting video file name sanitization check...")
        
        # Check if data/raw directory exists
        if not data_raw_path.exists():
            logging.info(f"📁 Data/raw directory not found at {data_raw_path}")
            logging.info("   This is normal if no video files have been added yet.")
            return True
        
        # Perform sanitization
        renamed_count, rename_operations = sanitize_video_filenames(data_raw_path)
        
        if renamed_count > 0:
            logging.info(f"✅ Video file sanitization completed!")
            logging.info(f"   📊 Renamed {renamed_count} file(s):")
            for operation in rename_operations:
                logging.info(f"      • {operation}")
        else:
            # Check if there are any video files at all
            # Video file extensions supported by the main program
            video_extensions = {'.mp4', '.mkv', '.avi', '.mov'}
            video_files = [f for f in data_raw_path.iterdir() 
                          if f.is_file() and f.suffix.lower() in video_extensions]
            
            if video_files:
                logging.info(f"✅ All {len(video_files)} video file(s) already have proper names (no spaces)")
            else:
                logging.info("📂 No video files found in data/raw directory")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Error during video file sanitization: {e}")
        logging.error("   Program will continue, but manual file renaming may be needed")
        return False


def preview_sanitization_changes(data_raw_path: Path = None) -> List[Tuple[str, str]]:
    """
    Preview what files would be renamed without actually renaming them.
    Useful for testing or showing users what will happen.
    
    Args:
        data_raw_path: Path to check (defaults to current directory + data/raw)
        
    Returns:
        List of (original_name, new_name) tuples for files that would be renamed
    """
    if data_raw_path is None:
        data_raw_path = Path.cwd() / "data" / "raw"
    
    if not data_raw_path.exists():
        return []
    
    # Video file extensions supported by the main program
    video_extensions = {'.mp4', '.mkv', '.avi', '.mov'}
    changes = []
    
    for file_path in data_raw_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            original_name = file_path.name
            if ' ' in original_name:
                new_name = original_name.replace(' ', '_')
                changes.append((original_name, new_name))
    
    return changes


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🎬 Video File Name Sanitizer")
    print("=" * 50)
    
    # Preview changes first
    preview_changes = preview_sanitization_changes()
    if preview_changes:
        print(f"📋 Found {len(preview_changes)} video file(s) with spaces:")
        for original, new in preview_changes:
            print(f"   • '{original}' → '{new}'")
        
        response = input("\n🤔 Proceed with renaming? (y/N): ").strip().lower()
        if response != 'y':
            print("❌ Operation cancelled by user")
            exit(0)
    else:
        print("✅ No video files with spaces found")
        exit(0)
    
    # Perform sanitization
    success = check_and_sanitize_video_files()
    if success:
        print("🎉 Video file sanitization completed successfully!")
    else:
        print("⚠️ Some issues occurred during sanitization")
        exit(1)
