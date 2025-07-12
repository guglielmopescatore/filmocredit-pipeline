import os
import logging
from pathlib import Path
from typing import List, Tuple
from .video_formats import get_supported_extensions_set

def sanitize_video_filenames(data_raw_path: Path) -> Tuple[int, List[str]]:
    """
    Sanitize video file names by replacing spaces with underscores.
    If a file with underscores already exists, delete the file with spaces.
    
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
                    try:
                        # Delete the file with spaces since underscored version already exists
                        file_path.unlink()
                        rename_operations.append(f"Deleted '{original_name}' ('{new_name}' already exists)")
                        logging.info(f"Deleted duplicate video file: '{original_name}' ('{new_name}' already exists)")
                        renamed_files.append(original_name)  # Count as processed
                        
                    except OSError as e:
                        logging.error(f"Failed to delete '{original_name}': {e}")
                        continue
                else:
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
            logging.info(f"   📊 Processed {renamed_count} file(s):")
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
    Preview what files would be renamed or deleted without actually doing it.
    Useful for testing or showing users what will happen.
    
    Args:
        data_raw_path: Path to check (defaults to current directory + data/raw)
        
    Returns:
        List of (original_name, action) tuples for files that would be processed
        where action is either the new_name or "DELETE (duplicate exists)"
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
                new_path = file_path.parent / new_name
                
                if new_path.exists():
                    changes.append((original_name, "DELETE (duplicate exists)"))
                else:
                    changes.append((original_name, new_name))
    
    return changes