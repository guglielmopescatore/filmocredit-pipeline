#!/usr/bin/env python3
"""
Video format utilities for the filmocredit pipeline
Centralizes the definition of supported video formats
"""

# Supported video file extensions
# These are the formats that the main program can process
SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.webm'}

def is_supported_video_file(file_path) -> bool:
    """
    Check if a file is a supported video format
    
    Args:
        file_path: Path object or string path to the file
        
    Returns:
        True if the file extension is supported, False otherwise
    """
    from pathlib import Path
    
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    return file_path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS

def get_supported_extensions_tuple():
    """
    Get supported extensions as a tuple (for compatibility with existing code)
    
    Returns:
        Tuple of supported video extensions
    """
    return tuple(SUPPORTED_VIDEO_EXTENSIONS)

def get_supported_extensions_set():
    """
    Get supported extensions as a set
    
    Returns:
        Set of supported video extensions
    """
    return SUPPORTED_VIDEO_EXTENSIONS.copy()

def format_supported_extensions():
    """
    Format supported extensions for display to users
    
    Returns:
        Human-readable string of supported formats
    """
    extensions = sorted(SUPPORTED_VIDEO_EXTENSIONS)
    return ", ".join(ext.upper() for ext in extensions)

if __name__ == "__main__":
    print("🎬 Supported Video Formats")
    print("=" * 30)
    print(f"Extensions: {format_supported_extensions()}")
    print(f"Count: {len(SUPPORTED_VIDEO_EXTENSIONS)} formats")
    print(f"Tuple format: {get_supported_extensions_tuple()}")
    print(f"Set format: {get_supported_extensions_set()}")
    
    # Test with sample files
    test_files = [
        "movie.mp4",     # ✅ Supported
        "video.avi",     # ✅ Supported
        "film.mkv",      # ✅ Supported
        "clip.mov",      # ✅ Supported
        "video.wmv",     # ❌ Not supported
        "movie.flv",     # ❌ Not supported
        "document.txt"   # ❌ Not a video
    ]
    
    print(f"\n📋 File Support Test:")
    for file_name in test_files:
        supported = is_supported_video_file(file_name)
        status = "✅" if supported else "❌"
        print(f"   {status} {file_name}")
