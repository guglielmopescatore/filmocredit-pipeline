#!/usr/bin/env python3
"""
Estrae costanti duplicate e crea file constants.py
"""

import ast
import re
from pathlib import Path
from collections import defaultdict

# Costanti da estrarre (aggiunte durante l'analisi)
CONSTANTS_TO_EXTRACT = {
    # Numeri magici ricorrenti
    'MIN_OCR_CONFIDENCE': 0.75,
    'MIN_OCR_TEXT_LENGTH': 4,
    'MAX_OCR_ATTEMPTS': 3,
    'HASH_SIZE': 16,
    'FADE_FRAME_THRESHOLD': 20.0,
    'FADE_FRAME_CONTRAST_THRESHOLD': 10.0,

    # Stringhe ricorrenti
    'DEFAULT_OCR_ENGINE': "paddleocr",
    'DB_TABLE_CREDITS': "credits",
    'DB_TABLE_EPISODES': "episodes",
}

def extract_magic_numbers(filepath):
    """Trova numeri magici nel codice."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Pattern per numeri
    numbers = re.findall(r'\b\d+\.?\d*\b', content)

    # Pattern per stringhe quotate
    strings = re.findall(r'["\']([^"\']+)["\']', content)

    return numbers, strings

def create_constants_file():
    """Crea il nuovo file constants.py."""

    content = '''"""
Costanti condivise per il progetto FilmOCredit.
Estratte automaticamente per evitare duplicazione.
"""

# OCR Configuration
MIN_OCR_CONFIDENCE = 0.75
MIN_OCR_TEXT_LENGTH = 4
MAX_OCR_ATTEMPTS = 3
OCR_TIMEOUT_SECONDS = 3

# Frame Processing
HASH_SIZE = 16
HASH_DIFFERENCE_THRESHOLD = 0
FADE_FRAME_THRESHOLD = 20.0
FADE_FRAME_CONTRAST_THRESHOLD = 10.0
SCROLL_FRAME_HEIGHT_RATIO = 0.9
MIN_ABS_SCROLL_FLOW_THRESHOLD = 0.5

# Text Processing
FUZZY_TEXT_SIMILARITY_THRESHOLD = 60
FUZZY_THRESHOLD_BASE = 60
FUZZY_THRESHOLD_SCALE_START = 200
FUZZY_THRESHOLD_SCALE_RATE = 0.02
FUZZY_THRESHOLD_MAX = 85

# Scene Detection
CONTENT_SCENE_DETECTOR_THRESHOLD = 10.0
THRESH_SCENE_DETECTOR_THRESHOLD = 5
SCENE_MIN_LENGTH_FRAMES = 10
DEFAULT_START_SCENES_COUNT = 100
DEFAULT_END_SCENES_COUNT = 100

# Database
DB_TABLE_CREDITS = "credits"
DB_TABLE_EPISODES = "episodes"

# File Patterns
FRAME_FILENAME_PATTERN = "{prefix}frame_num{num:05d}_sim{sim}.jpg"
SKIP_FILENAME_PATTERN = "{prefix}{mode}skipped_{reason}_num{num:05d}.jpg"

# Azure VLM
DEFAULT_VLM_MAX_NEW_TOKENS = 8192
MAX_API_RETRIES = 3
BACKOFF_FACTOR = 2.0
'''

    with open('scripts_v3/constants.py', 'w') as f:
        f.write(content)

    print("✓ Created constants.py")

if __name__ == "__main__":
    create_constants_file()
