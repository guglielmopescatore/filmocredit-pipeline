"""
Costanti condivise per il progetto FilmOCredit.
Centralizza valori magic numbers e stringhe ricorrenti.
"""

from typing import Final, List

# OCR Configuration
MIN_OCR_CONFIDENCE: Final[float] = 0.75
MIN_OCR_TEXT_LENGTH: Final[int] = 4
MAX_OCR_ATTEMPTS: Final[int] = 3
OCR_TIMEOUT_SECONDS: Final[int] = 3

# Frame Processing
HASH_SIZE: Final[int] = 16
HASH_DIFFERENCE_THRESHOLD: Final[int] = 0
FADE_FRAME_THRESHOLD: Final[float] = 20.0
FADE_FRAME_CONTRAST_THRESHOLD: Final[float] = 10.0
SCROLL_FRAME_HEIGHT_RATIO: Final[float] = 0.9
MIN_ABS_SCROLL_FLOW_THRESHOLD: Final[float] = 0.5
HASH_SAMPLE_INTERVAL_SECONDS: Final[float] = 0.5
MIN_CONTRAST_IMPROVEMENT_THRESHOLD: Final[float] = 5.0
CONTRAST_CALCULATION_METHOD: Final[str] = "stddev"

# Text Processing
FUZZY_TEXT_SIMILARITY_THRESHOLD: Final[int] = 60
FUZZY_THRESHOLD_BASE: Final[int] = 60
FUZZY_THRESHOLD_SCALE_START: Final[int] = 200
FUZZY_THRESHOLD_SCALE_RATE: Final[float] = 0.02
FUZZY_THRESHOLD_MAX: Final[int] = 85

# Scene Detection
CONTENT_SCENE_DETECTOR_THRESHOLD: Final[float] = 10.0
THRESH_SCENE_DETECTOR_THRESHOLD: Final[int] = 5
SCENE_MIN_LENGTH_FRAMES: Final[int] = 10
DEFAULT_START_SCENES_COUNT: Final[int] = 100
DEFAULT_END_SCENES_COUNT: Final[int] = 100
INITIAL_FRAME_SAMPLE_POINTS: Final[List[float]] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Azure VLM
DEFAULT_VLM_MAX_NEW_TOKENS: Final[int] = 8192
MAX_API_RETRIES: Final[int] = 3
BACKOFF_FACTOR: Final[float] = 2.0

# File Patterns
FRAME_FILENAME_PATTERN: Final[str] = "{prefix}frame_num{num:05d}_sim{sim}.jpg"
SKIP_FILENAME_PATTERN: Final[str] = "{prefix}{mode}skipped_{reason}_num{num:05d}.jpg"

# Optical Flow Parameters
OPTICAL_FLOW_PARAMS: Final[dict] = {
    'pyr_scale': 0.5,
    'levels': 3,
    'winsize': 15,
    'iterations': 3,
    'poly_n': 5,
    'poly_sigma': 1.2,
    'flags': 0
}
