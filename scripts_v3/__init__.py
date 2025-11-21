"""
FilmOCredit Scripts Package v3.

Moduli per l'estrazione automatica di crediti da video.
"""

__version__ = "3.0.0"
__author__ = "FilmOCredit Team"

# Rendi disponibili le eccezioni a livello di package
from .exceptions import (
    FilmocreditError,
    ConfigError,
    OCRError,
    SceneDetectionError,
    FrameAnalysisError,
    VLMProcessingError
)

__all__ = [
    'config',
    'constants',
    'utils',
    'scene_detection',
    'frame_analysis',
    'vlm_processing',
    'exceptions',
    # Eccezioni
    'FilmocreditError',
    'ConfigError',
    'OCRError',
    'SceneDetectionError',
    'FrameAnalysisError',
    'VLMProcessingError'
]
