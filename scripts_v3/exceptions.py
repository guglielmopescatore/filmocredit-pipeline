"""
Domain-specific exception classes for the Filmocredit pipeline.
"""


class FilmocreditError(Exception):
    """Base exception class for the Filmocredit pipeline."""


class ConfigError(FilmocreditError):
    """Raised for configuration-related errors."""


class OCRError(FilmocreditError):
    """Raised for OCR processing errors."""


class SceneDetectionError(FilmocreditError):
    """Raised for errors during scene detection."""


class FrameAnalysisError(FilmocreditError):
    """Raised for errors during frame analysis."""


class VLMProcessingError(FilmocreditError):
    """Raised for errors during VLM processing."""
