"""
Domain-specific exception classes for the Filmocredit pipeline.
"""

class FilmocreditError(Exception):
    """Base exception class for the Filmocredit pipeline."""
    pass

class ConfigError(FilmocreditError):
    """Raised for configuration-related errors."""
    pass

class OCRError(FilmocreditError):
    """Raised for OCR processing errors."""
    pass

class SceneDetectionError(FilmocreditError):
    """Raised for errors during scene detection."""
    pass

class FrameAnalysisError(FilmocreditError):
    """Raised for errors during frame analysis."""
    pass

class VLMProcessingError(FilmocreditError):
    """Raised for errors during VLM processing."""
    pass
