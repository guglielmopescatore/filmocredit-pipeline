# scripts_v3 Directory Overview

This directory contains core processing scripts for the FilmOCredit pipeline.

## Scripts

### app.py

Streamlit web application entry-point. Provides UI for:
- Selecting videos
- Configuring OCR engine and parameters
- Running Step 1 (scene detection & OCR sampling)
- Running Step 2 (frame-level OCR analysis)

This module should be launched via Streamlit CLI, e.g.:  
```
streamlit run scripts_v3/app.py
```

### scene_detection.py

Identifies candidate scenes in a video by:
- Splitting into custom or default segments
- Running content and threshold detectors
- Deduplicating and filtering by scene length
- Sampling frames for quick OCR-based text presence checks
- Saving analysis to `analysis/initial_scene_analysis.json`

Public API:
```python
identify_candidate_scenes(
    video_path: Path,
    episode_id: str,
    ocr_reader: Any,
    ocr_engine_type: str,
    user_stopwords: List[str],
    custom_time_segments: Optional[List[Tuple[str, str]]] = None
) -> Tuple[List[dict], SceneDetectionStatus, Optional[str]]
```

### frame_analysis.py

Analyzes frames within each candidate scene:
- Reads frames as NumPy arrays via `scenedetect` VideoStream
- Applies OCR and normalizes text
- Computes vertical optical flow for scrolling credit detection
- Selects representative frames based on deduplication (hash/text)
- Saves final frames and returns analysis metadata

Public API:
```python
analyze_candidate_scene_frames(
    video_path: Path,
    episode_id: str,
    scene_info: dict,
    fps: float,
    frame_height: int,
    frame_width: int,
    ocr_reader: Any,
    ocr_engine_type: str,
    user_stopwords: List[str],
    global_last_saved_ocr_text_input: Optional[str],
    global_last_saved_frame_hash_input: Optional[ImageHash],
    global_last_saved_ocr_bbox_input: Optional[Tuple[int,int,int,int]]
) -> Tuple[Dict[str,Any], Optional[str], Optional[ImageHash], Optional[Tuple[int,int,int,int]]]
```

### azure_vlm_processing.py

Runs Azure OpenAI-based VLM on saved frames:
- Encodes local JPEGs to data URLs
- Sends chat-completion requests with exponential backoff
- Cleans and parses JSON output
- Appends and deduplicates credit entries
- Writes `episode_id_credits_azure_vlm.json`

Public API:
```python
run_azure_vlm_ocr_on_frames(
    episode_id: str,
    role_map: Dict[str,str],
    max_new_tokens: int
) -> Tuple[int, str, Optional[str]]
```

### utils.py

Collection of helper functions and constants:
- OCR wrappers (`run_ocr`, `ocr_with_retry`)
- Image conversions (BGR↔️RGB PIL)
- Text normalization and JSON parsing
- Database initialization and deduplication helpers
- CLAHE and rotation utilities

Public API includes:
- `run_ocr`, `ocr_with_retry`, `apply_clahe_filter`, `rotate_image`, etc.

## Usage Examples

### Launch the Streamlit App
```bash
streamlit run scripts_v3/app.py
```

### Identify Candidate Scenes
```python
from pathlib import Path
from scripts_v3.scene_detection import identify_candidate_scenes, SceneDetectionStatus
scenes, status, error = identify_candidate_scenes(
    video_path=Path('data/raw/episode.mp4'),
    episode_id='episode',
    ocr_reader=your_reader,
    ocr_engine_type='paddleocr',
    user_stopwords=['RAI', 'BBC'],
)
```

### Analyze Scene Frames
```python
from pathlib import Path
from scripts_v3.frame_analysis import analyze_candidate_scene_frames
info, last_text, last_hash, last_bbox = analyze_candidate_scene_frames(
    video_path=Path('data/raw/episode.mp4'),
    episode_id='episode',
    scene_info={'start_frame':0, 'end_frame':100, 'scene_index':1},
    fps=25.0,
    frame_height=720,
    frame_width=1280,
    ocr_reader=your_reader,
    ocr_engine_type='paddleocr',
    user_stopwords=['RAI'],
    global_last_saved_ocr_text_input=None,
    global_last_saved_frame_hash_input=None,
    global_last_saved_ocr_bbox_input=None,
)
```

### Run Azure VLM on Frames
```python
from scripts_v3.azure_vlm_processing import run_azure_vlm_ocr_on_frames
count, status, err = run_azure_vlm_ocr_on_frames(
    episode_id='episode',
    role_map={'actor': 'Actor'},
    max_new_tokens=2048,
)
```

### Utilities and Exceptions
```python
from scripts_v3.utils import run_ocr, ocr_with_retry, apply_clahe_filter, rotate_image
from scripts_v3.exceptions import FilmocreditError, OCRError
```

## Further Improvements

Refer to `TODO.md` for outstanding enhancements like unit tests, type hints, and integration tests.
