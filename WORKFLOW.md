# Film Credit Extraction Workflow

## Overview

This document provides a detailed explanation of the FilmOCredit pipeline workflow, which automatically extracts and processes credits from video files using computer vision, OCR, and multimodal AI models.

## Architecture Components

The pipeline consists of three main processing steps:

1. **Scene Detection & Selection** - Identifies potential credit scenes
2. **Frame Analysis & Extraction** - Extracts representative frames with intelligent deduplication
3. **Multimodal LLM Processing** - Converts frames to structured credit data using Azure Vision Language Model

---

## Step 1: Scene Detection & Candidate Identification

**Module:** `scripts_v3/scene_detection.py`  
**Function:** `identify_candidate_scenes()`

### Purpose
Identify scenes in the video that likely contain credits by detecting scene boundaries and performing OCR sampling.

### Process Flow

#### 1.1 Video Analysis Setup
- Opens the video file using PySceneDetect
- Calculates total frames, FPS, and midpoint frame
- Checks for cached scene detection results to avoid reprocessing

#### 1.2 Scene Detection Methods
The pipeline supports three detection modes:

**A. Scene Count-Based (Default)**
- Analyzes the first N scenes and last M scenes
- Default: 100 scenes from start + 100 scenes from end
- Configurable via `scene_counts=(start_count, end_count)`

**B. Time-Based**
- Analyzes scenes within first X minutes and last Y minutes
- Configurable via `time_segments=(start_minutes, end_minutes)`
- Example: First 7 minutes + Last 7 minutes

**C. Whole Episode**
- Analyzes all scenes in the entire video
- Activated with `whole_episode=True`

#### 1.3 Scene Boundary Detection

Uses two PySceneDetect detectors in parallel:

**ContentDetector:**
- Threshold: 25.0 (configurable via `config.SceneDetectionConfig.CONTENT_DETECTOR_THRESHOLD`)
- Detects significant changes in frame content
- More conservative threshold helps preserve long scrolling credit sequences

**ThresholdDetector:**
- Threshold: 12.0 (configurable via `config.SceneDetectionConfig.THRESHOLD_DETECTOR_THRESHOLD`)
- Detects fade in/fade out transitions
- Higher threshold reduces false cuts during credits

#### 1.4 Scene Filtering

**Minimum Length Filter:**
- Scenes must be at least 10 frames long (`SCENE_MIN_LENGTH_FRAMES`)
- Filters out very short cuts that are unlikely to contain credits

**Temporal Tagging:**
- Each scene tagged as `first_half` or `second_half` based on position
- Uses video midpoint frame as dividing line
- Tags preserved for downstream processing

#### 1.5 OCR Sampling Strategy

For each valid scene, the pipeline samples frames at specific points:

**Sample Points (19 samples):**
```python
[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
```
- Samples distributed across the scene duration
- Each sample point represents a percentage through the scene
- Ensures comprehensive coverage without processing every frame

**OCR Processing:**
1. Extract frame at sample point
2. Convert to RGB format
3. Run OCR (PaddleOCR by default)
4. Up to 3 retry attempts with 0.5s delay between retries
5. Text normalization and filtering

**Text Validation:**
```python
# Normalization steps:
1. Strip whitespace
2. Remove user-defined stopwords (e.g., channel logos, watermarks)
3. Convert to lowercase
4. Remove extra whitespace

# Validation:
- Normalized text must be ≥ 4 characters (MIN_OCR_TEXT_LENGTH)
- If valid, scene marked as candidate and sampling stops
- Representative frame saved to step1_representative_frames/
```

#### 1.6 Output Files

**initial_scene_analysis.json:**
```json
{
  "episode_id": "episode_name",
  "status": "completed_found_candidate_scenes",
  "candidate_scenes": [
    {
      "scene_index": 0,
      "original_start_frame": 1234,
      "original_end_frame": 5678,
      "duration_frames": 4444,
      "position": "first_half",
      "segment_origin_tag": "full_video",
      "has_potential_text": true,
      "text_found_in_samples": ["DIRECTOR\nJohn Doe"],
      "representative_frames_saved": ["scene_000_first_half_frame_1500_ocr.jpg"],
      "selected": true
    }
  ],
  "raw_shots_detected": [...],
  "timestamp": "2025-10-16T10:30:00"
}
```

**raw_scenes_cache.json:**
- Caches scene detection results
- Avoids reprocessing same video
- Invalidated if video specs change

---

## Step 2: Frame Analysis & Intelligent Extraction

**Module:** `scripts_v3/frame_analysis.py`  
**Function:** `analyze_candidate_scene_frames()`

### Purpose
Extract the most informative frames from candidate scenes while eliminating duplicates and low-quality frames.

### Process Flow

#### 2.1 Scene Type Classification

Each scene is analyzed to determine its credit type:

**Optical Flow Analysis:**
```python
# Calculate vertical flow between consecutive frames
vertical_flows = []
for frame1, frame2 in consecutive_frame_pairs:
    gray1 = cv2.cvtColor(frame1, COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, ...)
    vertical_flow = np.mean(flow[:, :, 1])  # y-component
    vertical_flows.append(vertical_flow)

median_v_flow = median(vertical_flows)
```

**Classification Rules:**
- `|median_v_flow| > 0.5` pixels/frame → **Dynamic Scrolling Credits**
- `|median_v_flow| ≤ 0.5` pixels/frame → **Static/Slow Credits**

#### 2.2 Dynamic Scrolling Credits Processing

**Used for:** Rolling credits, scrolling text

**Strategy:** Sample frames at regular intervals based on scroll speed

**Calculation:**
```python
# How many pixels to scroll before saving next frame
pixels_to_scroll = frame_height * 0.9  # 90% of frame height

# Frames to skip between saves
frames_per_save = pixels_to_scroll / scroll_pixels_per_frame

# Example: 
# Frame height: 1080 pixels
# Scroll speed: 3 pixels/frame
# Pixels target: 972 pixels
# Interval: every 324 frames
```

**Frame Processing Loop:**
1. Sample frame at calculated interval
2. Check for fade frames (skip if detected)
3. Run OCR on frame
4. Normalize text and compare with previous frames
5. Apply episode-level deduplication (see below)
6. Save frame if unique, skip if duplicate

**Scrolling Mode Text Similarity:**
- Uses higher threshold: 70% (`FUZZY_TEXT_SIMILARITY_SCROLLING_THRESHOLD`)
- Only skips if text is nearly identical
- Allows capturing progression of scrolling credits

#### 2.3 Static/Slow Credits Processing

**Used for:** Static title cards, slow transitions

**Strategy:** Sample frames at fixed time intervals

**Calculation:**
```python
# Sample every 0.5 seconds (HASH_SAMPLE_INTERVAL_SECONDS)
frame_step = fps * 0.5

# Example at 25 fps:
# Sample frames: 0, 12, 24, 36, 48...
# Always include first and last frame of scene
```

**Frame Processing Loop:**
1. Sample frame at time intervals
2. Compute image hash (perceptual hash, 16x16)
3. Check for fade frames (skip if detected)
4. Compare hash with previous frame
5. If hash difference ≥ threshold, run OCR
6. Apply text deduplication
7. Save if unique

**Hash-based Visual Deduplication:**
```python
# Hash difference threshold: 0 (exact match required)
hash_diff = hamming_distance(current_hash, previous_hash)

if hash_diff < HASH_DIFFERENCE_THRESHOLD:
    # Frames too similar visually, skip
    skip_frame()
```

#### 2.4 Episode-Level Text Deduplication

**New in v3:** Cross-scene deduplication cache

**Problem Solved:**
- Same credit appearing in multiple scenes
- OCR variations of same text
- Duplicate frames from overlapping scenes

**Implementation:**

**Episode Cache:**
```python
episode_saved_texts_cache = []  # Normalized text from ALL saved frames
episode_saved_files_cache = []  # Corresponding file paths
```

**Deduplication Strategy:**

1. **Text Normalization:**
   ```python
   normalized = lowercase(remove_stopwords(strip_whitespace(text)))
   ```

2. **Fuzzy Matching:**
   ```python
   # Dynamic threshold based on text length
   def calculate_threshold(text_length):
       if text_length <= 200:
           return 60  # Base threshold
       else:
           # Increase threshold for longer texts
           extra = (text_length - 200) * 0.02
           return min(85, 60 + extra)
   
   similarity = fuzz.token_sort_ratio(current_text, last_saved_text)
   ```

3. **Length-Based Replacement:**
   ```python
   if similarity > threshold:
       if len(current_text) > len(last_saved_text):
           # Replace shorter with longer (more complete OCR)
           remove_from_cache(last_saved_text)
           remove_file(last_saved_file)
           save_current_frame()
       else:
           # Keep existing longer text
           skip_current_frame()
   ```

#### 2.5 Frame Quality Filters

**Fade Frame Detection:**
```python
def is_fade_frame(img):
    gray = cv2.cvtColor(img, COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    
    # Check for very dark or very bright frames
    if mean_brightness < 20.0 or mean_brightness > 235.0:
        # Check low contrast (fade in/out)
        if std_brightness < 10.0:
            return True
    return False
```

**OCR Text Length:**
- Minimum 4 characters after normalization
- Filters out noise, artifacts, single letters

**Visual Similarity (Static mode only):**
- Perceptual hash difference must be > 0
- Ensures visual uniqueness between frames

#### 2.6 Frame Processing Modes

**Three Processing Modes:**

1. **Single Frame:** Isolated frame, no comparison
2. **Dynamic Scroll:** Scrolling credits, interval-based sampling
3. **Static:** Time-based sampling with hash comparison

**State Management:**
```python
# Frame processing preserves state only on successful save
(metadata, new_text, new_hash, new_bbox) = process_frame(...)

if metadata:  # Frame was saved
    update_state(new_text, new_hash, new_bbox)
else:  # Frame was skipped
    preserve_previous_state()
```

#### 2.7 Output Files

**analysis_manifest.json:**
```json
{
  "scenes": {
    "scene_0": {
      "type": "dynamic_scroll",
      "median_flow_px_per_frame": 2.8,
      "save_interval_frames": 348,
      "frame_count": 12,
      "status": "completed",
      "position": "second_half",
      "output_files": [
        {
          "path": "episode/analysis/frames/scene_000_scroll_000_num01234_sim75.jpg",
          "frame_num": 1234,
          "ocr_text": "DIRECTOR\nJohn Doe",
          "scene_position": "second_half"
        }
      ]
    }
  }
}
```

**Saved Frames:**
- Location: `data/episodes/{episode_id}/analysis/frames/`
- Naming: `scene_{index}_{type}_{count}_num{frame}_sim{similarity}.jpg`
- Examples:
  - `scene_005_scroll_001_num12345_sim82.jpg` (scrolling)
  - `scene_042_static_003_num56789_sim65.jpg` (static)

**Skipped Frames:**
- Location: `data/episodes/{episode_id}/analysis/skipped_frames/`
- Naming includes skip reason
- Examples:
  - `scene_005_dynamic_skipped_fade_frame_num12340.jpg`
  - `scene_042_static_skipped_similar_hash_3_num56780.jpg`
  - `scene_010_skipped_no_text_len_2_num23456.jpg`

---

## Step 3: Multimodal LLM Processing

**Module:** `scripts_v3/azure_vlm_processing.py`  
**Function:** `run_azure_vlm_ocr_on_frames()`

### Purpose
Convert extracted frames into structured credit data using Azure's Vision Language Model (GPT-4 Vision).

### Process Flow

#### 3.1 Frame Preparation

**Input:**
- Frames from Step 2 stored in `analysis/frames/`
- Manifest file with frame metadata
- Episode context information

**Frame Selection:**
```python
# Load manifest
manifest = json.load("analysis_manifest.json")

# Get frames from each scene
for scene in manifest['scenes']:
    for frame_info in scene['output_files']:
        frame_path = frame_info['path']
        frame_num = frame_info['frame_num']
        scene_position = frame_info['scene_position']
        # Queue for VLM processing
```

**Image Encoding:**
```python
def local_image_to_data_url(image_path):
    # Read image file
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    # Base64 encode
    base64_data = base64.b64encode(image_bytes).decode('utf-8')
    
    # Create data URL
    mime_type = guess_type(image_path)[0] or 'image/jpeg'
    return f"data:{mime_type};base64,{base64_data}"
```

#### 3.2 Azure Vision Language Model (VLM) Configuration

**Model:** Azure OpenAI GPT-4 Vision (deployment configurable)

**API Settings:**
```python
{
    "model": deployment_name,  # e.g., "gpt-4-vision-preview"
    "max_tokens": 8192,        # Large for comprehensive credit extraction
    "temperature": 0.0          # Deterministic output
}
```

**Retry Strategy:**
```python
MAX_API_RETRIES = 3
BACKOFF_FACTOR = 2.0

for attempt in range(MAX_API_RETRIES):
    try:
        response = client.chat.completions.create(...)
        break
    except (RateLimitError, APIConnectionError):
        if attempt < MAX_API_RETRIES - 1:
            sleep(BACKOFF_FACTOR ** attempt)  # 1s, 2s, 4s
        else:
            raise
```

#### 3.3 Structured Prompt Engineering

**Prompt Template:**
```
You are an AI assistant specialized in extracting credits from video frames.

CONTEXT FROM PREVIOUS FRAMES:
{previous_credits_json}

TASK:
Analyze this new credit frame and extract all visible credits as structured JSON.

RULES:
1. Extract person names and company names
2. Assign appropriate role groups (see valid groups below)
3. Include detailed role information when visible
4. Maintain consistency with previously extracted credits
5. Handle variations in name formatting

VALID ROLE GROUPS:
- Direction (Director, Assistant Director)
- Writing (Writer, Screenwriter, Story)
- Production (Producer, Executive Producer)
- Cinematography (Director of Photography, Camera)
- Editing (Editor, Assistant Editor)
- Art (Production Designer, Art Director)
- Sound (Sound Designer, Sound Mixer)
- Music (Composer, Music Supervisor)
- Cast (Actor, Voice Actor)
- Crew (Various technical roles)
- Companies (Production companies, distributors)

OUTPUT FORMAT:
Return ONLY valid JSON array with this structure:
[
  {
    "name": "John Doe",
    "role_group": "Direction",
    "role_detail": "Director",
    "is_person": true
  },
  {
    "name": "XYZ Productions",
    "role_group": "Companies",
    "role_detail": "Production Company",
    "is_person": false
  }
]
```

#### 3.4 Sequential Processing with Context

**Context Accumulation:**
```python
previous_llm_output_json = "[]"  # Start with empty

for frame in frames_to_process:
    # Create prompt with previous context
    prompt = template.format(previous_credits_json=previous_llm_output_json)
    
    # Build message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # Get response
    response = azure_client.chat.completions.create(...)
    
    # Update context for next frame
    previous_llm_output_json = clean_response(response)
```

**Benefits of Context:**
- Maintains name consistency across frames
- Recognizes name variations (e.g., "J. Doe" vs "John Doe")
- Better handles partial text in scrolling credits
- Improves role group assignment accuracy

#### 3.5 Response Parsing & Validation

**JSON Cleaning:**
```python
def clean_vlm_output(raw_text):
    # Remove markdown code blocks
    text = re.sub(r'^```json\s*', '', raw_text)
    text = re.sub(r'\s*```$', '', text)
    
    # Remove any explanatory text before/after JSON
    # Find JSON array boundaries
    start = text.find('[')
    end = text.rfind(']') + 1
    
    if start >= 0 and end > start:
        return text[start:end]
    return text
```

**Parsing:**
```python
def parse_vlm_json(json_str, frame_filename):
    try:
        credits = json.loads(json_str)
        
        # Add frame metadata
        for credit in credits:
            credit['source_frame'] = [frame_filename]
            credit['original_frame_number'] = [frame_num]
            credit['scene_position'] = scene_position
        
        return credits
    except JSONDecodeError as e:
        logging.error(f"JSON parse error: {e}")
        return []
```

**Role Group Normalization:**
```python
# Role groups defined in config.py
VALID_ROLE_GROUPS = {
    "Direction", "Writing", "Production", "Cinematography",
    "Editing", "Art", "Sound", "Music", "Cast", "Crew",
    "Companies", "Unknown"
}

for credit in credits:
    role_group = credit.get('role_group')
    if role_group in VALID_ROLE_GROUPS:
        credit['role_group_normalized'] = role_group
    else:
        credit['role_group_normalized'] = "Unknown"
        logging.warning(f"Invalid role group: {role_group}")
```

#### 3.6 Deduplication & Merging

**Name-Based Deduplication:**
```python
dedup_map = {}  # key: (role_group, name)

for credit in all_credits:
    key = (credit['role_group_normalized'], credit['name'])
    
    if key in dedup_map:
        # Merge source frames
        existing = dedup_map[key]
        existing['source_frame'].extend(credit['source_frame'])
        existing['original_frame_number'].extend(credit['original_frame_number'])
    else:
        # New entry
        dedup_map[key] = credit
```

**Multi-Role Detection:**
```python
# Track all role groups for each name
name_to_roles = {}

for credit in credits:
    name = credit['name']
    role_group = credit['role_group_normalized']
    name_to_roles.setdefault(name, set()).add(role_group)

# Flag for manual review
for credit in credits:
    if len(name_to_roles[credit['name']]) > 1:
        credit['Need revisioning for deduplication'] = True
```

#### 3.7 Output Files

**{episode_id}_credits_azure_vlm.json:**
```json
[
  {
    "name": "John Doe",
    "role_group": "Direction",
    "role_group_normalized": "Direction",
    "role_detail": "Director",
    "is_person": true,
    "source_frame": [
      "scene_042_static_001_num56789_sim65.jpg",
      "scene_042_static_002_num57123_sim72.jpg"
    ],
    "original_frame_number": [56789, 57123],
    "scene_position": "second_half"
  },
  {
    "name": "Jane Smith",
    "role_group": "Writing",
    "role_group_normalized": "Writing",
    "role_detail": "Screenwriter",
    "is_person": true,
    "source_frame": ["scene_005_scroll_003_num12789_sim81.jpg"],
    "original_frame_number": [12789],
    "scene_position": "first_half"
  },
  {
    "name": "ABC Studios",
    "role_group": "Companies",
    "role_group_normalized": "Companies",
    "role_detail": "Production Company",
    "is_person": false,
    "source_frame": ["scene_001_static_000_num00456_sim90.jpg"],
    "original_frame_number": [456],
    "scene_position": "first_half"
  }
]
```

---

## Key Parameters & Configuration

### Scene Detection

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `CONTENT_DETECTOR_THRESHOLD` | 25.0 | Scene change sensitivity |
| `THRESHOLD_DETECTOR_THRESHOLD` | 12.0 | Fade detection threshold |
| `SCENE_MIN_LENGTH_FRAMES` | 10 | Minimum valid scene length |
| `DEFAULT_START_SCENES_COUNT` | 100 | Scenes from video start |
| `DEFAULT_END_SCENES_COUNT` | 100 | Scenes from video end |

### Frame Processing

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `MIN_ABS_SCROLL_FLOW_THRESHOLD` | 0.5 px/frame | Scroll detection trigger |
| `SCROLL_FRAME_HEIGHT_RATIO` | 0.9 | Scroll interval (90% of height) |
| `HASH_SAMPLE_INTERVAL_SECONDS` | 0.5 | Static frame sampling rate |
| `HASH_DIFFERENCE_THRESHOLD` | 0 | Visual similarity threshold |
| `HASH_SIZE` | 16 | Perceptual hash dimensions |

### Text Processing

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `MIN_OCR_TEXT_LENGTH` | 4 chars | Minimum valid text |
| `FUZZY_TEXT_SIMILARITY_THRESHOLD` | 60% | Static deduplication |
| `FUZZY_TEXT_SIMILARITY_SCROLLING_THRESHOLD` | 70% | Scroll deduplication |
| `FUZZY_THRESHOLD_MAX` | 85% | Max threshold for long texts |

### Quality Filters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `FADE_FRAME_THRESHOLD` | 20.0 | Brightness for fade detection |
| `FADE_FRAME_CONTRAST_THRESHOLD` | 10.0 | Contrast for fade detection |
| `MIN_OCR_CONFIDENCE` | 0.75 | OCR confidence threshold |

### Azure VLM

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `DEFAULT_VLM_MAX_NEW_TOKENS` | 8192 | Response length limit |
| `MAX_API_RETRIES` | 3 | Retry attempts |
| `BACKOFF_FACTOR` | 2.0 | Exponential backoff multiplier |

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT: Video File                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 1: Scene Detection                       │
├─────────────────────────────────────────────────────────────────┤
│  • PySceneDetect (ContentDetector + ThresholdDetector)          │
│  • Filter by time/count/whole video                             │
│  • OCR sampling (19 points per scene)                           │
│  • Text normalization & validation                              │
├─────────────────────────────────────────────────────────────────┤
│  Output: initial_scene_analysis.json                            │
│          step1_representative_frames/*.jpg                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   STEP 2: Frame Analysis                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │   Optical Flow Analysis                                 │   │
│  │   • Calculate vertical movement                         │   │
│  │   • Classify: Scrolling vs Static                       │   │
│  └────────┬──────────────────────────────┬─────────────────┘   │
│           │                               │                     │
│           ▼                               ▼                     │
│  ┌──────────────────┐          ┌──────────────────────┐       │
│  │ Dynamic Scroll   │          │ Static/Slow Credits  │       │
│  ├──────────────────┤          ├──────────────────────┤       │
│  │ • Interval-based │          │ • Time-based sampling│       │
│  │   sampling       │          │ • Hash comparison    │       │
│  │ • 90% height     │          │ • 0.5s intervals     │       │
│  │   scroll trigger │          │                      │       │
│  └────────┬─────────┘          └──────────┬───────────┘       │
│           └────────────┬───────────────────┘                   │
│                        ▼                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Episode-Level Deduplication                            │   │
│  │  • Cross-scene text comparison                          │   │
│  │  • Fuzzy matching (dynamic threshold)                   │   │
│  │  • Length-based replacement                             │   │
│  │  • Fade frame filtering                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  Output: analysis_manifest.json                                 │
│          frames/*.jpg (deduplicated)                            │
│          skipped_frames/*.jpg (debugging)                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STEP 3: Azure VLM Processing                    │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │   Sequential Frame Processing                           │   │
│  │   • Base64 encode frame                                 │   │
│  │   • Build context from previous frames                  │   │
│  │   • Send to Azure GPT-4 Vision                          │   │
│  │   • Parse structured JSON response                      │   │
│  └────────┬────────────────────────────────────────────────┘   │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │   Post-Processing                                       │   │
│  │   • Clean JSON (remove markdown)                        │   │
│  │   • Normalize role groups                               │   │
│  │   • Deduplicate by (name, role_group)                   │   │
│  │   • Merge source frames                                 │   │
│  │   • Flag multi-role credits                             │   │
│  └─────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  Output: {episode_id}_credits_azure_vlm.json                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FINAL: Structured Credits                     │
├─────────────────────────────────────────────────────────────────┤
│  • Name + Role Group + Role Detail                              │
│  • Person/Company classification                                │
│  • Source frame references                                      │
│  • Scene position metadata                                      │
│  • Deduplication flags                                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Performance Optimizations

### Caching Strategies

1. **Scene Detection Cache** (`raw_scenes_cache.json`)
   - Avoids reprocessing scene boundaries
   - Invalidated only if video specs change
   - Saves ~30-60 seconds per rerun

2. **Episode-Level Text Cache**
   - In-memory cache during Step 2
   - Prevents duplicate frames across scenes
   - Reduces Step 3 processing load by 40-60%

3. **OCR Model Caching**
   - PaddleOCR model loaded once per session
   - Reused across all scenes
   - Force refresh available if needed

### Parallel Processing Opportunities

Current implementation is sequential, but can be parallelized:

1. **Scene-Level Parallelization**
   - Process multiple scenes concurrently
   - Requires shared episode cache with locking

2. **Frame Batch Processing**
   - Send multiple frames to VLM in batch
   - Azure OpenAI supports batch endpoints

### File I/O Optimizations

1. **Reduced Frame Writes**
   - Skip frames saved to `skipped_frames/` only in debug mode
   - Production mode saves only selected frames

2. **JSON Streaming**
   - Incremental manifest updates
   - Crash recovery without full reprocessing

---

## Error Handling

### OCR Failures
- Retry up to 3 times with exponential backoff
- Log frame number and error details
- Continue processing remaining frames

### Azure API Issues
- Rate limit handling with 2x backoff
- Connection error retries
- Graceful degradation (skip problematic frames)

### File System Errors
- Windows file locking handling
- Temporary directory fallbacks
- Atomic write operations (write to temp, then rename)

---

## Quality Assurance

### Frame Selection Quality

**Metrics:**
- Text length (longer = more complete)
- OCR confidence
- Visual uniqueness (hash distance)
- Temporal distribution

### Deduplication Quality

**Metrics:**
- False positive rate (unique credits marked as duplicates)
- False negative rate (duplicate credits kept)
- Text length preservation (keep longer variants)

### VLM Output Quality

**Validation:**
- JSON schema compliance
- Role group validity
- Name format consistency
- Cross-reference with IMDB data

---

## Debugging & Troubleshooting

### Enable Debug Output

All skipped frames are saved to `skipped_frames/` with reason codes:

- `fade_frame` - Detected as fade in/out
- `no_text_len_X` - Text too short (X characters)
- `similar_hash_X` - Visual similarity (hash diff = X)
- `identical_text_scroll_X` - Duplicate in scroll mode (X% similar)
- `similar_to_last_saved_shorter_X` - Text similar but shorter (X% similar)

### Review Logs

**Log Locations:**
- Application logs: `filmocredit_pipeline.log`
- Streamlit UI: Built-in log viewer
- Console output: Real-time during processing

**Key Log Messages:**
```
[Episode] Frame XXXXX (processing_mode) OCR SUCCESS/FAILED
[Episode] Frame XXXXX (processing_mode) TEXT PROCESSING: Raw/Normalized
[Episode] Frame XXXXX (processing_mode) DECISION: SAVE/SKIP - reason
[Episode] Added text to episode cache. Cache size: N
```

### Common Issues

**Issue:** Too many frames extracted
- **Solution:** Increase `FUZZY_TEXT_SIMILARITY_THRESHOLD`
- **Solution:** Adjust `SCROLL_FRAME_HEIGHT_RATIO` (increase = fewer frames)

**Issue:** Missing credits
- **Solution:** Lower `MIN_OCR_TEXT_LENGTH`
- **Solution:** Review `skipped_frames/` for false rejections

**Issue:** Duplicate credits in final output
- **Solution:** Check VLM deduplication logic
- **Solution:** Review name normalization in Azure prompt

**Issue:** Wrong role groups assigned
- **Solution:** Refine Azure prompt with better examples
- **Solution:** Validate role group mapping in config

---

## Future Enhancements

### Potential Improvements

1. **GPU Acceleration**
   - Use GPU for optical flow calculation
   - CUDA-accelerated OCR inference

2. **Adaptive Sampling**
   - Machine learning to predict credit likelihood
   - Dynamic sample point adjustment

3. **Multi-Model Ensemble**
   - Combine multiple OCR engines
   - Consensus-based text extraction

4. **Real-Time Processing**
   - Stream processing during video capture
   - Live credit extraction

5. **Language Support**
   - Multi-language OCR models
   - Language-specific text normalization

---

## Conclusion

The FilmOCredit pipeline combines computer vision, OCR, and multimodal AI to automatically extract structured credits from video files. The three-step process ensures high accuracy while minimizing redundant processing through intelligent caching and deduplication strategies.

**Key Strengths:**
- Handles both scrolling and static credits
- Episode-level deduplication prevents duplicates
- Context-aware VLM processing improves consistency
- Comprehensive error handling and debugging support

**Typical Processing Time:**
- Step 1: 2-5 minutes per episode (scene detection + OCR sampling)
- Step 2: 5-15 minutes per episode (frame extraction + analysis)
- Step 3: 10-30 minutes per episode (VLM processing, depends on frame count)

**Total:** ~20-50 minutes per episode for fully automated credit extraction.
