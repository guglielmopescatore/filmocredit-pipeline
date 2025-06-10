import os
import sys
import sqlite3
import json
from pathlib import Path
import os
import sys
from pathlib import Path

from typing import Optional, Any
import logging
from scripts_v3.exceptions import ConfigError
# Disable Streamlit file watcher
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"


import streamlit as st
import logging
# For file operations during dedupe
import shutil
from pathlib import Path
from scripts_v3.exceptions import ConfigError
import sys
import json # Added for loading scene data in Step 2 fallback
from scenedetect import open_video # Ensure this is imported if used directly in app.py

try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path('.').resolve()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    # logging.info(f"Added project root to sys.path: {PROJECT_ROOT}") # Logging might not be set up

from scripts_v3 import config, utils, scene_detection, frame_analysis, azure_vlm_processing, manual_segmentation
# Helper for debugging and displaying assets
def list_dir_names(dir_path: Path) -> list[str]:
    """Return sorted list of file names in directory, or empty if not exist."""
    try:
        return sorted([p.name for p in dir_path.iterdir() if p.is_file()])
    except Exception:
        return []

from PIL import Image
from PIL import ImageStat
def display_image(path: Path, width: int = 200) -> None:
    """Display an image via PIL to avoid MediaFileHandler issues. Show debug info and fallback to bytes if needed."""
    if not path.exists():
        st.warning(f"Image not found: {path}")
        return
    try:
        img = Image.open(path)
        st.image(img.convert("RGB"), width=width, caption=f"{path.name}")
    except Exception as e:
        st.error(f"Error displaying image {path}: {e}")

        
def validate_startup() -> None:
    """Validate critical paths and environment variables at startup."""
    if not config.RAW_VIDEO_DIR.exists():
        raise ConfigError(f"Raw video directory not found: {config.RAW_VIDEO_DIR}")
    if not config.ROLE_MAP_PATH.exists():
        raise ConfigError(f"Role map file not found: {config.ROLE_MAP_PATH}")

def run_pipeline_step1(
    video_paths: list[str],
    ocr_reader: Any,
    ocr_engine: str,
    user_stopwords: list[str],
    scene_counts: tuple[int, int] | None = None
) -> dict[str, dict[str, Any]]:
    """Run scene detection (Step 1) for a batch of videos."""
    results = {}
    for video_path_str in video_paths:
        video_path_obj = Path(video_path_str)
        episode_id = video_path_obj.stem
        try:
            scenes, status, error = scene_detection.identify_candidate_scenes(
                video_path_obj,
                episode_id,
                ocr_reader,
                ocr_engine,
                user_stopwords,
                scene_counts=scene_counts
            )
            results[episode_id] = {'scenes': scenes, 'status': status, 'error': error}
        except Exception as e:
            logging.error(f"Error in pipeline Step 1 for {episode_id}: {e}", exc_info=True)
            results[episode_id] = {'scenes': [], 'status': 'error', 'error': str(e)}
    return results
 
def run_pipeline_step2(
    video_paths: list[str],
    user_selected_scenes: dict[str, list[int]],
    fps: float,
    ocr_reader: Any,
    ocr_engine: str,
    user_stopwords: list[str]
) -> dict[str, list[dict[str, Any]]]:
    """Run frame analysis (Step 2) for selected scenes."""
    results = {}
    for video_path_str in video_paths:
        episode_id = Path(video_path_str).stem
        scene_indices = user_selected_scenes.get(episode_id, [])
        results[episode_id] = []
        for idx in scene_indices:
            # Load scene info from JSON
            scene_file = config.EPISODES_BASE_DIR / episode_id / 'analysis' / 'initial_scene_analysis.json'
            with open(scene_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            scene_info = data['candidate_scenes'][idx]
            info, last_text, last_hash, last_bbox = frame_analysis.analyze_candidate_scene_frames(
                Path(video_path_str), episode_id, scene_info,
                fps, scene_info.get('frame_height', 0), scene_info.get('frame_width', 0),
                ocr_reader, ocr_engine, user_stopwords,
                None, None, None
            )
            results[episode_id].append({'scene_index': idx, 'analysis': info})
    return results

def run_pipeline_step3(
    episode_ids: list[str],
    role_map: dict[str, str],
    max_new_tokens: int
) -> dict[str, dict[str, Any]]:
    """Run Azure VLM OCR on frames (Step 3)."""
    results = {}
    for episode_id in episode_ids:
        count, status, err = azure_vlm_processing.run_azure_vlm_ocr_on_frames(
            episode_id, role_map, max_new_tokens
        )
        results[episode_id] = {'count': count, 'status': status, 'error': err}
    return results
# from scripts_v3 import azure_vlm_processing  # already imported above

if 'log_content' not in st.session_state:
    st.session_state.log_content = ""

utils.setup_logging() # Call this early
# Validate startup configurations

try:
    validate_startup()
except ConfigError as cfg_err:
    st.error(f"Configuration error: {cfg_err}")
    sys.exit(1)

openai_logger = logging.getLogger("openai")
openai_logger.setLevel(logging.INFO)

utils.init_db()

st.title("🎬 Film Credit Extraction Pipeline v3")

# Debug toggle in sidebar
with st.sidebar:
    st.session_state['debug_mode'] = st.checkbox("🔍 Debug Mode", value=st.session_state.get('debug_mode', False))

st.sidebar.header("Configuration")

st.session_state['ocr_language'] = st.sidebar.selectbox(
    "OCR Language",
    ['it', 'en', 'ch'],
    index=['it', 'en', 'ch'].index(st.session_state.get('ocr_language', 'it')),
    key='ocr_language_select_key_v3'
)

st.session_state['ocr_engine_type'] = st.sidebar.selectbox(
    "OCR Engine",
    options=config.SUPPORTED_OCR_ENGINES,
    index=config.SUPPORTED_OCR_ENGINES.index(st.session_state.get('ocr_engine_type', config.DEFAULT_OCR_ENGINE)),
    key='ocr_engine_type_select_key_v3'
)

# User-defined OCR Stopwords Management
st.sidebar.header("OCR Stopwords to Ignore")
if 'user_stopwords' not in st.session_state:
    st.session_state.user_stopwords = utils.load_user_stopwords()

edited_stopwords_text = st.sidebar.text_area(
    "Edit words (one per line):",
    value="\n".join(st.session_state.user_stopwords),  # Corrected: Use actual newline for display
    height=150,
    key="stopwords_text_area_key_v3"
)

if st.sidebar.button("Save Stopwords", key="save_stopwords_button_key_v3"):
    # Split by actual newline, then strip whitespace from each line
    updated_stopwords = [line.strip() for line in edited_stopwords_text.splitlines() if line.strip()]
    utils.save_user_stopwords(updated_stopwords)
    st.session_state.user_stopwords = updated_stopwords
    st.sidebar.success("Stopwords saved!")

# Define video_files_paths by scanning the directory
config.RAW_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
allowed_extensions = (".mp4", ".mkv", ".avi", ".mov")
try:
    video_files_paths = sorted([
        f for f in config.RAW_VIDEO_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in allowed_extensions
    ])
except FileNotFoundError:
    video_files_paths = []
    st.warning(f"Video directory {config.RAW_VIDEO_DIR} not found. Please create it and add videos.")


# Video processing portion selection
st.sidebar.header("Video Segments for Scene Detection")

# Video Preview
st.sidebar.subheader("Preview Video")
if 'selected_video_for_preview' not in st.session_state:
    st.session_state.selected_video_for_preview = None

# Use the main list of video_files_paths for the preview dropdown
if video_files_paths:
    video_options = {video_path.name: str(video_path) for video_path in video_files_paths}
    selected_video_name = st.sidebar.selectbox(
        "Select video to preview:",
        options=list(video_options.keys()),
        index=0, # Default to the first video
        key="preview_video_selector_v3"
    )
    if selected_video_name:
        st.session_state.selected_video_for_preview = video_options[selected_video_name]

if st.session_state.selected_video_for_preview:
    try:
        # Load video as raw bytes to avoid path issues
        video_path_obj = Path(st.session_state.selected_video_for_preview)
        with open(video_path_obj, 'rb') as vf:
            vid_bytes = vf.read()
        st.sidebar.video(vid_bytes, format=f"video/{video_path_obj.suffix.lstrip('.')}")
    except Exception as e:
        st.sidebar.error(f"Error loading video preview: {e}")
else:
    # Adjust message based on whether video_files_paths was populated
    if not video_files_paths:
        st.sidebar.info(f"No videos found in '{config.RAW_VIDEO_DIR}'. Add videos to enable preview.")
    else:
        st.sidebar.info("Select a video from 'data/raw' to enable preview.")



# --- Scene-based margin configuration ---
st.sidebar.subheader("Scene Detection Margins")
st.sidebar.caption(f"Specify how many scenes at the start and end of the video to consider for OCR (default: {config.DEFAULT_START_SCENES_COUNT} each). The middle scenes will be ignored.")
if 'scene_start_count' not in st.session_state:
    st.session_state['scene_start_count'] = config.DEFAULT_START_SCENES_COUNT
if 'scene_end_count' not in st.session_state:
    st.session_state['scene_end_count'] = config.DEFAULT_END_SCENES_COUNT

scene_start_count = st.sidebar.number_input(
    "Number of scenes at start:",
    min_value=1,
    max_value=200,
    value=st.session_state['scene_start_count'],
    step=1,
    key="scene_start_count_input"
)
scene_end_count = st.sidebar.number_input(
    "Number of scenes at end:",
    min_value=1,
    max_value=200,
    value=st.session_state['scene_end_count'],
    step=1,
    key="scene_end_count_input"
)
st.session_state['scene_start_count'] = scene_start_count
st.session_state['scene_end_count'] = scene_end_count

# Step 1 Options
st.subheader("Step 1 Configuration")
segmentation_method = st.radio(
    "Choose segmentation method:",
    options=["Automatic Scene Detection", "Manual Time Segments"],
    help="Automatic: Uses AI to detect scene changes. Manual: Uses specific time ranges you provide."
)

if segmentation_method == "Manual Time Segments":
    st.info("💡 **Manual Segmentation**: Specify exact time ranges to extract as clips using ffmpeg")

    manual_segments_input = st.text_area(
        "Enter time segments (one per line, format: HH:MM:SS-HH:MM:SS)",
        placeholder="00:00:30-00:02:15\n00:45:20-00:47:30\n01:23:45-01:25:10",
        help="Each line should contain a start and end time separated by a dash",
        key="manual_segments_input"
    )

    dry_run_manual = st.checkbox("Dry run (preview commands only)", key="manual_dry_run")

else:
    st.info("🤖 **Automatic Scene Detection**: Uses AI to identify potential credit scenes")

def get_cached_ocr_reader():
    selected_lang = st.session_state.get('ocr_language', 'it')
    selected_engine = st.session_state.get('ocr_engine_type', config.DEFAULT_OCR_ENGINE)
    reader_key = f"ocr_reader_{selected_engine}_{selected_lang}"
    current_reader_engine_key = "current_ocr_reader_engine_type"
    current_reader_lang_key = "current_ocr_reader_lang"

    if (reader_key not in st.session_state or
            st.session_state.get(reader_key) is None or
            st.session_state.get(current_reader_engine_key) != selected_engine or
            st.session_state.get(current_reader_lang_key) != selected_lang):

        logging.info(f"Initializing OCR reader: Engine={selected_engine}, Lang={selected_lang}")
        try:
            if selected_engine == "paddleocr":
                # Map to specific language codes PaddleOCR expects from PADDLEOCR_LANG_MAP
                paddle_lang_code = config.PADDLEOCR_LANG_MAP.get(selected_lang, selected_lang)
                st.session_state[reader_key] = utils.get_paddleocr_reader(lang=paddle_lang_code)
            elif selected_engine == "easyocr":
                # Only check CUDA availability when we actually need to initialize a reader
                use_gpu = config.is_cuda_available()
                # EasyOCR expects a list of language codes, get from EASYOCR_LANG_MAP
                easyocr_lang_codes = config.EASYOCR_LANG_MAP.get(selected_lang, [selected_lang])
                st.session_state[reader_key] = utils.get_easyocr_reader(lang=easyocr_lang_codes[0], use_gpu=use_gpu) # Pass first lang for now, or handle list
            else:
                st.session_state[reader_key] = None
                logging.error(f"Unsupported OCR engine: {selected_engine}")
            st.session_state[current_reader_engine_key] = selected_engine
            st.session_state[current_reader_lang_key] = selected_lang
            if st.session_state.get(reader_key): # Check if reader was successfully set
                logging.info(f"OCR Reader for {selected_engine} ({selected_lang}) initialized successfully.")
            else:
                logging.error(f"Failed to initialize OCR Reader for {selected_engine} ({selected_lang}). Reader is None.")
        except Exception as e:
            st.session_state[reader_key] = None # Ensure it's None on failure
            logging.error(f"Exception initializing OCR reader {selected_engine} ({selected_lang}): {e}", exc_info=True)
            st.error(f"Error initializing OCR for {selected_engine} ({selected_lang}): {e}")
    
    utils.setup_logging()
    
    return st.session_state.get(reader_key)


tab1, tab2, tab3 = st.tabs(["⚙️ Setup & Run Pipeline", "✏️ Review & Edit Credits", "📊 Logs"])

with tab1:
    st.header("1. Select Videos for Processing")
    selected_videos_str_paths = []
    if not video_files_paths:
        st.warning(f"No video files found in {config.RAW_VIDEO_DIR}. Please add videos with extensions: {', '.join(allowed_extensions)}")
    else:
        st.write("Select videos from `data/raw` to include in the batch:")
        for video_file_path_obj in video_files_paths:
            if st.checkbox(video_file_path_obj.name, key=f"vid_select_{video_file_path_obj.name}_v3"):
                selected_videos_str_paths.append(str(video_file_path_obj))

    if not selected_videos_str_paths:
        st.info("No videos selected for processing.")
    else:
        st.info(f"Selected {len(selected_videos_str_paths)} video(s) for processing.")

    st.header("2. Run Processing Steps")

    if 'episode_status' not in st.session_state:
        st.session_state.episode_status = {}

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        run_step1_button = st.button("STEP 1: Identify Candidate Scenes", key="run_step1_btn_v3")
    with col2:
        run_step2_button = st.button("STEP 2: Analyze Scene Frames", key="run_step2_btn_v3")
    with col3:
        run_step3_button = st.button("STEP 3: Azure VLM OCR", key="run_step3_btn_v3")
    with col4:
        run_all_steps_button = st.button("RUN ALL STEPS", key="run_all_steps_btn_v3")

    # Ensure user selection state exists before rendering review UI
    if 'user_selected_scenes_for_step2' not in st.session_state:
        st.session_state.user_selected_scenes_for_step2 = {}
    # --- Moved Scene Review UI Block ---
    if selected_videos_str_paths: # Only show if videos are selected in the UI
        st.subheader("2a. Review Candidate Scenes for Step 2")
        any_review_ui_shown_for_an_episode = False # Flag to see if any review content was actually displayed
        
        for video_path_str_review in selected_videos_str_paths:
            episode_id_review = Path(video_path_str_review).stem
            scene_analysis_file_review = config.EPISODES_BASE_DIR / episode_id_review / "analysis" / "initial_scene_analysis.json"
            step1_frames_base_dir_review = config.EPISODES_BASE_DIR / episode_id_review / "analysis" / "step1_representative_frames"

            if scene_analysis_file_review.exists():
                try:
                    with open(scene_analysis_file_review, 'r', encoding='utf-8') as f_review:
                        step1_output = json.load(f_review)
                    
                    candidate_scenes_from_step1 = step1_output.get("candidate_scenes", [])
                    
                    if not candidate_scenes_from_step1:
                        st.info(f"No candidate scenes found by Step 1 for {episode_id_review} to review.")
                        st.session_state.user_selected_scenes_for_step2[episode_id_review] = [] 
                        continue

                    any_review_ui_shown_for_an_episode = True
                    with st.expander(f"Review scenes for {episode_id_review} (Found {len(candidate_scenes_from_step1)} candidates)", expanded=True):
                        # Initialize session state for this episode's selections if not already present
                        if episode_id_review not in st.session_state.user_selected_scenes_for_step2:
                            initial_selected_indices = [
                                i for i, scene in enumerate(candidate_scenes_from_step1)
                                if scene.get("selected", True)  # Default to True if key somehow missing
                            ]
                            st.session_state.user_selected_scenes_for_step2[episode_id_review] = initial_selected_indices
                        
                        # Debug info about paths
                        if not step1_frames_base_dir_review.exists():
                            st.warning(f"Directory does not exist: {step1_frames_base_dir_review}")
                        
                        # This is the selection state that checkboxes should reflect (from session)
                        selected_indices_for_episode = st.session_state.user_selected_scenes_for_step2.get(episode_id_review, [])
                        
                        cols_review = st.columns(3) 
                        current_col_idx_review = 0
                        newly_selected_indices = [] # Collect what the UI renders as selected in this pass

                        for idx, scene_info_review in enumerate(candidate_scenes_from_step1):
                            with cols_review[current_col_idx_review]:
                                scene_label = f"Scene {scene_info_review.get('scene_index', idx)} ({scene_info_review.get('position', 'N/A')})"
                                is_selected_in_ui = st.checkbox(
                                    scene_label, 
                                    value=(idx in selected_indices_for_episode), # Use session state here
                                    key=f"scene_select_{episode_id_review}_{idx}"
                                )
                                if is_selected_in_ui:
                                    newly_selected_indices.append(idx)

                                st.caption(f"Frames: {scene_info_review.get('original_start_frame')} - {scene_info_review.get('original_end_frame')}")
                                if scene_info_review.get("text_found_in_samples"):
                                    ocr_sample_text = scene_info_review.get('text_found_in_samples')[0][:50]
                                    st.caption(f'OCR Sample: "{ocr_sample_text}..."')

                                representative_frames = scene_info_review.get("representative_frames_saved", [])
                                if representative_frames:
                                    img_filename = representative_frames[0]
                                    img_path = step1_frames_base_dir_review / img_filename
                                    # Use the new display_image helper for debug and display
                                    if img_path.exists():
                                        display_image(img_path, width=200)
                                    else:
                                        st.caption(f"Representative frame not found at {img_path}")
                                        # Try alternate location under analysis/step1_representative_frames
                                        alt_path = config.EPISODES_BASE_DIR / episode_id_review / "analysis" / "step1_representative_frames" / img_filename
                                        if alt_path.exists():
                                            display_image(alt_path, width=200)
                                        else:
                                            st.caption(f"Alternate image also not found: {alt_path}")
                                else:
                                    st.caption("No representative frame saved.")
                                st.markdown("---")
                            current_col_idx_review = (current_col_idx_review + 1) % 3
                        
                        # Check if the user's interaction changed the selection state compared to what was in session
                        if set(st.session_state.user_selected_scenes_for_step2.get(episode_id_review, [])) != set(newly_selected_indices):
                            st.session_state.user_selected_scenes_for_step2[episode_id_review] = newly_selected_indices
                            logging.info(f"User updated scene selection for {episode_id_review}. New selection: {newly_selected_indices}")

                            # Persist this new selection state to the JSON file
                            # step1_output is the dict loaded from JSON earlier (contains all keys like episode_id, status, etc.)
                            # candidate_scenes_from_step1 is the list of scenes from that dict (original state from file)
                            
                            updated_candidate_scenes_for_json_save = []
                            for i, scene_data_original in enumerate(candidate_scenes_from_step1):
                                scene_copy = scene_data_original.copy()
                                scene_copy["selected"] = (i in newly_selected_indices) # Use the latest UI selection
                                updated_candidate_scenes_for_json_save.append(scene_copy)
                            
                            # Prepare the full data structure to save by updating the candidate_scenes list in the loaded data
                            data_to_save_to_json = step1_output.copy() 
                            data_to_save_to_json["candidate_scenes"] = updated_candidate_scenes_for_json_save
                            
                            try:
                                with open(scene_analysis_file_review, 'w', encoding='utf-8') as f_save:
                                    json.dump(data_to_save_to_json, f_save, indent=4)
                                logging.info(f"Saved updated scene selections to {scene_analysis_file_review.name} for {episode_id_review}")
                            except Exception as e_save_json:
                                logging.error(f"Error saving scene selections to JSON for {episode_id_review}: {e_save_json}")
                                st.error(f"Failed to save selection changes for {episode_id_review}.")
                        
                        # Display current selection count based on the latest session state
                        current_selected_count_for_info = len(st.session_state.user_selected_scenes_for_step2.get(episode_id_review, []))
                        st.info(f"User has selected {current_selected_count_for_info} scenes for {episode_id_review} for Step 2 processing.")

                except Exception as e_review: # Added except block
                    st.error(f"Error loading or displaying scenes for review ({episode_id_review}): {e_review}")
                    logging.error(f"Error during scene review display for {episode_id_review}: {e_review}", exc_info=True)
            else:
                st.warning(f"Step 1 output file not found for {episode_id_review}. Run Step 1 to generate candidate scenes for review.")
        
        if any_review_ui_shown_for_an_episode:
            st.markdown("---") 
            st.success("Scene selections updated. Ready for Step 2 if desired.")
        elif selected_videos_str_paths: 
            st.info("Run Step 1 for selected videos to generate scenes for review.")
    # --- End of Moved Scene Review UI Block ---

# Button processing logic should be outside 'with tab1:' to correctly use Streamlit's flow
# This matches the original app.py structure where these if blocks are at the end.

if 'user_selected_scenes_for_step2' not in st.session_state:
    st.session_state.user_selected_scenes_for_step2 = {}

if run_step1_button:
    if not selected_videos_str_paths:
        st.warning("Please select at least one video for Step 1.")
    else:
        if segmentation_method == "Manual Time Segments":
            # Manual segmentation workflow
            st.subheader("Running Step 1: Manual Video Segmentation")
            
            if not manual_segments_input.strip():
                st.error("Please enter time segments for manual segmentation.")
            else:
                # Parse manual segments
                segments = manual_segmentation.parse_segments_string(manual_segments_input)
                if not segments:
                    st.error("No valid segments found. Please check the format (HH:MM:SS-HH:MM:SS)")
                else:
                    st.info(f"Found {len(segments)} valid segments to process")
                    
                    for video_path_str_proc in selected_videos_str_paths:
                        video_path_obj = Path(video_path_str_proc)
                        episode_id_proc = video_path_obj.stem
                        
                        if episode_id_proc not in st.session_state.episode_status:
                            st.session_state.episode_status[episode_id_proc] = {}
                        
                        st.session_state.episode_status[episode_id_proc]['step1_status'] = "running"
                        
                        with st.expander(f"Manual Segmentation: {episode_id_proc}", expanded=True):
                            st.write(f"Processing manual segmentation for {episode_id_proc}...")
                            with st.spinner(f"Segmenting video {episode_id_proc}..."):
                                try:
                                    output_clips = manual_segmentation.segment_video_manually(
                                        input_path=str(video_path_obj),
                                        segments=segments,
                                        episode_id=episode_id_proc,
                                        dry_run=dry_run_manual
                                    )
                                    
                                    st.session_state.episode_status[episode_id_proc]['step1_status'] = "completed"
                                    
                                    if dry_run_manual:
                                        st.info(f"DRY RUN: Would create {len(output_clips)} video clips")
                                        for clip in output_clips:
                                            st.text(f"  → {clip}")
                                    else:
                                        st.success(f"Successfully created {len(output_clips)} video clips")
                                        for clip in output_clips:
                                            if os.path.exists(clip):
                                                size_mb = os.path.getsize(clip) / (1024*1024)
                                                st.text(f"  → {os.path.basename(clip)} ({size_mb:.2f} MB)")
                                    
                                except Exception as e:
                                    st.session_state.episode_status[episode_id_proc]['step1_status'] = "error"
                                    st.session_state.episode_status[episode_id_proc]['step1_error'] = str(e)
                                    st.error(f"Error in manual segmentation ({episode_id_proc}): {e}")
                                    logging.error(f"Exception during manual segmentation for {episode_id_proc}: {e}", exc_info=True)
                    
                    if not dry_run_manual:
                        st.success("Manual segmentation completed! Video clips have been created and are ready for further processing.")
                    else:
                        st.info("Dry run completed. Remove 'Dry run' checkbox and run again to create actual clips.")
        
        else:
            # Automatic scene detection workflow (original)
            st.subheader("Running Step 1: Automatic Scene Detection")
            ocr_reader = get_cached_ocr_reader()
            current_ocr_engine = st.session_state.get('ocr_engine_type', config.DEFAULT_OCR_ENGINE)
            user_stopwords = st.session_state.get('user_stopwords', [])
            
            custom_segments_text = st.session_state.get('video_custom_time_segments_text', "")
            parsed_segments = []
            if custom_segments_text.strip():
                for line in custom_segments_text.strip().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('-')
                    if len(parts) == 2:
                        start_tc_str, end_tc_str = parts[0].strip(), parts[1].strip()
                        # Basic validation for HH:MM:SS or MM:SS format (can be improved with utils.is_valid_timecode_format)
                        if utils.is_valid_timecode_format(start_tc_str) and utils.is_valid_timecode_format(end_tc_str):
                            parsed_segments.append((start_tc_str, end_tc_str))
                        else:
                            st.warning(f"Invalid timecode format in segment: '{line}'. Please use HH:MM:SS-HH:MM:SS or MM:SS-MM:SS. This line will be skipped.")
                    else:
                        st.warning(f"Invalid segment format: '{line}'. Please use HH:MM:SS-HH:MM:SS or MM:SS-MM:SS. This line will be skipped.")
                
                if not parsed_segments and custom_segments_text.strip(): # If text was there but nothing parsed
                     st.warning("No valid time segments were parsed. Processing entire video(s) by default.")
                elif parsed_segments:
                    st.info(f"Using {len(parsed_segments)} custom time segment(s) for scene detection.")
            else:
                st.info("No custom time segments specified. Processing entire video(s) by default.")


            if ocr_reader is None:
                st.error(f"Cannot run Step 1: {current_ocr_engine.upper()} reader failed to initialize. Check logs.")
                logging.error(f"OCR Reader ({current_ocr_engine}) is None. Aborting Step 1.")
            else:
                for video_path_str_proc in selected_videos_str_paths:
                    video_path_obj = Path(video_path_str_proc)
                    episode_id_proc = video_path_obj.stem
                    if episode_id_proc not in st.session_state.episode_status:
                        st.session_state.episode_status[episode_id_proc] = {}
                    
                    st.session_state.episode_status[episode_id_proc]['step1_status'] = "running"
                    with st.expander(f"Step 1: {episode_id_proc}", expanded=True):
                        st.write(f"Processing Step 1 for {episode_id_proc}...")
                        with st.spinner(f"Identifying scenes for {episode_id_proc}..."):
                            try:
                                scenes, status, error_msg = scene_detection.identify_candidate_scenes(
                                    video_path_obj, 
                                    episode_id_proc, 
                                    ocr_reader, 
                                    current_ocr_engine, 
                                    user_stopwords,
                                    scene_counts=(scene_start_count, scene_end_count)
                                )
                                st.session_state.episode_status[episode_id_proc]['step1_status'] = status
                                if error_msg:
                                    st.session_state.episode_status[episode_id_proc]['step1_error'] = error_msg
                                    st.error(f"Error in Step 1 ({episode_id_proc}): {error_msg}")
                                else:
                                    st.success(f"Step 1 ({episode_id_proc}) -> Status: {status}. Found {len(scenes)} candidate scenes.")
                                logging.info(f"[{episode_id_proc}] Step 1 status: {status}. Scenes: {len(scenes)}. Error: {error_msg}")
                            except Exception as e:
                                st.session_state.episode_status[episode_id_proc]['step1_status'] = "error"
                                st.session_state.episode_status[episode_id_proc]['step1_error'] = str(e)
                                st.error(f"Exception in Step 1 ({episode_id_proc}): {e}")
                                logging.error(f"Exception during Step 1 for {episode_id_proc}: {e}", exc_info=True)
                st.success("Step 1 processing finished for selected videos. Review candidate scenes in '2a. Review Candidate Scenes' section if available.") # MODIFIED Message
        # REMOVED Scene Review UI from here

if run_step2_button:
    if not selected_videos_str_paths:
        st.warning("Please select at least one video for Step 2.")
    else:
        st.subheader("Running Step 2: Analyze Candidate Scene Frames")
        ocr_reader = get_cached_ocr_reader()
        current_ocr_engine = st.session_state.get('ocr_engine_type', config.DEFAULT_OCR_ENGINE)
        user_stopwords = st.session_state.get('user_stopwords', [])

        if ocr_reader is None:
            st.error(f"Cannot run Step 2: {current_ocr_engine.upper()} reader failed to initialize. Check logs.")
            logging.error(f"OCR Reader ({current_ocr_engine}) is None. Aborting Step 2.")
        else:
            for video_path_str_proc in selected_videos_str_paths:
                video_path_obj = Path(video_path_str_proc)
                episode_id_proc = video_path_obj.stem
                if episode_id_proc not in st.session_state.episode_status:
                    st.session_state.episode_status[episode_id_proc] = {}

                scene_analysis_file = config.EPISODES_BASE_DIR / episode_id_proc / "analysis" / "initial_scene_analysis.json"
                if not scene_analysis_file.exists():
                    st.warning(f"Step 1 output (initial_scene_analysis.json) not found for {episode_id_proc}. Please run Step 1 first.")
                    st.session_state.episode_status[episode_id_proc]['step2_status'] = "skipped_no_step1_output"
                    continue

                # Get user-selected scenes for this episode
                user_selected_indices_for_ep = st.session_state.user_selected_scenes_for_step2.get(episode_id_proc)
                
                if user_selected_indices_for_ep is None: # Should have been initialized by review step
                    st.warning(f"Scene selection for {episode_id_proc} not found. Please run Step 1 and review scenes first, or ensure review step completed.")
                    # Default to processing all if not reviewed, or skip. For now, let's try to load all.
                    logging.warning(f"[{episode_id_proc}] User selected scenes not found in session state. Attempting to load all from Step 1 output for Step 2.")
                    try:
                        with open(scene_analysis_file, 'r', encoding='utf-8') as f_scenes_fallback:
                            step1_data_fallback = json.load(f_scenes_fallback)
                        all_candidate_scenes_from_file = step1_data_fallback.get("candidate_scenes", [])
                        scenes_to_process_for_step2 = all_candidate_scenes_from_file
                    except Exception as e_fallback_load:
                        st.error(f"Failed to load scenes for {episode_id_proc} as fallback: {e_fallback_load}")
                        scenes_to_process_for_step2 = []
                elif not user_selected_indices_for_ep: # User reviewed and deselected all
                    st.info(f"No scenes selected by user for {episode_id_proc} for Step 2 processing. Skipping.")
                    st.session_state.episode_status[episode_id_proc]['step2_status'] = "skipped_no_user_selection"
                    scenes_to_process_for_step2 = []
                else: # User selected specific scenes
                    try:
                        with open(scene_analysis_file, 'r', encoding='utf-8') as f_scenes_main:
                            step1_data_main = json.load(f_scenes_main)
                        all_candidate_scenes_from_file = step1_data_main.get("candidate_scenes", [])
                        scenes_to_process_for_step2 = [all_candidate_scenes_from_file[i] for i in user_selected_indices_for_ep if i < len(all_candidate_scenes_from_file)]
                        st.info(f"Processing {len(scenes_to_process_for_step2)} user-selected scenes for {episode_id_proc} in Step 2.")
                    except Exception as e_load_selected:
                        st.error(f"Error loading selected scenes for {episode_id_proc}: {e_load_selected}")
                        scenes_to_process_for_step2 = []


                if not scenes_to_process_for_step2:
                    if st.session_state.episode_status[episode_id_proc].get('step2_status') not in ["skipped_no_user_selection", "skipped_no_step1_output"] : # Avoid overwriting specific skip reasons
                         st.info(f"No scenes to process for {episode_id_proc} in Step 2 after selection/loading.")
                         st.session_state.episode_status[episode_id_proc]['step2_status'] = "skipped_no_scenes_to_process"
                    continue # Move to next video if no scenes for this one

                st.session_state.episode_status[episode_id_proc]['step2_status'] = "running"
                with st.expander(f"Step 2: {episode_id_proc}", expanded=True):
                    st.write(f"Processing Step 2 for {episode_id_proc} (using {len(scenes_to_process_for_step2)} selected/loaded scenes)...")
                    with st.spinner(f"Analyzing frames for {episode_id_proc}..."):
                        final_status_msg_step2 = "unknown"
                        final_error_detail_step2 = None
                        try:
                            # The manual loop is the primary way now, using scenes_to_process_for_step2
                            # Remove or adapt the hasattr(frame_analysis, "analyze_frames_for_episode") block if it's not designed for selective scene input.
                            # For now, assuming the manual loop is the target.

                            logging.info(f"[{episode_id_proc}] Using manual scene loop for Step 2 with {len(scenes_to_process_for_step2)} scenes.")
                            
                            if not scenes_to_process_for_step2: # Double check after selection logic
                                st.info(f"No candidate scenes available for {episode_id_proc} to process in Step 2.")
                                final_status_msg_step2 = "skipped_no_scenes_after_selection"
                            else:
                                video_stream = open_video(str(video_path_obj))
                                fps = video_stream.frame_rate # Corrected attribute
                                frame_height = video_stream.frame_size[1] # Corrected from video_stream.resolution[1]
                                frame_width = video_stream.frame_size[0] # Corrected from video_stream.resolution[0]
                                  # Initialize episode-level saved text cache for this episode
                                episode_saved_texts_cache = []
                                episode_saved_files_cache = []  # Track file paths for cleanup when replacing
                                logging.info(f"[{episode_id_proc}] Initialized episode-level caches for Step 2")
                                
                                # Maintain list of all OCR texts found during this episode to dedupe across scenes
                                ep_global_last_saved_ocr_text = None
                                ep_global_last_saved_frame_hash = None
                                ep_global_last_saved_ocr_bbox = None
                                
                                num_scenes_processed_step2 = 0
                                for scene_info_item_step2 in scenes_to_process_for_step2: # Use the filtered list
                                    logging.info(f"Analyzing scene {scene_info_item_step2.get('scene_index', 'N/A')} for {episode_id_proc} in Step 2")
                                    if not all(k in scene_info_item_step2 for k in ['original_start_frame', 'original_end_frame']):
                                        logging.warning(f"Skipping scene in Step 2 due to missing keys: {scene_info_item_step2}")
                                        continue

                                    # Adapt analyze_candidate_scene_frames
                                    compatible_scene_info = {
                                        **scene_info_item_step2,
                                        'start_frame': scene_info_item_step2['original_start_frame'],
                                        'end_frame': scene_info_item_step2['original_end_frame']
                                    }

                                    # Analyze scene frames and get raw results WITH episode cache
                                    analysis_result, ep_global_last_saved_ocr_text, ep_global_last_saved_frame_hash, ep_global_last_saved_ocr_bbox = (
                                        frame_analysis.analyze_candidate_scene_frames(
                                            video_path=video_path_obj,
                                            episode_id=episode_id_proc,
                                            scene_info=compatible_scene_info,
                                            fps=fps,
                                            frame_height=frame_height, frame_width=frame_width,
                                            ocr_reader=ocr_reader, ocr_engine_type=current_ocr_engine,
                                            user_stopwords=user_stopwords,
                                            global_last_saved_ocr_text_input=ep_global_last_saved_ocr_text,
                                            global_last_saved_frame_hash_input=ep_global_last_saved_frame_hash,
                                            global_last_saved_ocr_bbox_input=ep_global_last_saved_ocr_bbox,
                                            episode_saved_texts_cache=episode_saved_texts_cache,  # Pass shared cache
                                            episode_saved_files_cache=episode_saved_files_cache   # Pass shared files cache
                                        )
                                    )
                                    
                                    # NOTE: The episode_saved_texts_cache is modified in place by _process_and_save_frame
                                    # so it accumulates saved texts across all scenes in this episode
                                    logging.info(f"[{episode_id_proc}] Episode cache now contains {len(episode_saved_texts_cache)} saved texts after scene {scene_info_item_step2.get('scene_index', 'N/A')}")
                                    
                                    # Remove the manual deduplication logic since it's now handled by episode cache
                                    # Simply use the results as-is from analyze_candidate_scene_frames
                                    
                                    # Process/save analysis_result, update overall manifest for Step 2
                                    try:
                                        # Prepare manifest file path
                                        analysis_dir = config.EPISODES_BASE_DIR / episode_id_proc / 'analysis'
                                        analysis_dir.mkdir(parents=True, exist_ok=True)
                                        manifest_path = analysis_dir / 'analysis_manifest.json'
                                        # Load existing manifest or start new
                                        if manifest_path.is_file():
                                            with open(manifest_path, 'r', encoding='utf-8') as mf:
                                                manifest_data = json.load(mf)
                                        else:
                                            manifest_data = {'scenes': {}}
                                    except Exception as mf_err:
                                        logging.error(f"[{episode_id_proc}] Failed to load Step 2 manifest: {mf_err}", exc_info=True)
                                        manifest_data = {'scenes': {}}
                                    # Update scene entry
                                    scene_index = scene_info_item_step2.get('scene_index')
                                    scene_key = f"scene_{scene_index}" if scene_index is not None else f"scene_{episode_id_proc}_{num_scenes_processed_step2}"
                                    # Store analysis result for this scene
                                    manifest_data.setdefault('scenes', {})[scene_key] = analysis_result
                                    # Save manifest back to disk
                                    try:
                                        with open(manifest_path, 'w', encoding='utf-8') as mf:
                                            json.dump(manifest_data, mf, indent=2)
                                        logging.debug(f"[{episode_id_proc}] Updated Step 2 manifest with scene {scene_key}")
                                    except Exception as write_err:
                                        logging.error(f"[{episode_id_proc}] Failed to write Step 2 manifest: {write_err}", exc_info=True)
                                    num_scenes_processed_step2 += 1

                                st.success(f"Step 2 ({episode_id_proc}) completed. Processed {num_scenes_processed_step2} scenes with episode-level deduplication.")
                                logging.info(f"[{episode_id_proc}] Step 2 completed with final episode cache containing {len(episode_saved_texts_cache)} unique texts")
                                
                            st.session_state.episode_status[episode_id_proc]['step2_status'] = final_status_msg_step2
                            if final_error_detail_step2:
                                st.session_state.episode_status[episode_id_proc]['step2_error'] = final_error_detail_step2
                                st.warning(f"Step 2 ({episode_id_proc}): {final_error_detail_step2}")

                        except Exception as e:
                            st.session_state.episode_status[episode_id_proc]['step2_status'] = "error"
                            st.session_state.episode_status[episode_id_proc]['step2_error'] = str(e)
                            st.error(f"Exception in Step 2 ({episode_id_proc}): {e}")
                            logging.error(f"Exception during Step 2 for {episode_id_proc}: {e}", exc_info=True)
        st.info("Step 2 processing finished for selected videos.")


if run_step3_button:
    if not selected_videos_str_paths:
        st.warning("Please select at least one video for Step 3.")
    else:
        st.subheader("Running Step 3: Azure VLM OCR")
        # user_stopwords = st.session_state.get('user_stopwords', []) # Not directly passed to VLM usually
        for video_path_str_proc in selected_videos_str_paths:
            video_path_obj = Path(video_path_str_proc)
            episode_id_proc = video_path_obj.stem
            
            # Initialize episode status if not exists
            if episode_id_proc not in st.session_state.episode_status:
                st.session_state.episode_status[episode_id_proc] = {}
            
            # Basic check: Step 2 should ideally produce frames in 'analysis/frames'
            frames_dir_for_vlm = config.EPISODES_BASE_DIR / episode_id_proc / "analysis" / "frames"
            if not frames_dir_for_vlm.is_dir() or not any(frames_dir_for_vlm.iterdir()):
                 st.warning(f"No frames found from Step 2 for {episode_id_proc} in {frames_dir_for_vlm}. VLM step might have no input.")
            
            with st.expander(f"Step 3: {episode_id_proc}", expanded=True):
                st.write(f"Processing Step 3 for {episode_id_proc}...")
                with st.spinner(f"Running Azure VLM for {episode_id_proc}..."):
                    try:
                        role_map_data = utils.load_role_map(config.ROLE_MAP_PATH)
                        count, status, err_msg = azure_vlm_processing.run_azure_vlm_ocr_on_frames(
                            episode_id_proc,
                            role_map_data,
                            config.DEFAULT_VLM_MAX_NEW_TOKENS
                        )
                        
                        # If VLM processing completed successfully, save results to database
                        if status == "completed" and not err_msg:
                            try:
                                # Load the VLM results JSON file
                                ocr_dir = config.EPISODES_BASE_DIR / episode_id_proc / "ocr"
                                vlm_json_path = ocr_dir / f"{episode_id_proc}_credits_azure_vlm.json"
                                
                                if vlm_json_path.exists():
                                    vlm_credits = utils.load_vlm_results_from_jsonl(vlm_json_path)
                                    if vlm_credits:
                                        # Save to database
                                        success, db_msg = utils.save_credits(episode_id_proc, vlm_credits)
                                        if success:
                                            logging.info(f"[{episode_id_proc}] Successfully saved {len(vlm_credits)} credits to database.")
                                        else:
                                            logging.error(f"[{episode_id_proc}] Failed to save credits to database: {db_msg}")
                                            st.warning(f"VLM completed but database save failed: {db_msg}")
                                    else:
                                        logging.warning(f"[{episode_id_proc}] No credits found in VLM results file.")
                                else:
                                    logging.warning(f"[{episode_id_proc}] VLM results file not found: {vlm_json_path}")
                            except Exception as db_save_err:
                                logging.error(f"[{episode_id_proc}] Error saving VLM results to database: {db_save_err}", exc_info=True)
                                st.warning(f"VLM completed but database save failed: {db_save_err}")
                        
                        # Update status in st.session_state.episode_status[episode_id_proc]
                        st.session_state.episode_status[episode_id_proc]['step3_status'] = status
                        if err_msg:
                            st.session_state.episode_status[episode_id_proc]['step3_error'] = err_msg
                            st.error(f"Error in Step 3 ({episode_id_proc}): {err_msg}")
                        else:
                            st.success(f"Step 3 ({episode_id_proc}) -> Status: {status}. New credits: {count}.")
                    except Exception as e:
                        st.session_state.episode_status[episode_id_proc]['step3_status'] = "error"
                        st.session_state.episode_status[episode_id_proc]['step3_error'] = str(e)
                        st.error(f"Exception in Step 3 ({episode_id_proc}): {e}")
                        logging.error(f"Exception during Step 3 for {episode_id_proc}: {e}", exc_info=True)
        st.info("Step 3 processing finished for selected videos.")


if run_all_steps_button:
    st.warning("RUN ALL STEPS functionality needs to be implemented by calling Step 1, 2, and 3 in sequence for each selected video, with appropriate checks for success before proceeding to the next step. This is a complex workflow to manage in Streamlit's execution model and is left as a manual process for now (run steps individually).")

with tab2:
    st.header("✏️ Review & Edit Credits")
    
    # Load available episodes from the database
    try:
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        
        # Get list of episodes that have credits
        cursor.execute(f"SELECT DISTINCT episode_id FROM {config.DB_TABLE_CREDITS} ORDER BY episode_id")
        available_episodes = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if not available_episodes:
            st.info("No episodes with credits found in the database. Please run the pipeline first to generate credits.")
        else:
            # Episode selection
            selected_episode = st.selectbox(
                "Select an episode to review:",
                options=available_episodes,
                key="episode_selector_review"
            )
            
            if selected_episode:
                st.subheader(f"Credits for: {selected_episode}")
                  # Load credits for selected episode
                conn = sqlite3.connect(config.DB_PATH)
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT role_group_normalized, name, role_detail, 
                           source_frame, original_frame_number, scene_position
                    FROM {config.DB_TABLE_CREDITS} 
                    WHERE episode_id = ? 
                    ORDER BY role_group_normalized, name
                """, (selected_episode,))
                
                credits_data = cursor.fetchall()
                conn.close()
                
                if not credits_data:
                    st.warning(f"No credits found for episode {selected_episode}")
                else:
                    # Create a DataFrame for easier editing
                    import pandas as pd
                    
                    # Process the data to handle comma-separated values
                    processed_data = []
                    for row in credits_data:
                        role_group, name, role_detail, source_frame, frame_numbers, scene_pos = row
                        # Convert comma-separated strings back to readable format
                        source_frames_display = source_frame if source_frame else "N/A"
                        frame_numbers_display = frame_numbers if frame_numbers else "N/A"
                        
                        processed_data.append([
                            role_group, name, role_detail, 
                            source_frames_display, frame_numbers_display, scene_pos
                        ])
                    
                    df = pd.DataFrame(processed_data, columns=[
                        'Role Group', 'Name', 'Role Detail', 
                        'Source Frames', 'Frame Numbers', 'Scene Position'
                    ])                    # Display summary statistics
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Total Credits", len(df))
                    with col2:
                        st.metric("Unique Role Groups", df['Role Group'].nunique())
                    with col3:
                        st.metric("Unique Names", df['Name'].nunique())
                    with col4:
                        # Count unknown role groups
                        unknown_count = len(df[df['Role Group'].str.contains('unknown', case=False, na=False)])
                        st.metric("Unknown Roles", unknown_count, delta=None if unknown_count == 0 else "⚠️")
                    with col5:
                        # Export button
                        if st.button("📥 Export CSV"):
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="💾 Download CSV",
                                data=csv,
                                file_name=f"{selected_episode}_credits.csv",
                                mime="text/csv"
                            )# Filter options
                    st.subheader("Filter & Analysis")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        role_groups = ['All'] + sorted(df['Role Group'].unique().tolist())
                        selected_role_group = st.selectbox(
                            "Filter by Role Group:",
                            options=role_groups,
                            key="role_group_filter"
                        )
                    
                    with col2:
                        search_name = st.text_input(
                            "Search by Name:",
                            key="name_search_filter"
                        )
                    
                    with col3:
                        # Filter for problematic entries
                        problematic_filter = st.selectbox(
                            "Show Problematic:",
                            options=["All", "Unknown Role Groups", "Duplicate Names", "All Problematic"],
                            key="problematic_filter"
                        )
                    
                    with col4:
                        # Show potential duplicates
                        name_counts = df['Name'].value_counts()
                        duplicates = name_counts[name_counts > 1]
                        if len(duplicates) > 0:
                            st.warning(f"⚠️ {len(duplicates)} names appear multiple times")
                            with st.expander("View Potential Duplicates"):
                                for name, count in duplicates.items():
                                    duplicate_roles = df[df['Name'] == name]['Role Group'].unique()
                                    st.write(f"**{name}** ({count}x): {', '.join(duplicate_roles)}")
                        else:
                            st.success("✅ No duplicate names found")
                      # Apply filters
                    filtered_df = df.copy()
                    if selected_role_group != 'All':
                        filtered_df = filtered_df[filtered_df['Role Group'] == selected_role_group]
                    if search_name:
                        filtered_df = filtered_df[filtered_df['Name'].str.contains(search_name, case=False, na=False)]
                    
                    # Apply problematic filter
                    if problematic_filter != 'All':
                        if problematic_filter == 'Unknown Role Groups':
                            filtered_df = filtered_df[filtered_df['Role Group'].str.contains('unknown', case=False, na=False)]
                        elif problematic_filter == 'Duplicate Names':
                            name_counts = df['Name'].value_counts()
                            duplicated_names = name_counts[name_counts > 1].index
                            filtered_df = filtered_df[filtered_df['Name'].isin(duplicated_names)]
                        elif problematic_filter == 'All Problematic':
                            # Show both unknown role groups and duplicate names
                            unknown_mask = filtered_df['Role Group'].str.contains('unknown', case=False, na=False)
                            name_counts = df['Name'].value_counts()
                            duplicated_names = name_counts[name_counts > 1].index
                            duplicate_mask = filtered_df['Name'].isin(duplicated_names)
                            filtered_df = filtered_df[unknown_mask | duplicate_mask]
                    
                    st.write(f"Showing {len(filtered_df)} of {len(df)} credits")
                    
                    # Show filtered stats if not showing all
                    if len(filtered_df) != len(df):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Filtered Credits", len(filtered_df))
                        with col2:
                            filtered_unknown = len(filtered_df[filtered_df['Role Group'].str.contains('unknown', case=False, na=False)])
                            st.metric("Unknown in Filter", filtered_unknown)
                        with col3:
                            filtered_name_counts = filtered_df['Name'].value_counts()
                            filtered_duplicates = len(filtered_name_counts[filtered_name_counts > 1])
                            st.metric("Duplicates in Filter", filtered_duplicates)
                    
                    # Legend for visual indicators
                    if problematic_filter in ['All', 'All Problematic']:
                        st.caption("**Legend:** ⚠️ = Unknown role group, 🔄 = Name appears in multiple role groups")
                    
                    # Bulk deduplication tool
                    if len(duplicates) > 0:
                        with st.expander("🔧 Bulk Deduplication Tool", expanded=False):
                            st.write("**Automatically resolve duplicates by keeping entries with the most specific role group (non-'unknown'):**")
                            
                            auto_dedupe_candidates = []
                            for name in duplicates.index:
                                name_entries = df[df['Name'] == name]
                                unknown_entries = name_entries[name_entries['Role Group'].str.contains('unknown', case=False, na=False)]
                                specific_entries = name_entries[~name_entries['Role Group'].str.contains('unknown', case=False, na=False)]
                                
                                if len(specific_entries) >= 1 and len(unknown_entries) >= 1:
                                    auto_dedupe_candidates.append({
                                        'name': name,
                                        'keep_count': len(specific_entries),
                                        'delete_count': len(unknown_entries),
                                        'keep_roles': specific_entries['Role Group'].unique().tolist(),
                                        'delete_roles': unknown_entries['Role Group'].unique().tolist()
                                    })
                            
                            if auto_dedupe_candidates:
                                st.write(f"Found {len(auto_dedupe_candidates)} names that can be auto-deduplicated:")
                                for candidate in auto_dedupe_candidates:
                                    st.write(f"- **{candidate['name']}**: Keep {candidate['keep_count']} specific entries ({', '.join(candidate['keep_roles'])}), delete {candidate['delete_count']} unknown entries")
                                
                                if st.button("🚀 Auto-Deduplicate All", key="auto_dedupe_all"):
                                    try:
                                        conn = sqlite3.connect(config.DB_PATH)
                                        cursor = conn.cursor()
                                        total_deleted = 0
                                        
                                        for candidate in auto_dedupe_candidates:
                                            name = candidate['name']
                                            name_entries = df[df['Name'] == name]
                                            unknown_entries = name_entries[name_entries['Role Group'].str.contains('unknown', case=False, na=False)]
                                            
                                            for _, entry in unknown_entries.iterrows():
                                                entry_original_data = processed_data[entry.name]  # entry.name is the index
                                                cursor.execute(f"""
                                                    DELETE FROM {config.DB_TABLE_CREDITS}
                                                    WHERE episode_id = ? AND name = ? AND role_group_normalized = ?
                                                          AND source_frame = ? AND COALESCE(role_detail, '') = COALESCE(?, '')
                                                """, (
                                                    selected_episode,
                                                    entry_original_data[1],  # name
                                                    entry_original_data[0],  # role_group
                                                    entry_original_data[3],  # source_frame
                                                    entry_original_data[2]   # role_detail
                                                ))
                                                total_deleted += 1
                                        
                                        conn.commit()
                                        conn.close()
                                        
                                        st.success(f"✅ Auto-deduplicated {len(auto_dedupe_candidates)} names, deleted {total_deleted} unknown entries")
                                        st.rerun()
                                        
                                    except Exception as e:
                                        st.error(f"Error during auto-deduplication: {e}")
                                        logging.error(f"Error during auto-deduplication: {e}", exc_info=True)
                            else:
                                st.info("No candidates found for auto-deduplication (need names with both specific and unknown role groups)")
                    
                    st.divider()
                    
                    # Credit editing interface
                    if len(filtered_df) > 0:
                        st.subheader("Edit Credits")
                        
                        # Use session state to track edits
                        if 'credit_edits' not in st.session_state:
                            st.session_state.credit_edits = {}
                        if 'credits_to_delete' not in st.session_state:
                            st.session_state.credits_to_delete = set()
                        
                        # Group credits by name and sort alphabetically
                        credits_by_name = {}
                        for idx, row in filtered_df.iterrows():
                            name = row['Name']
                            if name not in credits_by_name:
                                credits_by_name[name] = []
                            credits_by_name[name].append({
                                'idx': idx,
                                'row': row,
                                'original_data': processed_data[idx]
                            })
                        
                        # Sort names alphabetically
                        sorted_names = sorted(credits_by_name.keys())
                        
                        # Display credits grouped by name
                        for name in sorted_names:
                            credit_entries = credits_by_name[name]
                            
                            # Determine if this name has multiple entries (duplicates)
                            is_duplicate = len(credit_entries) > 1
                            has_unknown = any('unknown' in entry['row']['Role Group'].lower() for entry in credit_entries)
                            
                            # Create header with indicators
                            header = f"👤 {name}"
                            if has_unknown:
                                header += " ⚠️ UNKNOWN"
                            if is_duplicate:
                                header += f" 🔄 DUPLICATE ({len(credit_entries)} entries)"
                            
                            with st.expander(header, expanded=False):
                                if is_duplicate:
                                    # Show duplicate management interface
                                    st.subheader("🔄 Duplicate Management")
                                    st.warning(f"**{name}** appears {len(credit_entries)} times with different details.")
                                    
                                    # Create options for radio button selection
                                    variant_options = []
                                    variant_details = []
                                    
                                    for i, entry in enumerate(credit_entries):
                                        row = entry['row']
                                        role_detail = f" - {row['Role Detail']}" if row['Role Detail'] else ""
                                        option_label = f"Variant {i+1}: {row['Role Group']}{role_detail}"
                                        variant_options.append(option_label)
                                        variant_details.append({
                                            'entry': entry,
                                            'label': option_label,
                                            'row': row
                                        })
                                    
                                    # Add manual edit option
                                    variant_options.append("✏️ Manual Edit (customize)")
                                    
                                    # Selection interface
                                    safe_name_radio = name.replace(' ', '_').replace('(', '').replace(')', '').replace('.', '_').replace(',', '_')
                                    selected_option = st.radio(
                                        "Choose which version to keep:",
                                        options=range(len(variant_options)),
                                        format_func=lambda x: variant_options[x],
                                        key=f"variant_choice_{selected_episode}_{safe_name_radio}"
                                    )
                                      # Show details for selected variant (if not manual edit)
                                    if selected_option < len(credit_entries):
                                        selected_variant = variant_details[selected_option]
                                        st.info(f"**Selected:** {selected_variant['label']}")
                                        st.caption(f"Source: {selected_variant['row']['Source Frames']}")
                                        
                                        # Set up the variables for frame navigation using the selected variant
                                        row = selected_variant['row']
                                        original_data = selected_variant['entry']['original_data']
                                        idx = selected_variant['entry']['idx']
                                        show_edit_interface = True
                                          # Action buttons for the selected variant
                                        col1, col2, col3 = st.columns([1, 1, 1])
                                        
                                        with col1:
                                            if st.button(f"✅ Keep This Variant", key=f"keep_variant_{safe_name_radio}"):
                                                # Delete all other variants, keep only the selected one
                                                for i, entry in enumerate(credit_entries):
                                                    if i != selected_option:
                                                        st.session_state.credits_to_delete.add(entry['idx'])
                                                st.success(f"Marked other variants of {name} for deletion")
                                                st.rerun()
                                        
                                        with col2:
                                            if st.button(f"👥 Keep Both Separate", key=f"keep_both_{safe_name_radio}"):
                                                # Flag both entities for human modification and keep them separate
                                                for entry in credit_entries:
                                                    # Mark as manually modified to allow separate entities
                                                    entry['manually_modified'] = True
                                                st.success(f"Marked both variants of {name} to be kept as separate entities")
                                                st.info("Both entities will be available for individual modification")
                                                # Don't rerun here, let user see both for editing
                                        
                                        with col3:
                                            if st.button(f"🗑️ Delete All Variants", key=f"delete_all_{safe_name_radio}"):
                                                # Delete all variants
                                                for entry in credit_entries:
                                                    st.session_state.credits_to_delete.add(entry['idx'])
                                                st.success(f"Marked all variants of {name} for deletion")
                                                st.rerun()
                                    
                                    # Check if "Keep Both Separate" was selected for this duplicate
                                    keep_both_separate = any(entry.get('manually_modified', False) for entry in credit_entries)
                                    
                                    if keep_both_separate:
                                        st.subheader("👥 Separate Entity Management")
                                        st.info(f"Managing {len(credit_entries)} separate entities for {name}")
                                        
                                        # Show separate edit interfaces for each entity using containers instead of expanders
                                        for i, entry in enumerate(credit_entries):
                                            variant_info = variant_details[i] if i < len(variant_details) else {'label': f'Variant {i+1}', 'row': entry['row']}
                                            
                                            # Use a container with border styling instead of expander
                                            st.markdown(f"### ✏️ Edit {variant_info['label']}")
                                            
                                            # Create a styled container
                                            with st.container():
                                                st.markdown("""
                                                <div style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin: 10px 0; background-color: #f9f9f9;">
                                                """, unsafe_allow_html=True)
                                                
                                                st.caption(f"**Source:** {entry['row']['Source Frames']}")
                                                
                                                # Edit fields for this specific entity
                                                entity_key_suffix = f"_{selected_episode}_{safe_name_radio}_{i}_{entry['idx']}"
                                                
                                                col1, col2, col3 = st.columns([2, 2, 2])
                                                
                                                with col1:
                                                    new_name = st.text_input(
                                                        "Name", 
                                                        value=name, 
                                                        key=f"name_sep{entity_key_suffix}",
                                                        help=f"Edit name for {variant_info['label']}"
                                                    )
                                                
                                                with col2:
                                                    new_role_group = st.selectbox(
                                                        "Role Group", 
                                                        options=config.ROLE_GROUPS,
                                                        index=config.ROLE_GROUPS.index(entry['row']['Role Group']) if entry['row']['Role Group'] in config.ROLE_GROUPS else 0,
                                                        key=f"role_group_sep{entity_key_suffix}"
                                                    )
                                                    
                                                    # Show validation warning for invalid role groups
                                                    if not config.is_valid_role_group(new_role_group):
                                                        st.warning(f"⚠️ '{new_role_group}' is not in the predefined role groups and should be reviewed.")
                                                
                                                with col3:
                                                    new_role_detail = st.text_input(
                                                        "Role Detail", 
                                                        value=entry['row']['Role Detail'] or "", 
                                                        key=f"role_detail_sep{entity_key_suffix}"
                                                    )
                                                  # Update button for this entity
                                                if st.button(f"💾 Update {variant_info['label']}", key=f"update_sep{entity_key_suffix}"):
                                                    # Update logic for this specific entity
                                                    # This would need to be implemented based on your update mechanism
                                                    st.success(f"Updated {variant_info['label']}")
                                                
                                                # Close the styled container
                                                st.markdown("</div>", unsafe_allow_html=True)
                                                
                                                # Add separator between entities (except for the last one)
                                                if i < len(credit_entries) - 1:
                                                    st.markdown("---")
                                        
                                        # Don't show the regular edit interface
                                        show_edit_interface = False
                                    
                                    elif selected_option == len(variant_options) - 1:  # Manual Edit selected
                                        st.subheader("✏️ Manual Edit")
                                        st.info("Create a custom version based on the available variants")
                                        
                                        # Choose base variant for manual editing
                                        best_entry = None
                                        for entry in credit_entries:
                                            if 'unknown' not in entry['row']['Role Group'].lower():
                                                best_entry = entry
                                                break
                                        if not best_entry:
                                            best_entry = credit_entries[0]  # fallback to first entry
                                        
                                        # Set up for manual editing
                                        row = best_entry['row']
                                        original_data = best_entry['original_data']
                                        idx = best_entry['idx']
                                          # Show manual edit interface (will continue below)
                                        st.caption("Base variant: " + variant_details[0]['label'])
                                        
                                        # Show frame navigation and editing for manual edit
                                        show_edit_interface = True
                                else:
                                    # Single entry - show normal edit interface
                                    entry = credit_entries[0]
                                    row = entry['row']
                                    original_data = entry['original_data']
                                    idx = entry['idx']
                                    show_edit_interface = True
                                  # Show frame navigation and editing interface (only for single entries or manual edit)
                                if 'show_edit_interface' in locals() and show_edit_interface:
                                    # Show frame images if available
                                    source_frames = original_data[3]  # source_frame field
                                    if source_frames and source_frames != "manual_entry":
                                        st.subheader("📸 Frame Navigation")
                                          # Get all available frames (both saved and skipped) sorted by frame number
                                        frames_dir = config.EPISODES_BASE_DIR / selected_episode / "analysis" / "frames"
                                        skipped_frames_dir = config.EPISODES_BASE_DIR / selected_episode / "analysis" / "skipped_frames"
                                        
                                        # Debug: Show source frame information (not nested in expander)
                                        frame_files = source_frames.split(',') if ',' in source_frames else [source_frames]
                                        if st.session_state.get('debug_mode', False):
                                            st.info(f"🔍 **Debug:** Source frames: `{source_frames}` | Parsed: {[f.strip() for f in frame_files]} | Dir: `{frames_dir}`")
                                        # Helper function to extract frame number from filename
                                    def extract_frame_number(filename):
                                        """Extract frame number from various filename patterns"""
                                        try:
                                            if "num" in filename:
                                                # Handle patterns like "num12345_" or "num12345.jpg"
                                                num_part = filename.split("num")[1]
                                                # Remove file extension and any trailing parts after _
                                                num_str = num_part.split("_")[0].split(".")[0]
                                                return int(num_str)
                                        except (ValueError, IndexError):
                                            pass
                                        return None
                                    
                                    all_frames = []
                                    # Add frames from main frames directory
                                    if frames_dir.exists():
                                        for frame_file in frames_dir.glob("*.jpg"):
                                            # Extract frame number from filename
                                            frame_num = extract_frame_number(frame_file.name)
                                            if frame_num is not None:
                                                all_frames.append({
                                                    'path': frame_file,
                                                    'name': frame_file.name,
                                                    'num': frame_num,
                                                    'type': 'saved'
                                                })
                                    
                                    # Add frames from skipped frames directory
                                    if skipped_frames_dir.exists():
                                        for frame_file in skipped_frames_dir.glob("*.jpg"):
                                            frame_num = extract_frame_number(frame_file.name)
                                            if frame_num is not None:
                                                all_frames.append({
                                                    'path': frame_file,
                                                    'name': frame_file.name,
                                                    'num': frame_num,
                                                    'type': 'skipped'
                                                })
                                    
                                    # Sort all frames by frame number
                                    all_frames.sort(key=lambda x: x['num'])                                    # Find current frame in the list
                                    frame_files = source_frames.split(',') if ',' in source_frames else [source_frames]
                                    current_frame_name = frame_files[0].strip() if frame_files else ""
                                    
                                    current_index = 0
                                    source_frame_found = False
                                    for i, frame_info in enumerate(all_frames):
                                        if frame_info['name'] == current_frame_name:
                                            current_index = i
                                            source_frame_found = True
                                            break
                                    
                                    # If source frame is not found in available frames, try to add it
                                    if not source_frame_found and current_frame_name:
                                        # Try to find the source frame file directly
                                        source_frame_path = frames_dir / current_frame_name
                                        if not source_frame_path.exists():
                                            source_frame_path = skipped_frames_dir / current_frame_name
                                        
                                        if source_frame_path.exists():
                                            frame_num = extract_frame_number(current_frame_name)
                                            if frame_num is not None:
                                                frame_type = 'saved' if source_frame_path.parent.name == 'frames' else 'skipped'
                                                # Add the source frame to the list
                                                all_frames.append({
                                                    'path': source_frame_path,
                                                    'name': current_frame_name,
                                                    'num': frame_num,
                                                    'type': frame_type
                                                })
                                                # Re-sort by frame number
                                                all_frames.sort(key=lambda x: x['num'])
                                                # Find the new index of our source frame
                                                for i, frame_info in enumerate(all_frames):
                                                    if frame_info['name'] == current_frame_name:
                                                        current_index = i
                                                        source_frame_found = True
                                                        break
                                      # Debug information about frame matching
                                    if st.session_state.get('debug_mode', False):
                                        st.caption(f"🎯 **Target frame:** `{current_frame_name}` | **Found:** {source_frame_found} | **Index:** {current_index}")
                                        if all_frames:
                                            st.caption(f"📋 **Available frames:** {len(all_frames)} frames from {all_frames[0]['num'] if all_frames else 'N/A'} to {all_frames[-1]['num'] if all_frames else 'N/A'}")
                                            st.caption(f"🔢 **Frame numbers:** {[f['num'] for f in all_frames[:10]]}{'...' if len(all_frames) > 10 else ''}")
                                        else:
                                            st.caption("⚠️ **No frames found in directories**")
                                    # Initialize session state for frame navigation
                                    # Use name and idx to ensure unique keys across different people
                                    # Clean name to make it safe for Streamlit keys
                                    safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('.', '_').replace(',', '_')
                                    nav_key = f"frame_nav_{selected_episode}_{safe_name}_{idx}"
                                    
                                    # Debug session state
                                    if st.session_state.get('debug_mode', False):
                                        st.caption(f"🔑 **Nav Key:** `{nav_key}` | **Current Index (session):** {st.session_state.get(nav_key, 'Not Set')} | **Calculated Index:** {current_index}")
                                    
                                    if nav_key not in st.session_state:
                                        st.session_state[nav_key] = current_index
                                        if st.session_state.get('debug_mode', False):
                                            st.caption(f"🆕 **Initialized session state** to index {current_index}")
                                    else:
                                        # Always reset to the correct frame for this credit
                                        st.session_state[nav_key] = current_index
                                        if st.session_state.get('debug_mode', False):
                                            st.caption(f"🔄 **Reset session state** to index {current_index} (was {st.session_state.get(nav_key)})")
                                    
                                    # Navigation controls
                                    if len(all_frames) > 1:
                                        col1, col2, col3, col4, col5 = st.columns([1, 1, 3, 1, 1])
                                        
                                        with col1:
                                            if st.button("⬅️ Prev", key=f"prev_{nav_key}", disabled=st.session_state[nav_key] <= 0):
                                                st.session_state[nav_key] = max(0, st.session_state[nav_key] - 1)
                                                st.rerun()
                                        
                                        with col2:
                                            if st.button("➡️ Next", key=f"next_{nav_key}", disabled=st.session_state[nav_key] >= len(all_frames) - 1):
                                                st.session_state[nav_key] = min(len(all_frames) - 1, st.session_state[nav_key] + 1)
                                                st.rerun()
                                        
                                        with col3:
                                            # Frame selector dropdown
                                            frame_options = [f"Frame {frame['num']} ({frame['type']}): {frame['name']}" for frame in all_frames]
                                            selected_frame_idx = st.selectbox(
                                                "Jump to frame:",
                                                options=range(len(all_frames)),
                                                index=st.session_state[nav_key],
                                                format_func=lambda x: frame_options[x],
                                                key=f"select_{nav_key}"
                                            )
                                            if selected_frame_idx != st.session_state[nav_key]:
                                                st.session_state[nav_key] = selected_frame_idx
                                                st.rerun()
                                        
                                        with col4:
                                            st.write(f"{st.session_state[nav_key] + 1}/{len(all_frames)}")
                                        
                                        with col5:
                                            # Reset to original frame
                                            if st.button("🔄 Reset", key=f"reset_{nav_key}"):
                                                st.session_state[nav_key] = current_index
                                                st.rerun()                                    # Display current frame
                                    if all_frames:
                                        current_frame = all_frames[st.session_state[nav_key]]
                                        frame_type_badge = "🟢 SAVED" if current_frame['type'] == 'saved' else "🔴 SKIPPED"
                                        
                                        # Debug what frame is being displayed
                                        if st.session_state.get('debug_mode', False):
                                            st.caption(f"🖼️ **Displaying frame at index {st.session_state[nav_key]}:** `{current_frame['name']}` (Frame #{current_frame['num']})")
                                        
                                        st.write(f"**{frame_type_badge} | Frame {current_frame['num']} | {current_frame['name']}**")
                                        
                                        try:
                                            st.image(str(current_frame['path']), caption=f"{current_frame['name']} (Frame #{current_frame['num']})")
                                        except Exception as img_err:
                                            st.error(f"❌ Error loading frame: {img_err}")
                                        
                                        # Show all source frames if there are multiple
                                        if len(frame_files) > 1:
                                            st.subheader("📸 All Source Frames for this Credit")
                                            st.caption(f"This credit was found in {len(frame_files)} different frames:")
                                            
                                            # Create columns for multiple frames
                                            cols = st.columns(min(len(frame_files), 3))  # Max 3 columns
                                            for i, frame_file in enumerate(frame_files):
                                                frame_name = frame_file.strip()
                                                col_idx = i % 3
                                                
                                                with cols[col_idx]:
                                                    # Find this frame in all_frames
                                                    source_frame_info = None
                                                    for frame_info in all_frames:
                                                        if frame_info['name'] == frame_name:
                                                            source_frame_info = frame_info
                                                            break
                                                    
                                                    if source_frame_info:
                                                        badge = "🟢 SAVED" if source_frame_info['type'] == 'saved' else "🔴 SKIPPED"
                                                        st.write(f"**{badge}**")
                                                        try:
                                                            st.image(str(source_frame_info['path']), 
                                                                    caption=f"Frame #{source_frame_info['num']}")
                                                        except Exception as img_err:
                                                            st.error(f"❌ Error: {img_err}")
                                                    else:
                                                        # Frame not found in navigation, try direct path
                                                        frame_path = frames_dir / frame_name
                                                        if not frame_path.exists():
                                                            frame_path = skipped_frames_dir / frame_name
                                                        
                                                        if frame_path.exists():
                                                            try:
                                                                frame_num = extract_frame_number(frame_name)
                                                                frame_type = "🟢 SAVED" if frame_path.parent.name == 'frames' else "🔴 SKIPPED"
                                                                st.write(f"**{frame_type}**")
                                                                st.image(str(frame_path), caption=f"Frame #{frame_num}")
                                                            except Exception as img_err:
                                                                st.error(f"❌ Error: {img_err}")
                                                        else:
                                                            st.warning(f"❌ Frame not found: {frame_name}")
                                    else:
                                        # Fallback to original simple display
                                        frame_files = source_frames.split(',') if ',' in source_frames else [source_frames]
                                        for frame_file in frame_files[:4]:
                                            frame_path = config.EPISODES_BASE_DIR / selected_episode / "analysis" / "frames" / frame_file.strip()
                                            if frame_path.exists():
                                                try:
                                                    st.image(str(frame_path), caption=frame_file.strip())
                                                except Exception as img_err:
                                                    st.caption(f"❌ Error loading: {frame_file.strip()}")
                                            else:
                                                st.caption(f"❌ Not found: {frame_file.strip()}")                                
                                # Regular editing interface for this entry
                                # Check if this credit needs human modification
                                needs_human_modification = (
                                    row.get('Need revisioning for deduplication', False) or 
                                    'unknown' in row.get('Role Group', '').lower() or
                                    row.get('manually_modified', False)
                                )
                                
                                if needs_human_modification:
                                    st.subheader("✏️ Edit Credit")
                                    
                                    # Show why modification is needed
                                    reasons = []
                                    if row.get('Need revisioning for deduplication', False):
                                        reasons.append("🔄 Needs deduplication review")
                                    if 'unknown' in row.get('Role Group', '').lower():
                                        reasons.append("❓ Unknown role group")
                                    if row.get('manually_modified', False):
                                        reasons.append("✏️ Manually flagged for modification")
                                    
                                    if reasons:
                                        st.info("**Modification needed:** " + " • ".join(reasons))
                                    
                                    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                                else:
                                    # Show read-only view for credits that don't need modification
                                    st.subheader("👁️ Credit Info (Read-Only)")
                                    st.info("This credit doesn't require human modification. All fields appear to be correctly processed.")
                                    
                                    col1, col2, col3 = st.columns([2, 2, 3])
                                    
                                    with col1:
                                        st.text_input("Name", value=row['Name'], disabled=True, key=f"readonly_name_{idx}")
                                    with col2:
                                        st.text_input("Role Group", value=row['Role Group'], disabled=True, key=f"readonly_role_group_{idx}")
                                    with col3:
                                        st.text_input("Role Detail", value=row['Role Detail'] or "", disabled=True, key=f"readonly_role_detail_{idx}")
                                    
                                    # Option to enable editing if needed
                                    if st.button("🔓 Enable Editing", key=f"enable_edit_{idx}"):
                                        st.session_state[f"force_edit_{idx}"] = True
                                        st.rerun()
                                  # Check if editing is enabled (either needed or forced)
                                editing_enabled = needs_human_modification or st.session_state.get(f"force_edit_{idx}", False)
                                
                                if editing_enabled and not st.session_state.get(f"readonly_mode_{idx}", False):
                                    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                                    
                                    # Create unique keys for each field using name and idx
                                    safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('.', '_').replace(',', '_')
                                    name_key = f"name_{selected_episode}_{safe_name}_{idx}"
                                    role_group_key = f"role_group_{selected_episode}_{safe_name}_{idx}"
                                    role_detail_key = f"role_detail_{selected_episode}_{safe_name}_{idx}"
                                    
                                    with col1:
                                        new_name = st.text_input(
                                            "Name",
                                            value=row['Name'],
                                            key=name_key
                                        )
                                    
                                    with col2:
                                        # Role group dropdown with all available options
                                        all_role_groups = sorted(df['Role Group'].unique().tolist())
                                        current_index = all_role_groups.index(row['Role Group']) if row['Role Group'] in all_role_groups else 0
                                        new_role_group = st.selectbox(
                                            "Role Group",
                                            options=all_role_groups,
                                            index=current_index,
                                            key=role_group_key                                        )
                                        
                                        # Validate role group
                                        is_valid, validation_message = config.validate_role_group(new_role_group)
                                        if not is_valid:
                                            st.warning(f"⚠️ {validation_message}")
                                            st.caption("This role group may need manual review and categorization.")
                                    
                                    with col3:
                                        new_role_detail = st.text_input(
                                            "Role Detail",
                                            value=row['Role Detail'] if row['Role Detail'] else "",
                                            key=role_detail_key
                                        )
                                    
                                    with col4:
                                        delete_key = f"delete_{selected_episode}_{safe_name}_{idx}"
                                        if st.button("🗑️ Delete", key=delete_key):
                                            st.session_state.credits_to_delete.add(idx)
                                            st.rerun()
                                    
                                    # Track changes - store original database values for matching
                                    original_values = (row['Name'], row['Role Group'], row['Role Detail'])
                                    current_values = (new_name, new_role_group, new_role_detail if new_role_detail else None)
                                    
                                    if original_values != current_values:
                                        st.session_state.credit_edits[idx] = {
                                            'name': new_name,
                                            'role_group': new_role_group,
                                            'role_detail': new_role_detail if new_role_detail else None,
                                            'original_name': row['Name'],
                                            'original_role_group': row['Role Group'],
                                            'original_role_detail': row['Role Detail'],
                                            'source_frame': original_data[3],  # Keep original source_frame for DB matching
                                            'frame_numbers': original_data[4]   # Keep original frame_numbers for DB matching
                                        }
                                    elif idx in st.session_state.credit_edits:
                                        del st.session_state.credit_edits[idx]
                                    
                                    # Show metadata
                                    st.caption(f"**Source Frames:** {row['Source Frames']}")
                                    st.caption(f"**Frame Numbers:** {row['Frame Numbers']}")
                                    st.caption(f"**Scene Position:** {row['Scene Position']}")
                          # Show pending changes
                        if st.session_state.credit_edits or st.session_state.credits_to_delete:
                            st.subheader("Pending Changes")
                            
                            if st.session_state.credit_edits:
                                st.write("**Modified Credits:**")
                                for idx, changes in st.session_state.credit_edits.items():
                                    if idx < len(processed_data):
                                        original_data = processed_data[idx]
                                        st.write(f"- **{original_data[1]}** ({original_data[0]}) → **{changes['name']}** ({changes['role_group']})")
                            
                            if st.session_state.credits_to_delete:
                                st.write("**Credits to Delete:**")
                                for idx in st.session_state.credits_to_delete:
                                    if idx < len(processed_data):
                                        delete_data = processed_data[idx]
                                        st.write(f"- **{delete_data[1]}** ({delete_data[0]})")
                            
                            # Save/Cancel buttons
                            col1, col2, col3 = st.columns([1, 1, 2])
                            
                            with col1:
                                if st.button("💾 Save Changes", type="primary"):
                                    try:
                                        conn = sqlite3.connect(config.DB_PATH)
                                        cursor = conn.cursor()
                                        
                                        # Apply edits
                                        for idx, changes in st.session_state.credit_edits.items():
                                            if idx < len(processed_data):
                                                original_data = processed_data[idx]
                                                # Use original database values for WHERE clause matching
                                                cursor.execute(f"""
                                                    UPDATE {config.DB_TABLE_CREDITS}
                                                    SET name = ?, role_group_normalized = ?, role_detail = ?
                                                    WHERE episode_id = ? AND name = ? AND role_group_normalized = ? 
                                                          AND source_frame = ? AND COALESCE(role_detail, '') = COALESCE(?, '')
                                                """, (
                                                    changes['name'], 
                                                    changes['role_group'], 
                                                    changes['role_detail'],
                                                    selected_episode,
                                                    changes['original_name'],
                                                    changes['original_role_group'],
                                                    changes['source_frame'],
                                                    changes['original_role_detail']
                                                ))
                                        
                                        # Apply deletions
                                        for idx in st.session_state.credits_to_delete:
                                            if idx < len(processed_data):
                                                original_data = processed_data[idx]
                                                cursor.execute(f"""
                                                    DELETE FROM {config.DB_TABLE_CREDITS}
                                                    WHERE episode_id = ? AND name = ? AND role_group_normalized = ?
                                                          AND source_frame = ? AND COALESCE(role_detail, '') = COALESCE(?, '')
                                                """, (
                                                    selected_episode,
                                                    original_data[1],  # name
                                                    original_data[0],  # role_group
                                                    original_data[3],  # source_frame
                                                    original_data[2]   # role_detail
                                                ))
                                        
                                        conn.commit()
                                        conn.close()
                                        
                                        # Clear pending changes
                                        st.session_state.credit_edits = {}
                                        st.session_state.credits_to_delete = set()
                                        
                                        st.success("Changes saved successfully!")
                                        st.rerun()
                                        
                                    except Exception as e:
                                        st.error(f"Error saving changes: {e}")
                                        logging.error(f"Error saving credit changes: {e}", exc_info=True)
                            
                            with col2:
                                if st.button("❌ Cancel Changes"):
                                    st.session_state.credit_edits = {}
                                    st.session_state.credits_to_delete = set()
                                    st.rerun()
                        
                        # Add new credit section
                        st.subheader("Add New Credit")
                        with st.expander("➕ Add New Credit", expanded=False):
                            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                            
                            with col1:
                                new_credit_name = st.text_input("Name", key="new_credit_name")
                            
                            with col2:
                                all_role_groups = sorted(df['Role Group'].unique().tolist())
                                new_credit_role_group = st.selectbox(
                                    "Role Group",
                                    options=all_role_groups,
                                    key="new_credit_role_group"
                                )
                            
                            with col3:
                                new_credit_role_detail = st.text_input("Role Detail", key="new_credit_role_detail")
                            
                            with col4:
                                if st.button("➕ Add"):
                                    if new_credit_name:
                                        try:
                                            conn = sqlite3.connect(config.DB_PATH)
                                            cursor = conn.cursor()
                                            
                                            cursor.execute(f"""
                                                INSERT INTO {config.DB_TABLE_CREDITS}
                                                (episode_id, name, role_group_normalized, role_detail, 
                                                 source_frame, original_frame_number, scene_position)
                                                VALUES (?, ?, ?, ?, ?, ?, ?)
                                            """, (
                                                selected_episode,
                                                new_credit_name,
                                                new_credit_role_group,
                                                new_credit_role_detail if new_credit_role_detail else None,
                                                "manual_entry",  # Mark as manually added
                                                "",  # Empty frame number
                                                "manual"  # Mark as manually added
                                            ))
                                            
                                            conn.commit()
                                            conn.close()
                                            
                                            st.success(f"Added new credit: {new_credit_name}")
                                            st.rerun()
                                            
                                        except Exception as e:
                                            st.error(f"Error adding new credit: {e}")
                                            logging.error(f"Error adding new credit: {e}", exc_info=True)
                                    else:
                                        st.warning("Please enter a name for the new credit.")
                    
    except Exception as e:
        st.error(f"Error loading credits data: {e}")
        logging.error(f"Error in credit review interface: {e}", exc_info=True)

with tab3:
    st.header("Pipeline Logs")
    st.text_area("Live Logs:", value=st.session_state.get('log_content', ''), height=600, key="log_display_text_area_v3")