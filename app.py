import json
import logging
import sqlite3
from pathlib import Path

import streamlit as st
from PIL import Image

st.set_page_config(layout="wide")


from scenedetect import open_video

# Assuming scripts_v3 is in a location accessible by Python's path
# This can be achieved by running the app from the project's root directory
# or by installing the project package (e.g., using 'pip install -e .')
from scripts_v3 import azure_vlm_processing, config, constants, frame_analysis, scene_detection, utils
from scripts_v3.exceptions import ConfigError


def display_image(path: Path, width: int = 200) -> None:
    if not path.exists():
        st.warning(f"Image not found: {path}")
        return
    try:
        img = Image.open(path)
        st.image(img.convert("RGB"), width=width, caption=f"{path.name}")
    except Exception as e:
        st.error(f"Error displaying image {path}: {e}")


def get_cached_ocr_reader():
    selected_lang = st.session_state.get('ocr_language', 'it')
    selected_engine = st.session_state.get('ocr_engine_type', config.DEFAULT_OCR_ENGINE)
    reader_key = f"ocr_reader_{selected_engine}_{selected_lang}"
    current_reader_engine_key = "current_ocr_reader_engine_type"
    current_reader_lang_key = "current_ocr_reader_lang"

    if (
        reader_key not in st.session_state
        or st.session_state.get(reader_key) is None
        or st.session_state.get(current_reader_engine_key) != selected_engine
        or st.session_state.get(current_reader_lang_key) != selected_lang
    ):
        logging.info(f"Initializing OCR reader: Engine={selected_engine}, Lang={selected_lang}")
        try:
            if selected_engine == "paddleocr":
                paddle_lang_code = config.PADDLEOCR_LANG_MAP.get(selected_lang, selected_lang)
                st.session_state[reader_key] = utils.get_paddleocr_reader(lang=paddle_lang_code)
            else:
                st.session_state[reader_key] = None
                logging.error(f"Unsupported OCR engine: {selected_engine}")
            st.session_state[current_reader_engine_key] = selected_engine
            st.session_state[current_reader_lang_key] = selected_lang
            if st.session_state.get(reader_key):
                logging.info(f"OCR Reader for {selected_engine} ({selected_lang}) initialized successfully.")
            else:
                logging.error(
                    f"Failed to initialize OCR Reader for {selected_engine} ({selected_lang}). Reader is None."
                )
        except Exception as e:
            st.session_state[reader_key] = None
            logging.error(f"Exception initializing OCR reader {selected_engine} ({selected_lang}): {e}", exc_info=True)
            st.error(f"Error initializing OCR for {selected_engine} ({selected_lang}): {e}")

    return st.session_state.get(reader_key)


def force_refresh_ocr_reader():
    """Force refresh the OCR reader to fix model caching issues."""
    selected_lang = st.session_state.get('ocr_language', 'it')
    selected_engine = st.session_state.get('ocr_engine_type', config.DEFAULT_OCR_ENGINE)
    reader_key = f"ocr_reader_{selected_engine}_{selected_lang}"
    
    # Clear the cached reader
    if reader_key in st.session_state:
        logging.info(f"Clearing cached OCR reader for {selected_engine} ({selected_lang})")
        st.session_state[reader_key] = None
    
    # Clear tracking keys
    st.session_state.pop("current_ocr_reader_engine_type", None)
    st.session_state.pop("current_ocr_reader_lang", None)
    
    # Force recreation
    return get_cached_ocr_reader()


utils.setup_logging()

try:
    if not config.RAW_VIDEO_DIR.exists():
        raise ConfigError(f"Raw video directory not found: {config.RAW_VIDEO_DIR}")
    # Role groups are now defined directly in config.py, no external file needed
except ConfigError as cfg_err:
    st.error(f"Configuration error: {cfg_err}")
    st.stop()

openai_logger = logging.getLogger("openai")
openai_logger.setLevel(logging.INFO)

utils.init_db()

st.title("🎬 Film Credit Extraction Pipeline v3")

# Custom CSS to make the main content area wider
st.markdown(
    """
<style>
    .main .block-container {
        max-width: 100% !important;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Make sidebar narrower */
    [data-testid="stSidebar"] {
        width: 200px !important;
    }
    
    /* Adjust main content margin to match narrower sidebar */
    .css-1y4p8pa, [data-testid="stSidebar"] + div {
        margin-left: 220px !important;
    }
    
    /* Make tabs wider */
    .stTabs [data-baseweb="tab-list"] {
        width: 100% !important;
    }
    
    /* Make columns wider */
    .stHorizontalBlock {
        width: 100% !important;
    }
    
    /* Make dataframes wider */
    .stDataFrame {
        width: 100% !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.sidebar.header("Configuration")

st.session_state['ocr_language'] = st.sidebar.selectbox(
    "OCR Language",
    ['it', 'en', 'ch'],
    index=['it', 'en', 'ch'].index(st.session_state.get('ocr_language', 'it')),
    key='ocr_language_select_key_v3',
)

# Set OCR engine to PaddleOCR (no user selection needed)
st.session_state['ocr_engine_type'] = 'paddleocr'

# Add OCR refresh button for troubleshooting
if st.sidebar.button("🔄 Refresh OCR Reader", 
                    key="refresh_ocr_button", 
                    help="Click this if you encounter OCR errors or want to refresh the model"):
    force_refresh_ocr_reader()
    st.sidebar.success("OCR reader refreshed!")

st.sidebar.header("Insert here words that appears as channel logo or watermark in the video")
if 'user_stopwords' not in st.session_state:
    st.session_state.user_stopwords = utils.load_user_stopwords()

edited_stopwords_text = st.sidebar.text_area(
    "Edit words (one per line):",
    value="\n".join(st.session_state.user_stopwords),
    height=150,
    key="stopwords_text_area_key_v3",
)

if st.sidebar.button("Save Stopwords", key="save_stopwords_button_key_v3"):
    updated_stopwords = [line.strip() for line in edited_stopwords_text.splitlines() if line.strip()]
    utils.save_user_stopwords(updated_stopwords)
    st.session_state.user_stopwords = updated_stopwords
    st.sidebar.success("Stopwords saved!")

config.RAW_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
allowed_extensions = (".mp4", ".mkv", ".avi", ".mov")
try:
    video_files_paths = sorted(
        [f for f in config.RAW_VIDEO_DIR.iterdir() if f.is_file() and f.suffix.lower() in allowed_extensions]
    )
except FileNotFoundError:
    video_files_paths = []
    st.warning(f"Video directory {config.RAW_VIDEO_DIR} not found. Please create it and add videos.")

st.sidebar.header("Video Segments for Scene Detection")

st.sidebar.subheader("Scene Detection Margins")

# Initialize session state for scene detection method
if 'scene_detection_method' not in st.session_state:
    st.session_state['scene_detection_method'] = 'scene_count'

# Choose detection method
scene_detection_method = st.sidebar.radio(
    "Selection method:",
    options=['time_based', 'scene_count', 'whole_episode'],
    format_func=lambda x: {
        'scene_count': '📋 By scene count',
        'time_based': '⏱️ By time duration', 
        'whole_episode': '🎬 Run on the whole episode'
    }[x],
    key="scene_detection_method_selector",
    help="Choose how to select which parts of the video to analyze",
)
st.session_state['scene_detection_method'] = scene_detection_method

# Show start/end controls only when NOT running on whole episode
if scene_detection_method != 'whole_episode':
    if scene_detection_method == 'scene_count':
        # Scene count based selection (original method)
        st.sidebar.caption(
            f"Specify how many scenes at the start and end of the video to consider for OCR (default: {constants.DEFAULT_START_SCENES_COUNT} each)."
        )

        if 'scene_start_count' not in st.session_state:
            st.session_state['scene_start_count'] = constants.DEFAULT_START_SCENES_COUNT
        if 'scene_end_count' not in st.session_state:
            st.session_state['scene_end_count'] = constants.DEFAULT_END_SCENES_COUNT

        scene_start_count = st.sidebar.number_input(
            "Number of scenes at start:",
            min_value=0,
            max_value=9999,
            value=st.session_state['scene_start_count'],
            step=1,
            key="scene_start_count_input",
        )
        scene_end_count = st.sidebar.number_input(
            "Number of scenes at end:",
            min_value=0,
            max_value=9999,
            value=st.session_state['scene_end_count'],
            step=1,
            key="scene_end_count_input",
        )
        st.session_state['scene_start_count'] = scene_start_count
        st.session_state['scene_end_count'] = scene_end_count

    else:
        # Time-based selection
        st.sidebar.caption("Specify how many minutes at the start and end of the video to consider for OCR.")

        if 'scene_start_minutes' not in st.session_state:
            st.session_state['scene_start_minutes'] = 7.0  # Default 3 minutes
        if 'scene_end_minutes' not in st.session_state:
            st.session_state['scene_end_minutes'] = 7.0  # Default 5 minutes

        scene_start_minutes = st.sidebar.number_input(
            "Minutes at start:",
            min_value=0.0,
            max_value=9999.0,
            value=st.session_state['scene_start_minutes'],
            step=0.5,
            format="%.1f",
            key="scene_start_minutes_input",
            help="How many minutes from the beginning to analyze",
        )
        scene_end_minutes = st.sidebar.number_input(
            "Minutes at end:",
            min_value=0.0,
            max_value=9999.0,
            value=st.session_state['scene_end_minutes'],
            step=0.5,
            format="%.1f",
            key="scene_end_minutes_input",
            help="How many minutes from the end to analyze",
        )
        st.session_state['scene_start_minutes'] = scene_start_minutes
        st.session_state['scene_end_minutes'] = scene_end_minutes
else:
    # When running on whole episode, show a message
    st.sidebar.info("🎬 Analyzing the entire episode - no time/scene limits will be applied")

# Store the current method for use in processing
st.session_state['current_detection_method'] = scene_detection_method


st.sidebar.subheader("Preview Video")
if 'selected_video_for_preview' not in st.session_state:
    st.session_state.selected_video_for_preview = None

# Add checkbox to control video loading
show_video_preview = st.sidebar.checkbox(
    "📺 Enable Video Preview",
    value=False,
    help="Check this to load and display the video preview (may consume memory)",
    key="enable_video_preview_checkbox",
)

if video_files_paths:
    video_options = {video_path.name: str(video_path) for video_path in video_files_paths}
    selected_video_name = st.sidebar.selectbox(
        "Select video to preview:", options=list(video_options.keys()), index=0, key="preview_video_selector_v3"
    )
    if selected_video_name:
        st.session_state.selected_video_for_preview = video_options[selected_video_name]

# Only load and display video if checkbox is checked
if show_video_preview and st.session_state.selected_video_for_preview:
    try:
        video_path_obj = Path(st.session_state.selected_video_for_preview)
        with open(video_path_obj, 'rb') as vf:
            vid_bytes = vf.read()
        st.sidebar.video(vid_bytes, format=f"video/{video_path_obj.suffix.lstrip('.')}")
    except Exception as e:
        st.sidebar.error(f"Error loading video preview: {e}")
elif show_video_preview and not st.session_state.selected_video_for_preview:
    if not video_files_paths:
        st.sidebar.info(f"No videos found in '{config.RAW_VIDEO_DIR}'. Add videos to enable preview.")
    else:
        st.sidebar.info("Select a video from 'data/raw' to enable preview.")
elif not show_video_preview:
    st.sidebar.info("💡 Check 'Enable Video Preview' above to load and display the selected video.")


st.subheader("Step 1 Configuration")
st.info(
    "🤖 **Automatic Scene Detection**: The pipeline will use AI to identify potential credit scenes based on the margins you set in the sidebar."
)

if 'log_content' not in st.session_state:
    st.session_state.log_content = ""

# Add tab state management
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 0

# Add persistent episode tracking (separate from preserve flags)
if 'last_selected_episode' not in st.session_state:
    st.session_state.last_selected_episode = None

# Add session state flags for preserving tab and episode state
if 'preserve_review_tab' not in st.session_state:
    st.session_state.preserve_review_tab = False

if 'preserve_episode' not in st.session_state:
    st.session_state.preserve_episode = None

# Show a note if we're preserving the review tab and auto-navigate
if st.session_state.preserve_review_tab:
    episode_info = ""
    if st.session_state.preserve_episode:
        episode_info = f" for **{st.session_state.preserve_episode}**"
    st.success(f"� **Staying in Review tab**{episode_info} - Your work continues below!")
    # Set current tab to review tab (index 1) when preserving
    st.session_state.current_tab = 1
    # Don't clear preserve flags immediately - let the review tab handle it
else:
    # Normal tab behavior - default to first tab if not preserving
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = 0

# Create navigation buttons that look like tabs
col_tab1, col_tab2, col_tab3 = st.columns(3)

with col_tab1:
    if st.button("⚙️ Setup & Run Pipeline", key="tab_button_1", 
                 type="primary" if st.session_state.current_tab == 0 else "secondary"):
        st.session_state.current_tab = 0
        st.session_state.preserve_review_tab = False  # Clear preservation when manually switching
        st.rerun()

with col_tab2:
    if st.button("✏️ Review & Edit Credits", key="tab_button_2", 
                 type="primary" if st.session_state.current_tab == 1 else "secondary"):
        st.session_state.current_tab = 1
        st.session_state.preserve_review_tab = False  # Clear preservation when manually switching
        st.rerun()

with col_tab3:
    if st.button("📊 Logs", key="tab_button_3", 
                 type="primary" if st.session_state.current_tab == 2 else "secondary"):
        st.session_state.current_tab = 2
        st.session_state.preserve_review_tab = False  # Clear preservation when manually switching
        st.rerun()

st.markdown("---")  # Visual separator

# Render content based on current tab
if st.session_state.current_tab == 0:
    st.header("1. Select Videos for Processing")

    if not video_files_paths:
        st.warning(
            f"No video files found in {config.RAW_VIDEO_DIR}. Please add videos with extensions: {', '.join(allowed_extensions)}"
        )
        selected_videos_str_paths = []
    else:
        # Add Select All / Deselect All buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("✅ Select All", key="select_all_videos"):
                for video_file_path_obj in video_files_paths:
                    st.session_state[f"vid_select_{video_file_path_obj.name}_v3"] = True
                st.rerun()

        with col2:
            if st.button("❌ Deselect All", key="deselect_all_videos"):
                for video_file_path_obj in video_files_paths:
                    st.session_state[f"vid_select_{video_file_path_obj.name}_v3"] = False
                st.rerun()

        st.write("Select videos from `data/raw` to include in the batch:")

        # Video selection checkboxes
        selected_videos_str_paths = []
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

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        run_step1_button = st.button("STEP 1: Identify Candidate Scenes", key="run_step1_btn_v3")
    with col2:
        run_step2_button = st.button("STEP 2: Analyze Scene Frames", key="run_step2_btn_v3")
    with col3:
        run_step3_button = st.button("STEP 3: Azure VLM OCR", key="run_step3_btn_v3")
    with col4:
        run_step4_button = st.button("STEP 4: IMDB Validation", key="run_step4_btn_v3")
    with col5:
        run_all_steps_button = st.button("RUN ALL STEPS", key="run_all_steps_btn_v3")

    if 'user_selected_scenes_for_step2' not in st.session_state:
        st.session_state.user_selected_scenes_for_step2 = {}

    # Initialize step 2 running state
    if 'step2_running' not in st.session_state:
        st.session_state.step2_running = False

    # Show scene review based on selected videos from Step 1 (but hide when Step 2 is running)
    if selected_videos_str_paths and not st.session_state.step2_running:
        st.subheader("2a. Review Candidate Scenes for Step 2")
        any_review_ui_shown_for_an_episode = False

        # Check which of the selected videos have completed Step 1
        episodes_with_step1_complete = []
        for video_path_str in selected_videos_str_paths:
            episode_id = Path(video_path_str).stem
            scene_analysis_file = config.EPISODES_BASE_DIR / episode_id / "analysis" / "initial_scene_analysis.json"
            if scene_analysis_file.exists():
                episodes_with_step1_complete.append(episode_id)

        if not episodes_with_step1_complete:
            st.info("⏳ Selected videos haven't completed Step 1 yet. Run Step 1 first to review candidate scenes.")
        else:
            st.info(
                f"📋 Found {len(episodes_with_step1_complete)} of {len(selected_videos_str_paths)} selected episode(s) with completed Step 1. Review and select scenes for Step 2 below."
            )

            for episode_id_review in episodes_with_step1_complete:
                scene_analysis_file_review = (
                    config.EPISODES_BASE_DIR / episode_id_review / "analysis" / "initial_scene_analysis.json"
                )
                step1_frames_base_dir_review = (
                    config.EPISODES_BASE_DIR / episode_id_review / "analysis" / "step1_representative_frames"
                )

            if scene_analysis_file_review.exists():
                try:
                    with open(scene_analysis_file_review, 'r', encoding='utf-8') as f_review:
                        step1_output = json.load(f_review)

                    candidate_scenes_from_step1 = step1_output.get("candidate_scenes", [])

                    if not candidate_scenes_from_step1:
                        st.info(f"No candidate scenes found by Step 1 for {episode_id_review} to review.")
                        st.session_state.user_selected_scenes_for_step2[episode_id_review] = []

                    any_review_ui_shown_for_an_episode = True
                    with st.expander(
                        f"Review scenes for {episode_id_review} (Found {len(candidate_scenes_from_step1)} candidates)",
                        expanded=True,
                    ):
                        if episode_id_review not in st.session_state.user_selected_scenes_for_step2:
                            initial_selected_indices = [
                                i for i, scene in enumerate(candidate_scenes_from_step1) if scene.get("selected", True)
                            ]
                            st.session_state.user_selected_scenes_for_step2[
                                episode_id_review
                            ] = initial_selected_indices

                        if not step1_frames_base_dir_review.exists():
                            st.warning(f"Directory does not exist: {step1_frames_base_dir_review}")

                        selected_indices_for_episode = st.session_state.user_selected_scenes_for_step2.get(
                            episode_id_review, []
                        )

                        cols_review = st.columns(3)
                        current_col_idx_review = 0
                        newly_selected_indices = []

                        for idx, scene_info_review in enumerate(candidate_scenes_from_step1):
                            with cols_review[current_col_idx_review]:
                                scene_label = f"Scene {scene_info_review.get('scene_index', idx)} ({scene_info_review.get('position', 'N/A')})"
                                is_selected_in_ui = st.checkbox(
                                    scene_label,
                                    value=(idx in selected_indices_for_episode),
                                    key=f"scene_select_{episode_id_review}_{idx}",
                                )
                                if is_selected_in_ui:
                                    newly_selected_indices.append(idx)

                                st.caption(
                                    f"Frames: {scene_info_review.get('original_start_frame')} - {scene_info_review.get('original_end_frame')}"
                                )
                                if scene_info_review.get("text_found_in_samples"):
                                    ocr_sample_text = scene_info_review.get('text_found_in_samples')[0][:50]
                                    st.caption(f'OCR Sample: "{ocr_sample_text}..."')

                                representative_frames = scene_info_review.get("representative_frames_saved", [])
                                if representative_frames:
                                    img_filename = representative_frames[0]
                                    img_path = step1_frames_base_dir_review / img_filename
                                    if img_path.exists():
                                        display_image(img_path, width=200)
                                    else:
                                        st.caption(f"Representative frame not found at {img_path}")
                                        alt_path = (
                                            config.EPISODES_BASE_DIR
                                            / episode_id_review
                                            / "analysis"
                                            / "step1_representative_frames"
                                            / img_filename
                                        )
                                        if alt_path.exists():
                                            display_image(alt_path, width=200)
                                        else:
                                            st.caption(f"Alternate image also not found: {alt_path}")
                                else:
                                    st.caption("No representative frame saved.")
                                st.markdown("---")
                            current_col_idx_review = (current_col_idx_review + 1) % 3

                        if set(st.session_state.user_selected_scenes_for_step2.get(episode_id_review, [])) != set(
                            newly_selected_indices
                        ):
                            st.session_state.user_selected_scenes_for_step2[episode_id_review] = newly_selected_indices
                            logging.info(
                                f"User updated scene selection for {episode_id_review}. New selection: {newly_selected_indices}"
                            )

                            updated_candidate_scenes_for_json_save = []
                            for i, scene_data_original in enumerate(candidate_scenes_from_step1):
                                scene_copy = scene_data_original.copy()
                                scene_copy["selected"] = i in newly_selected_indices
                                updated_candidate_scenes_for_json_save.append(scene_copy)

                            data_to_save_to_json = step1_output.copy()
                            data_to_save_to_json["candidate_scenes"] = updated_candidate_scenes_for_json_save

                            try:
                                with open(scene_analysis_file_review, 'w', encoding='utf-8') as f_save:
                                    json.dump(data_to_save_to_json, f_save, indent=4)
                                logging.info(
                                    f"Saved updated scene selections to {scene_analysis_file_review.name} for {episode_id_review}"
                                )
                            except Exception as e_save_json:
                                logging.error(
                                    f"Error saving scene selections to JSON for {episode_id_review}: {e_save_json}"
                                )
                                st.error(f"Failed to save selection changes for {episode_id_review}.")

                        current_selected_count_for_info = len(
                            st.session_state.user_selected_scenes_for_step2.get(episode_id_review, [])
                        )
                        st.info(
                            f"User has selected {current_selected_count_for_info} scenes for {episode_id_review} for Step 2 processing."
                        )

                except Exception as e_review:
                    st.error(f"Error loading or displaying scenes for review ({episode_id_review}): {e_review}")
                    logging.error(
                        f"Error during scene review display for {episode_id_review}: {e_review}", exc_info=True
                    )
            else:                st.warning(
                    f"Step 1 output file not found for {episode_id_review}. Run Step 1 to generate candidate scenes for review."
                )

        if any_review_ui_shown_for_an_episode:
            st.markdown("---")
            st.success("Scene selections updated. Ready for Step 2 if desired.")
    elif not st.session_state.step2_running:  # Only show this message when Step 2 is not running
        if not selected_videos_str_paths:            
            st.info("📝 Select videos in Step 1 above to review their candidate scenes for Step 2.")
        # If videos are selected but Step 2 is running, don't show the scene review section

    # Initialize user selected scenes for step 2
    if 'user_selected_scenes_for_step2' not in st.session_state:
        st.session_state.user_selected_scenes_for_step2 = {}

    # Button handling for Setup tab
    if run_step1_button:
        if not selected_videos_str_paths:
            st.warning("Please select at least one video for Step 1.")
        else:
            st.subheader("Running Step 1: Automatic Scene Detection")
            
            # Force refresh OCR reader to ensure clean state
            st.info("🔄 Initializing fresh OCR reader...")
            ocr_reader = force_refresh_ocr_reader()
            current_ocr_engine = st.session_state.get('ocr_engine_type', config.DEFAULT_OCR_ENGINE)
            user_stopwords = st.session_state.get('user_stopwords', [])

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
                                # Determine scene selection parameters based on method
                                if st.session_state.get('current_detection_method') == 'whole_episode':
                                    # Run on whole episode
                                    scenes, status, error_msg = scene_detection.identify_candidate_scenes(
                                        video_path_obj,
                                        episode_id_proc,
                                        ocr_reader,
                                        current_ocr_engine,
                                        user_stopwords,
                                        whole_episode=True,
                                    )
                                elif st.session_state.get('current_detection_method') == 'time_based':
                                    # Convert time to scene counts (approximate)
                                    scene_start_minutes = st.session_state.get('scene_start_minutes', 3.0)
                                    scene_end_minutes = st.session_state.get('scene_end_minutes', 5.0)

                                    # Use time-based selection
                                    scenes, status, error_msg = scene_detection.identify_candidate_scenes(
                                        video_path_obj,
                                        episode_id_proc,
                                        ocr_reader,
                                        current_ocr_engine,
                                        user_stopwords,
                                        time_segments=(scene_start_minutes, scene_end_minutes),
                                    )
                                else:
                                    # Use scene count-based selection (default)
                                    scene_start_count = st.session_state.get(
                                        'scene_start_count', constants.DEFAULT_START_SCENES_COUNT
                                    )
                                    scene_end_count = st.session_state.get(
                                        'scene_end_count', constants.DEFAULT_END_SCENES_COUNT
                                    )

                                    scenes, status, error_msg = scene_detection.identify_candidate_scenes(
                                        video_path_obj,
                                        episode_id_proc,
                                        ocr_reader,
                                        current_ocr_engine,
                                        user_stopwords,
                                        scene_counts=(scene_start_count, scene_end_count),
                                    )
                                st.session_state.episode_status[episode_id_proc]['step1_status'] = status
                                if error_msg:
                                    st.session_state.episode_status[episode_id_proc]['step1_error'] = error_msg
                                    st.error(f"Error in Step 1 ({episode_id_proc}): {error_msg}")
                                else:
                                    st.success(
                                        f"Step 1 ({episode_id_proc}) -> Status: {status}. Found {len(scenes)} candidate scenes."
                                    )
                                logging.info(
                                    f"[{episode_id_proc}] Step 1 status: {status}. Scenes: {len(scenes)}. Error: {error_msg}"
                                )
                            except Exception as e:
                                st.session_state.episode_status[episode_id_proc]['step1_status'] = "error"
                                st.session_state.episode_status[episode_id_proc]['step1_error'] = str(e)
                                st.error(f"Exception in Step 1 ({episode_id_proc}): {e}")
                                logging.error(f"Exception during Step 1 for {episode_id_proc}: {e}", exc_info=True)
                
                st.success(
                    "Step 1 processing finished for selected videos. Review candidate scenes in '2a. Review Candidate Scenes' section if available."
                )

    if run_step2_button:
        # Set the step2_running flag immediately to hide scene review
        st.session_state.step2_running = True        
        if not selected_videos_str_paths:
            st.warning("Please select at least one video for Step 2.")
            st.session_state.step2_running = False  # Reset flag if no videos selected
        else:
            st.subheader("Running Step 2: Analyze Candidate Scene Frames")
            
            # Force refresh OCR reader to prevent model caching issues
            st.info("🔄 Refreshing OCR reader to prevent model caching issues...")
            ocr_reader = force_refresh_ocr_reader()
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

                scene_analysis_file = (
                    config.EPISODES_BASE_DIR / episode_id_proc / "analysis" / "initial_scene_analysis.json"
                )
                if not scene_analysis_file.exists():
                    st.warning(
                        f"Step 1 output (initial_scene_analysis.json) not found for {episode_id_proc}. Please run Step 1 first."
                    )
                    st.session_state.episode_status[episode_id_proc]['step2_status'] = "skipped_no_step1_output"
                    continue

                user_selected_indices_for_ep = st.session_state.user_selected_scenes_for_step2.get(episode_id_proc)

                if user_selected_indices_for_ep is None:
                    st.warning(
                        f"Scene selection for {episode_id_proc} not found. Please run Step 1 and review scenes first, or ensure review step completed."
                    )
                    logging.warning(
                        f"[{episode_id_proc}] User selected scenes not found in session state. Attempting to load all from Step 1 output for Step 2."
                    )
                    try:
                        with open(scene_analysis_file, 'r', encoding='utf-8') as f_scenes_fallback:
                            step1_data_fallback = json.load(f_scenes_fallback)
                        all_candidate_scenes_from_file = step1_data_fallback.get("candidate_scenes", [])
                        scenes_to_process_for_step2 = all_candidate_scenes_from_file
                    except Exception as e_fallback_load:
                        st.error(f"Failed to load scenes for {episode_id_proc} as fallback: {e_fallback_load}")
                        scenes_to_process_for_step2 = []
                elif not user_selected_indices_for_ep:
                    st.info(f"No scenes selected by user for {episode_id_proc} for Step 2 processing. Skipping.")
                    st.session_state.episode_status[episode_id_proc]['step2_status'] = "skipped_no_user_selection"
                    scenes_to_process_for_step2 = []
                else:
                    try:
                        with open(scene_analysis_file, 'r', encoding='utf-8') as f_scenes_main:
                            step1_data_main = json.load(f_scenes_main)
                        all_candidate_scenes_from_file = step1_data_main.get("candidate_scenes", [])
                        scenes_to_process_for_step2 = [
                            all_candidate_scenes_from_file[i]
                            for i in user_selected_indices_for_ep
                            if i < len(all_candidate_scenes_from_file)
                        ]
                        st.info(
                            f"Processing {len(scenes_to_process_for_step2)} user-selected scenes for {episode_id_proc} in Step 2."
                        )
                    except Exception as e_load_selected:
                        st.error(f"Error loading selected scenes for {episode_id_proc}: {e_load_selected}")
                        scenes_to_process_for_step2 = []

                if not scenes_to_process_for_step2:
                    if st.session_state.episode_status[episode_id_proc].get('step2_status') not in [
                        "skipped_no_user_selection",
                        "skipped_no_step1_output",
                    ]:
                        st.info(f"No scenes to process for {episode_id_proc} in Step 2 after selection/loading.")
                        st.session_state.episode_status[episode_id_proc][
                            'step2_status'
                        ] = "skipped_no_scenes_to_process"
                    continue

                st.session_state.episode_status[episode_id_proc]['step2_status'] = "running"
                with st.expander(f"Step 2: {episode_id_proc}", expanded=True):
                    st.write(
                        f"Processing Step 2 for {episode_id_proc} (using {len(scenes_to_process_for_step2)} selected/loaded scenes)..."
                    )
                    with st.spinner(f"Analyzing frames for {episode_id_proc}..."):
                        final_status_msg_step2 = "unknown"
                        final_error_detail_step2 = None
                        try:
                            logging.info(
                                f"[{episode_id_proc}] Using manual scene loop for Step 2 with {len(scenes_to_process_for_step2)} scenes."
                            )

                            if not scenes_to_process_for_step2:
                                st.info(f"No candidate scenes available for {episode_id_proc} to process in Step 2.")
                                final_status_msg_step2 = "skipped_no_scenes_after_selection"
                            else:
                                video_stream = open_video(str(video_path_obj))
                                fps = video_stream.frame_rate
                                frame_height = video_stream.frame_size[1]
                                frame_width = video_stream.frame_size[0]
                                episode_saved_texts_cache = []
                                episode_saved_files_cache = []
                                logging.info(f"[{episode_id_proc}] Initialized episode-level caches for Step 2")

                                ep_global_last_saved_ocr_text = None
                                ep_global_last_saved_frame_hash = None
                                ep_global_last_saved_ocr_bbox = None

                                num_scenes_processed_step2 = 0
                                for scene_info_item_step2 in scenes_to_process_for_step2:
                                    logging.info(
                                        f"Analyzing scene {scene_info_item_step2.get('scene_index', 'N/A')} for {episode_id_proc} in Step 2"
                                    )
                                    if not all(
                                        k in scene_info_item_step2
                                        for k in ['original_start_frame', 'original_end_frame']
                                    ):
                                        logging.warning(
                                            f"Skipping scene in Step 2 due to missing keys: {scene_info_item_step2}"
                                        )
                                        continue

                                    compatible_scene_info = {
                                        **scene_info_item_step2,
                                        'start_frame': scene_info_item_step2['original_start_frame'],
                                        'end_frame': scene_info_item_step2['original_end_frame'],
                                    }

                                    (
                                        analysis_result,
                                        ep_global_last_saved_ocr_text,
                                        ep_global_last_saved_frame_hash,
                                        ep_global_last_saved_ocr_bbox,
                                    ) = frame_analysis.analyze_candidate_scene_frames(
                                        video_path=video_path_obj,
                                        episode_id=episode_id_proc,
                                        scene_info=compatible_scene_info,
                                        fps=fps,
                                        frame_height=frame_height,
                                        frame_width=frame_width,
                                        ocr_reader=ocr_reader,
                                        ocr_engine_type=current_ocr_engine,
                                        user_stopwords=user_stopwords,
                                        global_last_saved_ocr_text_input=ep_global_last_saved_ocr_text,
                                        global_last_saved_frame_hash_input=ep_global_last_saved_frame_hash,
                                        global_last_saved_ocr_bbox_input=ep_global_last_saved_ocr_bbox,
                                        episode_saved_texts_cache=episode_saved_texts_cache,
                                        episode_saved_files_cache=episode_saved_files_cache,
                                    )

                                    logging.info(
                                        f"[{episode_id_proc}] Episode cache now contains {len(episode_saved_texts_cache)} saved texts after scene {scene_info_item_step2.get('scene_index', 'N/A')}"
                                    )

                                    try:
                                        analysis_dir = config.EPISODES_BASE_DIR / episode_id_proc / 'analysis'
                                        analysis_dir.mkdir(parents=True, exist_ok=True)
                                        manifest_path = analysis_dir / 'analysis_manifest.json'
                                        if manifest_path.is_file():
                                            with open(manifest_path, 'r', encoding='utf-8') as mf:
                                                manifest_data = json.load(mf)
                                        else:
                                            manifest_data = {'scenes': {}}
                                    except Exception as mf_err:
                                        logging.error(
                                            f"[{episode_id_proc}] Failed to load Step 2 manifest: {mf_err}",
                                            exc_info=True,
                                        )
                                        manifest_data = {'scenes': {}}

                                    scene_index = scene_info_item_step2.get('scene_index')
                                    scene_key = (
                                        f"scene_{scene_index}"
                                        if scene_index is not None
                                        else f"scene_{episode_id_proc}_{num_scenes_processed_step2}"
                                    )
                                    manifest_data.setdefault('scenes', {})[scene_key] = analysis_result

                                    try:
                                        with open(manifest_path, 'w', encoding='utf-8') as mf:
                                            json.dump(manifest_data, mf, indent=2)
                                        logging.debug(
                                            f"[{episode_id_proc}] Updated Step 2 manifest with scene {scene_key}"
                                        )
                                    except Exception as write_err:
                                        logging.error(
                                            f"[{episode_id_proc}] Failed to write Step 2 manifest: {write_err}",
                                            exc_info=True,
                                        )
                                    num_scenes_processed_step2 += 1

                                st.success(
                                    f"Step 2 ({episode_id_proc}) completed. Processed {num_scenes_processed_step2} scenes with episode-level deduplication."
                                )
                                logging.info(
                                    f"[{episode_id_proc}] Step 2 completed with final episode cache containing {len(episode_saved_texts_cache)} unique texts"
                                )

                            st.session_state.episode_status[episode_id_proc]['step2_status'] = final_status_msg_step2
                            if final_error_detail_step2:
                                st.session_state.episode_status[episode_id_proc][
                                    'step2_error'
                                ] = final_error_detail_step2
                                st.warning(f"Step 2 ({episode_id_proc}): {final_error_detail_step2}")

                        except Exception as e:
                            st.session_state.episode_status[episode_id_proc]['step2_status'] = "error"
                            st.session_state.episode_status[episode_id_proc]['step2_error'] = str(e)
                            st.error(f"Exception in Step 2 ({episode_id_proc}): {e}")
                            logging.error(f"Exception during Step 2 for {episode_id_proc}: {e}", exc_info=True)

        # Reset the step2_running flag when Step 2 completes
        st.session_state.step2_running = False
        st.info("Step 2 processing finished for selected videos.")

    if run_step3_button:
        if not selected_videos_str_paths:
            st.warning("Please select at least one video for Step 3.")
        else:
            st.subheader("Running Step 3: Azure VLM OCR")
            for video_path_str_proc in selected_videos_str_paths:
                video_path_obj = Path(video_path_str_proc)
                episode_id_proc = video_path_obj.stem

                if episode_id_proc not in st.session_state.episode_status:
                    st.session_state.episode_status[episode_id_proc] = {}

                frames_dir_for_vlm = config.EPISODES_BASE_DIR / episode_id_proc / "analysis" / "frames"
                
                if not frames_dir_for_vlm.is_dir() or not any(frames_dir_for_vlm.iterdir()):
                    st.warning(
                        f"No frames found from Step 2 for {episode_id_proc} in {frames_dir_for_vlm}. VLM step might have no input."
                    )
                
                with st.expander(f"Step 3: {episode_id_proc}", expanded=True):
                    st.write(f"Processing Step 3 for {episode_id_proc}...")
                    
                    with st.spinner(f"Running Azure VLM for {episode_id_proc}..."):
                        try:
                            # Role groups are now defined directly in config.py
                            count, status, err_msg = azure_vlm_processing.run_azure_vlm_ocr_on_frames(
                                episode_id_proc, constants.DEFAULT_VLM_MAX_NEW_TOKENS
                            )

                            if status == "completed" and not err_msg:
                                try:
                                    ocr_dir = config.EPISODES_BASE_DIR / episode_id_proc / "ocr"
                                    vlm_json_path = ocr_dir / f"{episode_id_proc}_credits_azure_vlm.json"

                                    if vlm_json_path.exists():
                                        vlm_credits = utils.load_vlm_results_from_jsonl(vlm_json_path)
                                        if vlm_credits:
                                            success, db_msg = utils.save_credits(episode_id_proc, vlm_credits)
                                            if success:
                                                logging.info(
                                                    f"[{episode_id_proc}] Successfully saved {len(vlm_credits)} credits to database."
                                                )
                                            else:
                                                logging.error(
                                                    f"[{episode_id_proc}] Failed to save credits to database: {db_msg}"
                                                )
                                                st.warning(f"VLM completed but database save failed: {db_msg}")
                                        else:
                                            logging.warning(f"[{episode_id_proc}] No credits found in VLM results file.")
                                    else:
                                        logging.warning(f"[{episode_id_proc}] VLM results file not found: {vlm_json_path}")
                                except Exception as db_save_err:
                                    logging.error(
                                        f"[{episode_id_proc}] Error saving VLM results to database: {db_save_err}",
                                        exc_info=True,
                                    )
                                    st.warning(f"VLM completed but database save failed: {db_save_err}")

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

    if run_step4_button:
        if not selected_videos_str_paths:
            st.warning("Please select at least one video for Step 4.")
        else:
            st.subheader("Running Step 4: IMDB Validation")
            
            # Import the batch validator
            from scripts_v3.imdb_batch_validation import IMDBBatchValidator
            
            for video_path_str_proc in selected_videos_str_paths:
                video_path_obj = Path(video_path_str_proc)
                episode_id_proc = video_path_obj.stem
                if episode_id_proc not in st.session_state.episode_status:
                    st.session_state.episode_status[episode_id_proc] = {}
                
                try:
                    with st.expander(f"Step 4 IMDB Validation: {episode_id_proc}", expanded=True):
                        st.write(f"🔍 Validating credits against IMDB database for episode: {episode_id_proc}")
                        
                        # Create batch validator
                        batch_validator = IMDBBatchValidator()
                        
                        # Get credits to process for this episode
                        credits = batch_validator.get_unprocessed_credits(episode_id=episode_id_proc)
                        
                        if not credits:
                            st.success(f"✅ No credits need IMDB validation for {episode_id_proc}")
                            st.session_state.episode_status[episode_id_proc]['step4_complete'] = True
                            continue
                        
                        st.write(f"📊 Found {len(credits)} credits to validate for {episode_id_proc}")
                        
                        # Process credits with progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        total_credits = len(credits)
                        processed = 0
                        
                        for i, credit in enumerate(credits):
                            credit_id = credit['id']
                            name = credit.get('name', '')
                            is_person = credit.get('is_person')
                            
                            # Update progress
                            progress = (i + 1) / total_credits
                            progress_bar.progress(progress)
                            status_text.text(f"Processing {i+1}/{total_credits}: {name}")
                            
                            # Skip companies
                            if is_person is False or is_person == 0:
                                batch_validator.update_imdb_status(credit_id, True)
                                batch_validator.stats['companies_skipped'] += 1
                            else:
                                # Validate person names
                                is_found = batch_validator.validate_credit(credit)
                                batch_validator.update_imdb_status(credit_id, is_found)
                                
                                if is_found:
                                    batch_validator.stats['found_in_imdb'] += 1
                                else:
                                    batch_validator.stats['not_found_in_imdb'] += 1
                                
                                batch_validator.stats['persons_processed'] += 1
                            
                            processed += 1
                        
                        # Show final results
                        progress_bar.progress(1.0)
                        status_text.text("IMDB validation complete!")
                        
                        stats = batch_validator.stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("✅ Found in IMDB", stats['found_in_imdb'])
                        with col2:
                            st.metric("❌ Not Found", stats['not_found_in_imdb'])
                        with col3:
                            st.metric("🏢 Companies", stats['companies_skipped'])
                        
                        if stats['persons_processed'] > 0:
                            match_rate = (stats['found_in_imdb'] / stats['persons_processed']) * 100
                            st.info(f"📈 IMDB match rate for persons: {match_rate:.1f}%")
                        
                        st.session_state.episode_status[episode_id_proc]['step4_complete'] = True
                        st.success(f"✅ Step 4 completed successfully for {episode_id_proc}")
                        
                except Exception as e:
                    st.session_state.episode_status[episode_id_proc]['step4_error'] = str(e)
                    st.error(f"Exception in Step 4 ({episode_id_proc}): {e}")
                    logging.error(f"Exception during Step 4 for {episode_id_proc}: {e}", exc_info=True)
            
            st.info("Step 4 IMDB validation finished for selected videos.")

    if run_all_steps_button:
        st.warning(
            "RUN ALL STEPS functionality needs to be implemented by calling Step 1, 2, and 3 in sequence for each selected video, with appropriate checks for success before proceeding to the next step. This is a complex workflow to manage in Streamlit's execution model and is left as a manual process for now (run steps individually)."
        )

elif st.session_state.current_tab == 1:    
    st.header("✏️ Review & Edit Credits")
    
    # Helper function to preserve review tab state
    def preserve_review_state(episode=None):
        """Set flags to preserve the review tab and current episode after rerun"""
        st.session_state.preserve_review_tab = True
        if episode:
            st.session_state.preserve_episode = episode
        else:
            # Clear episode preservation if not specified
            st.session_state.preserve_episode = None
    
    # Clear preserve flags when user enters review tab
    if st.session_state.get('preserve_review_tab'):
        st.session_state.preserve_review_tab = False

    try:
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute(f"SELECT DISTINCT episode_id FROM {config.DB_TABLE_CREDITS} ORDER BY episode_id")
        available_episodes = [row[0] for row in cursor.fetchall()]
        conn.close()

        if not available_episodes:
            st.info(
                "No episodes with credits found in the database. Please run the pipeline first to generate credits."
            )
        else:
            # Add checkbox to filter episodes that need reviewing
            col1, col2 = st.columns([3, 1])

            with col1:
                # Determine which episodes to show based on filter
                show_only_needing_review = st.checkbox(
                    "Show only episodes that need reviewing",
                    value=False,
                    key="filter_episodes_needing_review",
                    help="Filter to show only episodes with problematic credits that need human review",
                )

                if show_only_needing_review:                    # Filter episodes to show only those with problematic credits
                    episodes_with_problems = []
                    for episode in available_episodes:
                        problematic_count = utils.identify_problematic_credits_fast(episode)
                        if problematic_count > 0:
                            episodes_with_problems.append(f"{episode} ({problematic_count} issues)")

                    if not episodes_with_problems:
                        st.info("🎉 No episodes need reviewing! All credits have been processed.")
                        display_episodes = []
                        episode_options = []
                    else:
                        display_episodes = episodes_with_problems
                        episode_options = [ep.split(' (')[0] for ep in episodes_with_problems]  # Extract episode name
                else:                    # Show all episodes with issue counts
                    episodes_with_counts = []
                    episode_options = []
                    for episode in available_episodes:
                        problematic_count = utils.identify_problematic_credits_fast(episode)
                        if problematic_count > 0:
                            episodes_with_counts.append(f"{episode} ({problematic_count} issues)")
                        else:
                            episodes_with_counts.append(f"{episode} (✅ complete)")
                        episode_options.append(episode)
                    display_episodes = episodes_with_counts

            with col2:
                if st.button("🔄 Refresh Episodes"):
                    # Set flag to preserve review tab when refreshing episodes
                    st.session_state.preserve_review_tab = True
                    st.rerun()

            if display_episodes:
                # Determine default episode selection with robust logic
                default_index = 0
                selected_episode_name = None
                
                # Priority 1: Use preserved episode if available
                preserved_episode = st.session_state.get('preserve_episode')
                if preserved_episode and preserved_episode in episode_options:
                    try:
                        default_index = episode_options.index(preserved_episode)
                        selected_episode_name = preserved_episode
                        st.success(f"🎯 **Continuing work on**: {preserved_episode}")
                        # Clear the preserve flag after successful use
                        st.session_state.preserve_episode = None
                    except (ValueError, IndexError):
                        st.warning(f"Preserved episode '{preserved_episode}' no longer available")
                        st.session_state.preserve_episode = None
                
                # Priority 2: Use last selected episode if no preserved episode
                if not selected_episode_name:
                    last_episode = st.session_state.get('last_selected_episode')
                    if last_episode and last_episode in episode_options:
                        try:
                            default_index = episode_options.index(last_episode)
                            selected_episode_name = last_episode
                        except (ValueError, IndexError):
                            default_index = 0
                
                # Priority 3: Default to first episode
                if not selected_episode_name and episode_options:
                    selected_episode_name = episode_options[0]

                selected_display = st.selectbox(
                    "Select an episode to review:", 
                    options=display_episodes, 
                    index=default_index,
                    key="episode_selector_review"
                )                # Extract the actual episode name from the display string
                if show_only_needing_review:
                    selected_episode = selected_display.split(' (')[0] if selected_display else None
                else:
                    selected_episode = (
                        episode_options[display_episodes.index(selected_display)] if selected_display else None
                    )
                
                # Always update last selected episode for persistence
                if selected_episode:
                    st.session_state.last_selected_episode = selected_episode
            else:
                selected_episode = None

            if selected_episode:
                queue_key = f"credit_queue_{selected_episode}"
                index_key = f"credit_index_{selected_episode}"
                decisions_key = f"credit_decisions_{selected_episode}"

                if queue_key not in st.session_state:
                    initial_queue = utils.identify_problematic_credits(selected_episode)
                    st.session_state[queue_key] = initial_queue
                    st.session_state[index_key] = 0
                    st.session_state[decisions_key] = {}
                    logging.info(
                        f"[Queue Init] Initialized queue for {selected_episode} with {len(initial_queue)} problematic credits"
                    )
                else:
                    logging.info(
                        f"[Queue Init] Using existing queue for {selected_episode} with {len(st.session_state[queue_key])} credits"
                    )

                problematic_queue = st.session_state[queue_key]
                current_index = st.session_state[index_key]

                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    review_mode = st.selectbox(
                        "Review Mode:",
                        options=["🎯 Focus Mode (One at a time)", "📋 Overview Mode (All credits)"],
                        key="review_mode_selector",
                    )

                with col2:
                    if st.button("🔄 Refresh Queue"):
                        st.session_state[queue_key] = utils.identify_problematic_credits(selected_episode)
                        st.session_state[index_key] = 0
                        # Reset the original total when queue is refreshed
                        if f"{queue_key}_original" in st.session_state:
                            del st.session_state[f"{queue_key}_original"]
                          # Set flags to preserve review tab and episode state
                        st.session_state.preserve_review_tab = True
                        st.session_state.preserve_episode = selected_episode
                        
                        st.success("Queue refreshed!")
                        st.rerun()

                with col3:
                    if st.button("🔓 Reset Reviews", help="Reset all 'kept' status for this episode"):
                        try:
                            conn = sqlite3.connect(config.DB_PATH)
                            cursor = conn.cursor()
                            cursor.execute(
                                f"""
                                UPDATE {config.DB_TABLE_CREDITS}
                                SET reviewed_status = 'pending', reviewed_at = NULL
                                WHERE episode_id = ? AND reviewed_status = 'kept'
                            """,
                                (selected_episode,),
                            )
                            affected_rows = cursor.rowcount
                            conn.commit()
                            conn.close()                            # Refresh the queue
                            st.session_state[queue_key] = utils.identify_problematic_credits(selected_episode)
                            st.session_state[index_key] = 0
                            # Reset the original total when queue is refreshed
                            if f"{queue_key}_original" in st.session_state:
                                del st.session_state[f"{queue_key}_original"]
                            
                            st.success(f"Reset review status for {affected_rows} credits. Queue refreshed.")
                            
                            # Set flags to preserve review tab and episode state
                            st.session_state.preserve_review_tab = True
                            st.session_state.preserve_episode = selected_episode
                            st.rerun()

                        except Exception as e:
                            st.error(f"Error resetting review status: {e}")

                if review_mode.startswith("🎯 Focus Mode") and problematic_queue:
                    total_problematic = len(problematic_queue)

                    if f"{queue_key}_original" not in st.session_state:
                        st.session_state[f"{queue_key}_original"] = problematic_queue.copy()

                    original_total = len(st.session_state[f"{queue_key}_original"])
                    resolved_count = max(0, original_total - total_problematic)  # Ensure non-negative
                    progress = resolved_count / original_total if original_total > 0 else 1.0
                    progress = max(0.0, min(1.0, progress))  # Clamp between 0.0 and 1.0
                    st.progress(
                        progress,
                        text=f"Progress: {resolved_count}/{original_total} resolved • {total_problematic} remaining",
                    )

                    if current_index >= total_problematic and total_problematic > 0:
                        st.session_state[index_key] = total_problematic - 1
                        current_index = total_problematic - 1

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🎯 Current", f"{current_index + 1}/{total_problematic}")
                    with col2:
                        completed = len([d for d in st.session_state[decisions_key].values() if d != 'skip'])
                        st.metric("✅ Resolved", completed)
                    with col3:
                        deleted = len([d for d in st.session_state[decisions_key].values() if d == 'delete'])
                        st.metric("🗑️ Deleted", deleted)
                    with col4:
                        remaining = total_problematic - current_index
                        st.metric("⏳ Remaining", remaining)

                    if current_index < total_problematic:
                        current_credit = problematic_queue[current_index]
                        logging.info(
                            f"[UI Display] Displaying credit at index {current_index}: {current_credit['name']} (ID: {current_credit['id']})"
                        )
                        st.markdown("---")

                        problem_desc = utils.format_problem_description(current_credit['problem_types'])
                        st.subheader(f"👤 {current_credit['name']}")
                        st.error(f"**Issues:** {problem_desc}")

                        is_duplicate = current_credit.get('total_variants', 1) > 1
                        duplicate_entries = current_credit.get('duplicate_entries', [current_credit])

                        if is_duplicate:
                            st.warning(
                                f"**{current_credit['name']}** appears {len(duplicate_entries)} times with different details."
                            )
                            st.markdown("### 📋 Compare All Variants:")

                            variant_data = []
                            for i, entry in enumerate(duplicate_entries):
                                variant_data.append(
                                    {
                                        'Variant': f"#{i+1}",
                                        'Role Group': entry['role_group'],
                                        'Role Detail': entry.get('role_detail', 'N/A'),
                                        'Scene Position': entry.get('scene_position', 'unknown'),
                                        'Source Frames': len(entry.get('source_frame', '').split(','))
                                        if entry.get('source_frame')
                                        else 0,
                                    }
                                )

                            import pandas as pd

                            comparison_df = pd.DataFrame(variant_data)
                            st.dataframe(comparison_df, use_container_width=True)

                            st.markdown("### 🔍 Detailed View:")
                            tab_names = [
                                f"Variant {i+1}: {entry['role_group']}" for i, entry in enumerate(duplicate_entries)
                            ]
                            tabs = st.tabs(tab_names)

                            for i, (tab, entry) in enumerate(zip(tabs, duplicate_entries)):
                                with tab:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write(f"**Name:** {entry['name']}")
                                        st.write(f"**Role Group:** {entry['role_group']}")
                                        st.write(f"**Role Detail:** {entry.get('role_detail', 'N/A')}")
                                        st.write(f"**Scene Position:** {entry.get('scene_position', 'unknown')}")

                                    with col2:
                                        source_frames = entry.get('source_frame', '')
                                        if source_frames:
                                            frame_list = [f.strip() for f in source_frames.split(',') if f.strip()]
                                            st.write(f"**Source Frames ({len(frame_list)}):**")
                                            for frame in frame_list[:3]:
                                                st.caption(f"• {frame}")
                                            if len(frame_list) > 3:
                                                st.caption(f"... and {len(frame_list) - 3} more")
                                        else:
                                            st.write("**Source Frames:** None")

                                    if source_frames:
                                        best_frames = utils.get_best_frames_for_credit(entry, max_frames=2)
                                        if best_frames:
                                            st.write("**Sample Frames:**")
                                            for j, frame_filename in enumerate(best_frames):
                                                frame_path = utils.find_frame_path(selected_episode, frame_filename)
                                                if frame_path and frame_path.exists():
                                                    st.image(
                                                        str(frame_path),
                                                        caption=f"Frame: {frame_filename}",
                                                        use_container_width=True,
                                                    )
                                                else:
                                                    st.caption(f"❌ {frame_filename}")

                        else:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Role Group:** {current_credit['role_group']}")
                                st.write(f"**Role Detail:** {current_credit.get('role_detail', 'N/A')}")
                            with col2:
                                st.write(f"**Scene Position:** {current_credit.get('scene_position', 'unknown')}")

                            st.markdown("### 🖼️ Source Frame(s)")
                            best_frames = utils.get_best_frames_for_credit(current_credit, max_frames=2)

                            if best_frames:
                                frame_cols = st.columns(len(best_frames))
                                for i, frame_filename in enumerate(best_frames):
                                    with frame_cols[i]:
                                        frame_path = utils.find_frame_path(selected_episode, frame_filename)
                                        if frame_path and frame_path.exists():
                                            st.image(str(frame_path), caption=f"Frame: {frame_filename}")
                                        else:
                                            st.error(f"Frame not found: {frame_filename}")
                            else:
                                st.warning("No source frames available for this credit")                        # Auto-open the variants modal - no buttons needed
                        st.markdown("---")
                        st.markdown("### ✏️ Edit/Delete Variants")
                        
                        # Initialize state for new entries if not exists
                        new_entries_key = f"new_entries_{current_credit['id']}"
                        if new_entries_key not in st.session_state:
                            st.session_state[new_entries_key] = []
                        
                        # Show helpful info about what can be done
                        new_entries_count = len(st.session_state[new_entries_key])
                        if new_entries_count > 0:
                            st.info(
                                f"Edit each variant individually, delete unwanted ones, or add new entries. You have {new_entries_count} new entry(ies) ready to be added. Click 'Save All Changes' when done."
                            )
                        else:
                            st.info(
                                "Edit each variant individually, delete unwanted ones, or add new entries. Click 'Save All Changes' when done."
                            )                        # Collect available frames from existing entries for new entry creation
                        available_frames = []
                        for entry in duplicate_entries:
                            source_frames = entry.get('source_frame', '')
                            if source_frames:
                                frame_list = [f.strip() for f in source_frames.split(',') if f.strip()]
                                available_frames.extend(frame_list)
                        
                        # Remove duplicates and sort
                        available_frames = sorted(list(set(available_frames)))                        # Frame selection for new entries (if there are available frames)
                        selected_frame = None
                        if available_frames:
                            st.markdown("**Frame Selection for New Entries:**")
                            if len(available_frames) == 1:
                                selected_frame = available_frames[0]
                                st.info(f"New entries will use frame: **{selected_frame}**")
                            else:
                                frame_selection_key = f"frame_selection_{current_credit['id']}"
                                selected_frame = st.selectbox(
                                    "Choose frame for new entries:",
                                    options=available_frames,
                                    key=frame_selection_key,
                                    help="Select which frame new entries should be associated with"
                                )

                        # Add button to create new entries (outside the form to allow state changes)
                        col_add, col_space = st.columns([2, 6])
                        with col_add:
                            button_type = "primary" if len(st.session_state[new_entries_key]) == 0 else "secondary"
                            button_text = "➕ Add New Entry" if len(st.session_state[new_entries_key]) == 0 else f"➕ Add Another ({len(st.session_state[new_entries_key])} pending)"
                            if st.button(button_text, key=f"add_entry_{current_credit['id']}", type=button_type):
                                # Add a new empty entry
                                new_entry = {
                                    'id': f"new_{len(st.session_state[new_entries_key])}_{current_credit['id']}",
                                    'name': current_credit['name'],  # Default to the main credit name
                                    'role_group': duplicate_entries[0]['role_group'] if duplicate_entries else '',
                                    'role_detail': '',
                                    'episode_id': current_credit['episode_id'],
                                    'is_person': True,  # Default to person
                                    'is_new': True,
                                    'source_frame': selected_frame or 'manual_entry'  # Use selected frame or default
                                }
                                st.session_state[new_entries_key].append(new_entry)
                                st.rerun()

                        with st.form(f"form_edit_variants_{current_credit['id']}"):
                            # Collect form inputs for all variants (existing + new)
                            form_data = []
                            
                            # Combine existing entries with new entries
                            all_entries = duplicate_entries + st.session_state[new_entries_key]
                            
                            for i, entry in enumerate(all_entries):
                                is_new = entry.get('is_new', False)
                                entry_type = "New Entry" if is_new else f"Variant {i+1 - len(st.session_state[new_entries_key])}"
                                if is_new:
                                    # For new entries, show them after existing variants
                                    existing_count = len(duplicate_entries)
                                    new_index = i - existing_count + 1
                                    entry_type = f"New Entry #{new_index}"
                                else:
                                    entry_type = f"Variant {i+1}"
                                    
                                # Add visual distinction for new entries
                                if is_new:
                                    st.markdown(f"#### 🆕 {entry_type}")
                                    st.markdown("*This is a new entry that will be added to the database*")
                                else:
                                    st.markdown(f"#### {entry_type}")
                                    
                                col1, col2, col3, col4, col5 = st.columns([2.5, 2.5, 2.5, 1.5, 1])

                                with col1:
                                    name = st.text_input(f"Name", value=entry['name'], key=f"variant_name_{i}")

                                with col2:
                                    # Get available roles
                                    conn = sqlite3.connect(config.DB_PATH)
                                    cursor = conn.cursor()
                                    cursor.execute(
                                        f"SELECT DISTINCT role_group_normalized FROM {config.DB_TABLE_CREDITS} ORDER BY role_group_normalized"
                                    )
                                    all_roles = [row[0] for row in cursor.fetchall()]
                                    conn.close()

                                    current_role_index = (
                                        all_roles.index(entry['role_group']) if entry['role_group'] in all_roles else 0
                                    )
                                    role_group = st.selectbox(
                                        f"Role Group",
                                        options=all_roles,
                                        index=current_role_index,
                                        key=f"variant_role_{i}",
                                    )

                                with col3:
                                    role_detail = st.text_input(
                                        f"Role Detail", value=entry.get('role_detail', ''), key=f"variant_detail_{i}"
                                    )

                                with col4:
                                    # Company/Person selector
                                    is_person_current = entry.get('is_person', True)  # Default to person
                                    entity_type = st.selectbox(
                                        "Type",
                                        options=["Person", "Company"],
                                        index=0 if is_person_current else 1,
                                        key=f"variant_type_{i}",
                                        help="Select whether this is a person or company"
                                    )
                                    is_person = (entity_type == "Person")

                                with col5:
                                    if is_new:
                                        # For new entries, show a remove checkbox
                                        delete_variant = st.checkbox(
                                            "❌", 
                                            key=f"remove_new_{i}", 
                                            help="Check to remove this new entry (won't be saved)"
                                        )
                                    else:                                        # For existing entries, show delete checkbox
                                        delete_variant = st.checkbox(
                                            "🗑️", key=f"variant_delete_{i}", help="Check to delete this variant"
                                        )                                # Store form data for processing on submit
                                form_data.append(
                                    {
                                        'id': entry['id'],
                                        'name': name,
                                        'role_group': role_group,
                                        'role_detail': role_detail,
                                        'is_person': is_person,
                                        'delete': delete_variant,
                                        'is_new': is_new,
                                        'episode_id': entry.get('episode_id', current_credit['episode_id']),
                                        'source_frame': entry.get('source_frame', 'manual_entry')
                                    }
                                )                                # Show source frames for this variant
                                if not is_new:
                                    source_frames = entry.get('source_frame', '')
                                    if source_frames:
                                        frame_list = [f.strip() for f in source_frames.split(',') if f.strip()]
                                        st.caption(f"**Source Frames ({len(frame_list)}):** {', '.join(frame_list[:3])}")
                                        if len(frame_list) > 3:
                                            st.caption(f"... and {len(frame_list) - 3} more")
                                    else:
                                        st.caption("**Source Frames:** None")
                                else:
                                    # For new entries, show the selected source frame
                                    new_source_frame = entry.get('source_frame', 'manual_entry')
                                    if new_source_frame and new_source_frame != 'manual_entry':
                                        st.caption(f"**Source Frame:** {new_source_frame}")
                                    else:
                                        st.caption("**Source Frame:** Will be set when saved")

                            col1, col2 = st.columns(2)
                            with col1:
                                if st.form_submit_button("💾 Save All Changes", type="primary"):
                                    try:
                                        # Build the actual lists based on current form values
                                        variants_to_delete = []
                                        edited_variants = []
                                        new_variants = []
                                        removed_new_entries = 0

                                        for variant_data in form_data:
                                            if variant_data['delete']:
                                                # Only add to delete list if it's not a new entry
                                                if not variant_data.get('is_new', False):
                                                    variants_to_delete.append(variant_data['id'])
                                                else:
                                                    # Count removed new entries
                                                    removed_new_entries += 1
                                            else:
                                                if variant_data.get('is_new', False):
                                                    # This is a new entry to be inserted
                                                    new_variants.append(
                                                        {
                                                            'name': variant_data['name'],
                                                            'role_group': variant_data['role_group'],
                                                            'role_detail': variant_data['role_detail'],
                                                            'episode_id': variant_data['episode_id'],
                                                            'is_person': variant_data['is_person'],
                                                            'source_frame': variant_data['source_frame']
                                                        }
                                                    )
                                                else:
                                                    # This is an existing entry to be updated
                                                    edited_variants.append(
                                                        {
                                                            'id': variant_data['id'],
                                                            'name': variant_data['name'],
                                                            'role_group': variant_data['role_group'],
                                                            'role_detail': variant_data['role_detail'],
                                                            'is_person': variant_data['is_person']
                                                        }
                                                    )

                                        conn = sqlite3.connect(config.DB_PATH)
                                        cursor = conn.cursor()

                                        # Delete marked variants first (only existing ones)
                                        for variant_id in variants_to_delete:
                                            cursor.execute(
                                                f"DELETE FROM {config.DB_TABLE_CREDITS} WHERE id = ?", (variant_id,)                                            )
                                            
                                        # Insert new variants
                                        for new_variant in new_variants:
                                            cursor.execute(
                                                f"""
                                                INSERT INTO {config.DB_TABLE_CREDITS} 
                                                (episode_id, name, role_group_normalized, role_detail, scene_position, source_frame, reviewed_status, reviewed_at, is_person)
                                                VALUES (?, ?, ?, ?, 'manual_entry', ?, 'kept', CURRENT_TIMESTAMP, ?)
                                            """,
                                                (
                                                    new_variant['episode_id'],
                                                    new_variant['name'],
                                                    new_variant['role_group'],
                                                    new_variant['role_detail'],
                                                    new_variant['source_frame'],
                                                    new_variant['is_person']
                                                ),
                                            )
                                            
                                        # Update remaining existing variants
                                        for variant in edited_variants:
                                            cursor.execute(
                                                f"""
                                                UPDATE {config.DB_TABLE_CREDITS}
                                                SET name = ?, role_group_normalized = ?, role_detail = ?, is_person = ?,
                                                    reviewed_status = 'kept', reviewed_at = CURRENT_TIMESTAMP
                                                WHERE id = ?
                                            """,
                                                (
                                                    variant['name'],
                                                    variant['role_group'],
                                                    variant['role_detail'],
                                                    variant['is_person'],
                                                    variant['id'],
                                                ),
                                            )

                                        conn.commit()
                                        conn.close()
                                        
                                        # Update session state: remove processed new entries
                                        # Keep only new entries that weren't processed (neither saved nor removed)
                                        remaining_new_entries = []
                                        for i, entry in enumerate(st.session_state[new_entries_key]):
                                            # Find the corresponding form data
                                            form_entry_index = len(duplicate_entries) + i
                                            if form_entry_index < len(form_data):
                                                form_entry = form_data[form_entry_index]
                                                # If it wasn't marked for deletion, keep it for further editing
                                                if not form_entry['delete']:
                                                    remaining_new_entries.append(entry)
                                        st.session_state[new_entries_key] = remaining_new_entries
                                        
                                        # Check if any variants remain
                                        total_remaining = len(edited_variants) + len(new_variants)
                                        if total_remaining == 0:
                                            # All variants were deleted
                                            success_msg = f"🗑️ All variants deleted successfully!"
                                            decision_type = 'delete_all'
                                        else:
                                            # Some variants remain
                                            updates_msg = f"{len(edited_variants)} updated" if edited_variants else ""
                                            adds_msg = f"{len(new_variants)} added" if new_variants else ""
                                            deletes_msg = f"{len(variants_to_delete)} deleted" if variants_to_delete else ""
                                            removes_msg = f"{removed_new_entries} new entries removed" if removed_new_entries else ""
                                            
                                            actions = [msg for msg in [updates_msg, adds_msg, deletes_msg, removes_msg] if msg]
                                            success_msg = f"💾 Changes saved! {', '.join(actions)}."
                                            decision_type = 'variants_edited'
                                            
                                        # Remove from queue and update navigation (for all cases)
                                        # Get fresh references directly from session state
                                        problematic_queue = st.session_state[queue_key]
                                        current_index = st.session_state[index_key]
                                        # Get the current credit from the fresh queue/index
                                        if current_index < len(problematic_queue):
                                            current_credit_fresh = problematic_queue[current_index]
                                            logging.info(
                                                f"[Variants Save] Processing credit: {current_credit_fresh['name']} (ID: {current_credit_fresh['id']}) at index {current_index}"
                                            )

                                            # Log all duplicate entries in this group
                                            duplicate_entries_fresh = current_credit_fresh.get('duplicate_entries', [])
                                            logging.info(
                                                f"[Variants Save] This credit group has {len(duplicate_entries_fresh)} variants:"
                                            )
                                            for i, entry in enumerate(duplicate_entries_fresh):
                                                logging.info(
                                                    f"[Variants Save]   Variant {i+1}: {entry['name']} (ID: {entry['id']}) - Role: {entry['role_group']}"
                                                )

                                            # Log what we're planning to do
                                            logging.info(
                                                f"[Variants Save] Planning to delete {len(variants_to_delete)} variants, keep {len(edited_variants)} variants"
                                            )
                                            logging.info(f"[Variants Save] Delete IDs: {variants_to_delete}")
                                            logging.info(
                                                f"[Variants Save] Keep IDs: {[v['id'] for v in edited_variants]}"
                                            )

                                            # Record the decision using the fresh credit ID
                                            st.session_state[decisions_key][current_credit_fresh['id']] = decision_type

                                            # Remove the current problematic credit group from queue
                                            problematic_queue.pop(current_index)
                                            st.session_state[queue_key] = problematic_queue
                                            logging.info(
                                                f"[Variants Save] Removed credit from queue. New queue length: {len(problematic_queue)}"
                                            )

                                            # Update index to automatically show next credit
                                            if len(problematic_queue) == 0:
                                                # No more credits to review
                                                st.session_state[index_key] = 0
                                                logging.info(f"[Variants Save] No more credits - set index to 0")
                                            elif current_index >= len(problematic_queue):
                                                # We were at the last credit, go to the new last credit
                                                new_index = len(problematic_queue) - 1
                                                st.session_state[index_key] = new_index
                                                logging.info(
                                                    f"[Variants Save] Was at last credit - set index to {new_index}"
                                                )
                                            else:                                                logging.info(
                                                    f"[Variants Save] Staying at index {current_index} - next credit will appear here"
                                                )
                                            # If current_index < len(problematic_queue), stay at current_index
                                            # because the next credit automatically moves into this position
                                        else:
                                            logging.error(
                                                f"[Variants Save] Current index {current_index} is out of bounds for queue length {len(problematic_queue)}"
                                            )

                                        st.success(success_msg)

                                        # Set flags to preserve review tab and episode state
                                        st.session_state.preserve_review_tab = True
                                        
                                        # Only preserve episode if there are still credits to review
                                        if len(st.session_state[queue_key]) > 0:
                                            st.session_state.preserve_episode = selected_episode
                                            logging.info(f"[Variants Save] Preserving episode {selected_episode} - {len(st.session_state[queue_key])} credits remaining")
                                        else:
                                            st.session_state.preserve_episode = None
                                            logging.info(f"[Variants Save] All credits completed for {selected_episode} - not preserving episode")
                                        
                                        # Log completion
                                        logging.info(
                                            f"[Variants Save] Completed processing. Queue now has {len(st.session_state[queue_key])} credits, index is {st.session_state[index_key]}"
                                        )
                                          # Always rerun to show next credit or completion
                                        if len(st.session_state[queue_key]) == 0:
                                            st.info("🎉 All credits reviewed!")
                                        else:
                                            st.info("✅ Moving to next credit...")
                                        
                                        # Force rerun to show updated state and navigate to review tab
                                        st.rerun()

                                    except Exception as e:
                                        st.error(f"Error saving changes: {e}")

                            with col2:
                                if st.form_submit_button("⏭️ Skip All Variants"):
                                    st.session_state[decisions_key][current_credit['id']] = 'skip'
                                    if current_index < len(problematic_queue) - 1:
                                        st.session_state[index_key] += 1
                                    else:
                                        st.session_state[index_key] = 0
                                      # Set flags to preserve review tab and episode state
                                    st.session_state.preserve_review_tab = True
                                    
                                    # Always preserve episode when skipping (there are still credits)
                                    st.session_state.preserve_episode = selected_episode
                                    
                                    st.success("⏭️ Skipped - moved to next credit")
                                    st.rerun()

                    else:
                        st.success("🎉 All problematic credits have been reviewed!")
                        decisions = st.session_state[decisions_key]
                        if decisions:
                            st.subheader("📊 Review Summary")
                            decision_counts = {}
                            for decision in decisions.values():
                                decision_counts[decision] = decision_counts.get(decision, 0) + 1

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("✅ Kept", decision_counts.get('keep', 0))
                            with col2:
                                st.metric("🗑️ Deleted", decision_counts.get('delete', 0))
                            with col3:
                                st.metric("👥 Keep Separate", decision_counts.get('keep_separate', 0))
                            with col4:
                                st.metric("🎯 Choose One", decision_counts.get('choose_one', 0))

                            # Show additional metrics in a second row if there are more decision types
                            other_decisions = {
                                k: v
                                for k, v in decision_counts.items()
                                if k not in ['keep', 'delete', 'keep_separate', 'choose_one']
                            }
                            if other_decisions:
                                st.markdown("**Other Decisions:**")
                                cols = st.columns(len(other_decisions))
                                for i, (decision, count) in enumerate(other_decisions.items()):
                                    with cols[i]:
                                        st.metric(decision.replace('_', ' ').title(), count)

                        if st.button("🔄 Start New Review"):
                            del st.session_state[queue_key]
                            del st.session_state[index_key]
                            del st.session_state[decisions_key]
                            st.rerun()

                elif review_mode.startswith("🎯 Focus Mode"):
                    st.success("🎉 No problematic credits found! All credits appear to be correctly processed.")
                    conn = sqlite3.connect(config.DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute(
                        f"SELECT COUNT(*) FROM {config.DB_TABLE_CREDITS} WHERE episode_id = ?", (selected_episode,)
                    )
                    total_credits = cursor.fetchone()[0]
                    conn.close()
                    st.info(f"📊 Total credits in database: {total_credits}")

                else:
                    st.subheader(f"Overview: {selected_episode}")
                    conn = sqlite3.connect(config.DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute(
                        f"""
                        SELECT id, role_group_normalized, name, role_detail,
                            source_frame, original_frame_number, scene_position
                        FROM {config.DB_TABLE_CREDITS}
                        WHERE episode_id = ?
                        ORDER BY role_group_normalized, name
                    """,
                        (selected_episode,),
                    )
                    credits_data = cursor.fetchall()
                    conn.close()

                    if not credits_data:
                        st.warning(f"No credits found for episode {selected_episode}")
                    else:
                        import pandas as pd

                        processed_data = []
                        for row in credits_data:
                            credit_id, role_group, name, role_detail, source_frame, frame_numbers, scene_pos = row
                            processed_data.append(
                                [
                                    credit_id,
                                    role_group,
                                    name,
                                    role_detail,
                                    source_frame or "N/A",
                                    frame_numbers or "N/A",
                                    scene_pos,
                                ]
                            )

                        df = pd.DataFrame(
                            processed_data,
                            columns=[
                                'ID',
                                'Role Group',
                                'Name',
                                'Role Detail',
                                'Source Frames',
                                'Frame Numbers',
                                'Scene Position',
                            ],
                        )

                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Total Credits", len(df))
                        with col2:
                            st.metric("Unique Role Groups", df['Role Group'].nunique())
                        with col3:
                            st.metric("Unique Names", df['Name'].nunique())
                        with col4:
                            unknown_count = len(df[df['Role Group'].str.contains('unknown', case=False, na=False)])
                            st.metric("Unknown Roles", unknown_count, delta=None if unknown_count == 0 else "⚠️")
                        with col5:
                            if st.button("📥 Export CSV"):
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    label="💾 Download CSV",
                                    data=csv,
                                    file_name=f"{selected_episode}_credits.csv",
                                    mime="text/csv",
                                )

                        problematic_credits = utils.identify_problematic_credits(selected_episode)
                        problematic_ids = {c['id'] for c in problematic_credits}
                        problematic_count = len(problematic_ids)
                        if problematic_count > 0:
                            st.warning(f"⚠️ {problematic_count} credits need attention")

                        search_filter = st.text_input(
                            "🔍 Search credits:", placeholder="Enter name or role to filter..."
                        )

                        def highlight_problematic(row):
                            return (
                                ['background-color: #b59736'] * len(row)
                                if row['ID'] in problematic_ids
                                else [''] * len(row)
                            )

                        if search_filter:
                            filtered_df = df[
                                df['Name'].str.contains(search_filter, case=False, na=False)
                                | df['Role Group'].str.contains(search_filter, case=False, na=False)
                                | df['Role Detail'].str.contains(search_filter, case=False, na=False)
                            ]
                            st.write(f"Showing {len(filtered_df)} of {len(df)} credits matching '{search_filter}'")
                            styled = filtered_df.style.apply(highlight_problematic, axis=1).hide(
                                subset=['ID'], axis='columns'
                            )
                            st.dataframe(styled, use_container_width=True, hide_index=True)
                        else:
                            styled = df.style.apply(highlight_problematic, axis=1).hide(subset=['ID'], axis='columns')
                            st.dataframe(styled, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error loading credits data: {e}")
        logging.error(f"Error in credit review interface: {e}", exc_info=True)

elif st.session_state.current_tab == 2:
    st.header("Pipeline Logs")
    st.text_area(
        "Live Logs:", value=st.session_state.get('log_content', ''), height=600, key="log_display_text_area_v3"
    )