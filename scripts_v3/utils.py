import json
import json as _json
import logging  # Ensure logging is imported
import re
import sqlite3
import sys  # for console logging handler
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

import cv2
import imagehash
import numpy as np
import streamlit as st
from PIL import Image
from thefuzz import fuzz

from . import config, constants

# Import IMDB validation for name checking
try:
    from scripts_v3.imdb_name_validation import IMDBNameValidator
    IMDB_VALIDATION_AVAILABLE = True
    # Create a module-level validator instance for efficiency
    _imdb_validator = None
    
    def get_imdb_validator():
        """Get or create IMDB validator instance"""
        global _imdb_validator
        logging.debug(f"[IMDB Validator] get_imdb_validator called, current instance: {_imdb_validator}")
        if _imdb_validator is None:
            logging.info(f"[IMDB Validator] Creating new IMDBNameValidator instance")
            _imdb_validator = IMDBNameValidator()
            logging.info(f"[IMDB Validator] Created new instance: {_imdb_validator}")
        else:
            logging.debug(f"[IMDB Validator] Returning existing instance")
        return _imdb_validator
        
except ImportError:
    IMDB_VALIDATION_AVAILABLE = False
    logging.warning("IMDB name validation not available")
    
    def get_imdb_validator():
        return None


# La vostra classe personalizzata per il formato JSON
class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        # Includere campi extra se presenti
        if hasattr(record, 'episode_id'):
            log_record['episode_id'] = record.episode_id
        return _json.dumps(log_record)


# Il vostro handler custom per Streamlit
class StreamlitLogHandler(logging.Handler):
    """Custom logging handler che appende i log alla session state di Streamlit."""

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            # Appende solo se st.session_state.log_content è inizializzata
            if hasattr(st, 'session_state') and hasattr(st.session_state, 'log_content'):
                st.session_state.log_content += msg + "\n"
        except Exception:
            # Se non siamo nel contesto Streamlit, ignoriamo silenziosamente
            pass


def setup_logging() -> None:
    """
    Configura il sistema di logging in tre fasi:
      1. Rimuove/disabilita ogni handler preesistente di Streamlit (logger figli).
      2. Svuota gli handler del logger radice e imposta il livello DEBUG.
      3. Aggiunge tre handler distinti al root:
         • console_handler: StreamHandler su stdout (formato semplice)
         • streamlit_handler: il vostro StreamlitLogHandler (formato semplice)
         • file_handler: RotatingFileHandler con JSONFormatter
    """

    # ───────────────────────────────────────────────────────────
    # 1. Disabilitare ogni logger che inizia con "streamlit"
    # ───────────────────────────────────────────────────────────
    for logger_name in list(logging.root.manager.loggerDict.keys()):
        if logger_name.startswith("streamlit"):
            st_logger = logging.getLogger(logger_name)
            st_logger.handlers.clear()
            st_logger.propagate = False

    # ───────────────────────────────────────────────────────────
    # 2. Cancellare eventuali handler sul logger radice e impostare livello DEBUG
    # ───────────────────────────────────────────────────────────
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG)

    # ───────────────────────────────────────────────────────────
    # 3. Creare i formatter
    # ───────────────────────────────────────────────────────────
    simple_format_string = "%(levelname)s - %(message)s"
    simple_log_formatter = logging.Formatter(simple_format_string)

    # Formatter JSON per il RotatingFileHandler
    json_formatter = JSONFormatter(datefmt='%Y-%m-%dT%H:%M:%S')

    # ───────────────────────────────────────────────────────────
    # 4. Configurare il Console Handler (StreamHandler su stdout)
    # ───────────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(simple_log_formatter)
    root_logger.addHandler(console_handler)

    # ───────────────────────────────────────────────────────────
    # 5. Configurare il Custom Streamlit Handler
    # ───────────────────────────────────────────────────────────
    streamlit_handler = StreamlitLogHandler()
    streamlit_handler.setLevel(logging.DEBUG)
    streamlit_handler.setFormatter(simple_log_formatter)
    root_logger.addHandler(streamlit_handler)

    # ───────────────────────────────────────────────────────────
    # 6. Configurare il Rotating File Handler con JSONFormatter
    # ───────────────────────────────────────────────────────────
    try:
        log_file_path = config.LOG_FILE_PATH
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            filename=log_file_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"  # 5 MB
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(json_formatter)
        root_logger.addHandler(file_handler)

        # Per confermare che il FileHandler è stato configurato correttamente,
        # possiamo emettere un LogRecord di prova direttamente sul nuovo handler:
        file_handler.handle(
            logging.LogRecord(
                name="setup",
                level=logging.DEBUG,
                pathname=__file__,
                lineno=0,
                msg=f"RotatingFileHandler configurato su {log_file_path} con JSONFormatter.",
                args=(),
                exc_info=None,
                func="setup_logging",
            )
        )
    except Exception as exc:
        # Se l’aggiunta del FileHandler fallisce, viene gestita l’eccezione
        root_logger.error(
            f"Impossibile aggiungere il RotatingFileHandler per {config.LOG_FILE_PATH}: {exc}"
        )  # ───────────────────────────────────────────────────────────
    # 7. Messaggio di conferma finale
    # ───────────────────────────────────────────────────────────
    root_logger.info("Setup del logging completato: " "Console e Streamlit (formato semplice), File rotante JSON.")


def calculate_dynamic_fuzzy_threshold(text_length: int) -> int:
    """
    Calculate a dynamic fuzzy text similarity threshold based on text length.

    For longer texts, we need a higher threshold because:
    1. Minor OCR differences have bigger impact on similarity scores
    2. Token-based comparison becomes less reliable with very long texts
    3. We want to be more lenient with long credit texts to avoid duplicates

    Args:
        text_length: Length of the text being compared

    Returns:
        Dynamic threshold value between FUZZY_THRESHOLD_BASE and FUZZY_THRESHOLD_MAX
    """
    base_threshold = getattr(config, 'FUZZY_THRESHOLD_BASE', 60)
    scale_start = getattr(config, 'FUZZY_THRESHOLD_SCALE_START', 200)
    scale_rate = getattr(config, 'FUZZY_THRESHOLD_SCALE_RATE', 0.01)
    max_threshold = getattr(config, 'FUZZY_THRESHOLD_MAX', 85)

    if text_length <= scale_start:
        return base_threshold

    # Calculate progressive increase for text longer than scale_start
    extra_length = text_length - scale_start
    threshold_increase = extra_length * scale_rate

    # Apply the increase but cap at max_threshold
    dynamic_threshold = min(base_threshold + threshold_increase, max_threshold)

    return int(dynamic_threshold)


# Maximum attempts for OCR retries
MAX_OCR_ATTEMPTS: int = 3
import sqlite3


# Named tuple for OCR results to replace ambiguous multiple-return values
class OCRResult(NamedTuple):
    """
    Represents the structured results of an OCR operation.
    """

    text: Optional[str]
    details: Optional[Any]
    bbox: Optional[Tuple[int, int, int, int]]
    error: Optional[str]


# Helper functions for image conversions
def bgr_to_rgb_pil(img_bgr: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR image to a PIL RGB Image."""
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def pil_to_bgr_np(img_pil: Image.Image) -> np.ndarray:
    """Convert a PIL Image to an OpenCV BGR numpy array."""
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def load_role_map(json_path: Path) -> Dict[str, str]:
    """
    Loads and parses the role mapping JSON file mapping variants to canonical roles.

    Returns:
        A dict mapping lowercase role variants to their canonical form.
    """
    try:
        text = json.loads(json_path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        logging.error(f"Role map file not found: {json_path}")
        raise
    except json.JSONDecodeError as jde:
        logging.error(f"Invalid JSON in role map {json_path}: {jde}")
        return {}
    # Build a mapping of all variants to their canonical form
    role_map: Dict[str, str] = {}
    for canonical, variants in text.items():
        role_map[canonical.lower()] = canonical
        for variant in variants:
            role_map[variant.lower()] = canonical
    return role_map


def clean_vlm_output(raw_text: str) -> str:
    """
    Cleans the raw VLM output by stripping code fences and extracting a JSON list.

    Args:
        raw_text: The raw string output from the VLM model.

    Returns:
        A JSON-formatted string representing a list of credit objects.
    """

    cleaned = re.sub(r"```json\s*(.*?)\s*```", r"\1", raw_text, flags=re.DOTALL)
    cleaned = cleaned.strip()

    if cleaned.startswith('[') and cleaned.endswith(']'):
        return cleaned
    elif cleaned.startswith('{') and cleaned.endswith('}'):
        logging.warning(f"VLM returned a JSON object, expected list. Wrapping in list. Raw: {raw_text[:200]}...")
        return f"[{cleaned}]"
    else:
        logging.warning(f"VLM output doesn't look like JSON list: '{cleaned[:200]}...'")
        match = re.search(r"(\[.*\])", cleaned, re.DOTALL)
        if match:
            extracted_list = match.group(1).strip()
            logging.warning(f"Extracted potential JSON list using regex: '{extracted_list[:200]}...'")
            return extracted_list
        else:
            logging.error(f"Could not extract valid JSON list from VLM output: '{cleaned[:200]}...'")
            return "[]"


def parse_vlm_json(json_string: str, source_identifier: str, name_key: str = "name") -> List[Dict[str, Any]]:
    """
    Parses a JSON string from VLM and normalizes it into a list of credit dicts.

    Args:
        json_string: The JSON text returned by the VLM.
        source_identifier: Identifier for logging context.
        name_key: Key to use for the credit name field.

    Returns:
        A list of dictionaries, each containing at least 'name', 'role_group', and 'role_detail'.
    """
    parsed_list = []
    if not json_string or not json_string.strip():
        logging.warning(f"Empty JSON string received for {source_identifier}.")
        return []

    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as jde:
        logging.error(f"JSON decode error for {source_identifier}: {jde} - Input truncated: {json_string[:200]}...")
        return []
    # Ensure data is a list
    if isinstance(data, dict):
        logging.warning(f"VLM JSON for {source_identifier} is a dict; wrapping into list.")
        data = [data]
    elif not isinstance(data, list):
        logging.error(f"Unexpected VLM JSON type for {source_identifier}: {type(data)}; expected list or dict.")
        return []
    # Validate each entry
    for item in data:
        if not isinstance(item, dict):
            logging.warning(f"Skipping non-dict item in VLM JSON for {source_identifier}: {item}")
            continue
        if name_key not in item or not item[name_key]:
            logging.warning(f"Skipping item missing '{name_key}' in VLM JSON for {source_identifier}: {item}")
            continue
        item.setdefault('role_group', None)
        item.setdefault('role_detail', None)
        parsed_list.append(item)
    return parsed_list

    return parsed_list


def is_fade_frame(frame):
    """
    Determines whether a video frame is a fade (very dark or low contrast).

    Args:
        frame: A NumPy array representing the video frame.

    Returns:
        True if the frame is classified as a fade; False otherwise.
    """
    if frame is None or frame.size == 0:
        logging.warning("is_fade_frame received invalid frame (None or zero size).")
        return False
    try:
        processed_frame = frame.copy()

        if processed_frame.dtype != np.uint8:
            if (
                np.issubdtype(processed_frame.dtype, np.floating)
                and processed_frame.max() <= 1.0
                and processed_frame.min() >= 0.0
            ):
                processed_frame = (processed_frame * 255).astype(np.uint8)
            else:
                processed_frame = processed_frame.astype(np.uint8)

        if len(processed_frame.shape) == 2:
            gray = processed_frame

        elif len(processed_frame.shape) == 3 and processed_frame.shape[2] == 3:
            gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        elif len(processed_frame.shape) == 3 and processed_frame.shape[2] == 4:
            bgr_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        else:
            logging.error(
                f"is_fade_frame received frame with unsupported shape: {processed_frame.shape}. Cannot process for fade detection."
            )
            return False

        mean_brightness, std_dev_contrast = cv2.meanStdDev(gray)
        mean_brightness = mean_brightness[0][0]
        std_dev_contrast = std_dev_contrast[0][0]

        is_low_brightness = mean_brightness < constants.FADE_FRAME_THRESHOLD
        is_high_brightness = mean_brightness > (255 - constants.FADE_FRAME_THRESHOLD)
        is_low_contrast = std_dev_contrast < constants.FADE_FRAME_CONTRAST_THRESHOLD

        if (is_low_brightness or is_high_brightness) and is_low_contrast:
            return True
    except cv2.error as e:
        logging.warning(f"OpenCV error in is_fade_frame: {e}")
    except Exception as e:
        logging.error(f"Unexpected error in is_fade_frame: {e}", exc_info=True)
    return False


def calculate_vertical_flow(prev_gray, current_gray):
    """Calculates the median vertical optical flow between two grayscale frames."""
    if prev_gray is None or current_gray is None:
        return 0.0
    if prev_gray.shape != current_gray.shape:
        logging.warning(f"Shape mismatch in flow: {prev_gray.shape} vs {current_gray.shape}. Resizing current.")
        try:
            current_gray = cv2.resize(current_gray, (prev_gray.shape[1], prev_gray.shape[0]))
        except cv2.error as resize_e:
            logging.error(f"Resize failed: {resize_e}")
            return 0.0
    try:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, **constants.OPTICAL_FLOW_PARAMS)
        if flow is None:
            return 0.0
        v_flow = flow[:, :, 1]
        significant_flow = v_flow[np.abs(v_flow) > 0.1]
        if significant_flow.size == 0:
            return 0.0
        return np.median(significant_flow)
    except cv2.error as e:
        logging.error(f"OpenCV flow error: {e}")
        return 0.0
    except Exception as e:
        logging.error(f"Unexpected flow error: {e}")
        return 0.0


def calculate_hash_difference(hash1: Optional[imagehash.ImageHash], hash2: Optional[imagehash.ImageHash]) -> int:
    """Calculates the Hamming difference between two image hashes."""
    if hash1 is None or hash2 is None:
        if hash1 is None and hash2 is None:
            return 0
        return constants.HASH_SIZE * constants.HASH_SIZE
    return hash1 - hash2


def get_paddleocr_reader(lang: str = 'it'):
    """Initializes and returns a PaddleOCR reader instance for the specified language."""
    # Lazy import to avoid DLL conflicts when OCR engine is not selected
    from paddleocr import PaddleOCR

    actual_lang_code = config.PADDLEOCR_LANG_MAP.get(lang, lang)

    logging.info(f"Attempting PaddleOCR init for lang='{actual_lang_code}' (PP-OCRv5)")
    ocr_reader = PaddleOCR(
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        lang=actual_lang_code,
    )
    logging.info(f"PaddleOCR reader initialized for '{actual_lang_code}'")
    return ocr_reader


def paddleocr_predict_with_retry(ocr_reader, img, max_retries=2):
    """
    Run PaddleOCR prediction with automatic retry on model caching issues.
    
    Args:
        ocr_reader: The PaddleOCR reader instance
        img: Image array to process
        max_retries: Maximum number of retries (default: 2)
        
    Returns:
        tuple: (success, results_or_error)
    """
    for attempt in range(max_retries + 1):
        try:
            results = ocr_reader.predict(img)
            return True, results
        except RuntimeError as e:
            if "Unknown exception" in str(e) and attempt < max_retries:
                logging.warning(f"PaddleOCR model caching issue detected (attempt {attempt + 1}), will retry...")
                # Force garbage collection to help clear any cached states
                import gc
                gc.collect()
                continue
            else:
                return False, f"PaddleOCR RuntimeError: {e}"
        except Exception as e:
            return False, f"PaddleOCR Exception: {e}"
    
    return False, "PaddleOCR failed after all retries"


def load_user_stopwords() -> list[str]:
    """Loads user-defined stopwords from the path specified in config.
    If the file doesn't exist, it creates it with default values.
    """
    stopwords_path = config.OCR_USER_STOPWORDS_PATH
    if stopwords_path.exists():
        try:
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                stopwords = [line.strip() for line in f if line.strip()]
                logging.info(f"Stopwords loaded: {str(stopwords)}")
            if not stopwords:  # File exists but is empty or only whitespace
                logging.info(f"User stopwords file {stopwords_path} is empty. Using default stopwords.")
                # Optionally, rewrite with defaults if empty
                # save_user_stopwords(config.DEFAULT_OCR_USER_STOPWORDS)
                # return config.DEFAULT_OCR_USER_STOPWORDS
                return config.DEFAULT_OCR_USER_STOPWORDS[:]  # Return a copy
            logging.info(f"Loaded {len(stopwords)} user stopwords from {stopwords_path}.")
            return stopwords
        except Exception as e:
            logging.error(f"Error loading user stopwords from {stopwords_path}: {e}. Using default stopwords.")
            return config.DEFAULT_OCR_USER_STOPWORDS[:]  # Return a copy
    else:
        logging.info(f"User stopwords file {stopwords_path} not found. Creating with default stopwords.")
        try:
            # Create the directory if it doesn't exist (though PROJECT_ROOT should exist)
            stopwords_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stopwords_path, 'w', encoding='utf-8') as f:
                for word in config.DEFAULT_OCR_USER_STOPWORDS:
                    f.write(f"{word}\n")
            logging.info(f"Created {stopwords_path} with {len(config.DEFAULT_OCR_USER_STOPWORDS)} default stopwords.")
            return config.DEFAULT_OCR_USER_STOPWORDS[:]  # Return a copy
        except Exception as e:
            logging.error(f"Error creating user stopwords file {stopwords_path}: {e}. Using default stopwords.")
            return config.DEFAULT_OCR_USER_STOPWORDS[:]  # Return a copy


def save_user_stopwords(stopwords: list[str]) -> None:
    """Saves the list of stopwords to the path specified in config."""
    stopwords_path = config.OCR_USER_STOPWORDS_PATH
    try:
        # Create the directory if it doesn't exist
        stopwords_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stopwords_path, 'w', encoding='utf-8') as f:
            for word in stopwords:
                f.write(f"{word}\n")
        logging.info(f"Saved {len(stopwords)} user stopwords to {stopwords_path}.")
    except Exception as e:
        logging.error(f"Error saving user stopwords to {stopwords_path}: {e}")


def init_db() -> None:
    """Initializes the SQLite database and creates tables if they don't exist."""
    try:
        config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            f"""
        CREATE TABLE IF NOT EXISTS {config.DB_TABLE_EPISODES} (
            episode_id TEXT PRIMARY KEY,
            series_title TEXT,
            season_number INTEGER,
            episode_number INTEGER,
            video_filename TEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        )
        logging.info(f"Table '{config.DB_TABLE_EPISODES}' checked/created successfully.")

        cursor.execute(
            f"""
        CREATE TABLE IF NOT EXISTS {config.DB_TABLE_CREDITS} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id TEXT NOT NULL,
            source_frame TEXT NOT NULL,
            role_group TEXT,            name TEXT,
            role_detail TEXT,
            role_group_normalized TEXT,
            source_image_index INTEGER, -- Kept for historical reasons, but might be original_frame_number now
            scene_position TEXT,        -- Added for deduplication preference
            original_frame_number TEXT, -- Added to store original frame numbers as text            reviewed_status TEXT DEFAULT 'pending', -- Track review status: 'pending' or 'kept'
            reviewed_at TIMESTAMP,      -- When the credit was reviewed
            is_person BOOLEAN,          -- Whether the name refers to a person (true) or company (false)
            normalized_name TEXT,       -- Normalized name for IMDB validation
            is_present_in_imdb BOOLEAN, -- Whether the name was found in IMDB database
            FOREIGN KEY (episode_id) REFERENCES {config.DB_TABLE_EPISODES} (episode_id)
        )
        """        )
          # Add the new columns to existing tables if they don't exist
        try:
            cursor.execute(f"ALTER TABLE {config.DB_TABLE_CREDITS} ADD COLUMN reviewed_status TEXT DEFAULT 'pending'")
            logging.info("Added 'reviewed_status' column to credits table.")
        except sqlite3.OperationalError:
            # Column already exists
            pass

        try:
            cursor.execute(f"ALTER TABLE {config.DB_TABLE_CREDITS} ADD COLUMN reviewed_at TIMESTAMP")
            logging.info("Added 'reviewed_at' column to credits table.")
        except sqlite3.OperationalError:
            # Column already exists
            pass

        try:
            cursor.execute(f"ALTER TABLE {config.DB_TABLE_CREDITS} ADD COLUMN is_person BOOLEAN")
            logging.info("Added 'is_person' column to credits table.")
        except sqlite3.OperationalError:
            # Column already exists
            pass

        try:
            cursor.execute(f"ALTER TABLE {config.DB_TABLE_CREDITS} ADD COLUMN normalized_name TEXT")
            logging.info("Added 'normalized_name' column to credits table.")
        except sqlite3.OperationalError:
            # Column already exists
            pass

        try:
            cursor.execute(f"ALTER TABLE {config.DB_TABLE_CREDITS} ADD COLUMN is_present_in_imdb BOOLEAN")
            logging.info("Added 'is_present_in_imdb' column to credits table.")
        except sqlite3.OperationalError:
            # Column already exists
            pass

        cursor.execute(
            f"""
        CREATE INDEX IF NOT EXISTS idx_credits_episode_id ON {config.DB_TABLE_CREDITS} (episode_id);
        """
        )
        logging.info(f"Table '{config.DB_TABLE_CREDITS}' checked/created successfully.")

        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        logging.error(f"Database error during initialization: {e}", exc_info=True)
        st.error(f"Database initialization failed: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during DB initialization: {e}", exc_info=True)
        st.error(f"An unexpected error occurred during database initialization: {e}")


def deduplicate_credits(credits: list[dict]) -> list[dict]:
    """
    Deduplicate credits by (role_group_normalized or role_group, name).
    Merge source_frame and original_frame_number into lists.
    Prioritize entries from "second_half" if conflicts arise for frame/number/scene_pos.
    Prefer longer role_detail string.
    If the same name appears with different role groups, add a revision flag.
    """
    dedup_credits_map = {}
    name_to_role_groups = {}

    def _make_list(value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    for credit in credits:
        role_group = credit.get("role_group_normalized") or credit.get("role_group")
        name = (credit.get("name") or "").strip()
        key = (role_group, name)

        if name and role_group:
            name_to_role_groups.setdefault(name, set()).add(role_group)

        current_source_frames = _make_list(credit.get("source_frame"))
        current_frame_numbers = _make_list(credit.get("original_frame_number"))
        current_scene_pos = credit.get("scene_position", "unknown")

        current_rd = credit.get("role_detail") or ""

        if key not in dedup_credits_map:
            new_entry = credit.copy()
            new_entry["source_frame"] = current_source_frames
            new_entry["original_frame_number"] = current_frame_numbers
            new_entry["scene_position"] = current_scene_pos
            new_entry["role_detail"] = current_rd
            new_entry.pop("source_image_batch_index", None)
            new_entry.pop("source_image_index_issue", None)
            dedup_credits_map[key] = new_entry
        else:
            existing_entry = dedup_credits_map[key]
            existing_scene_pos = existing_entry.get("scene_position", "unknown")

            existing_rd = existing_entry.get("role_detail") or ""

            is_current_preferred_scene = current_scene_pos == "second_half" and existing_scene_pos != "second_half"
            is_existing_preferred_scene = existing_scene_pos == "second_half" and current_scene_pos != "second_half"

            final_role_detail = existing_rd
            if len(current_rd) > len(existing_rd):
                final_role_detail = current_rd

            if is_current_preferred_scene:
                preferred_entry_base = credit.copy()
                preferred_entry_base["source_frame"] = current_source_frames
                preferred_entry_base["original_frame_number"] = current_frame_numbers
                preferred_entry_base["scene_position"] = current_scene_pos
                preferred_entry_base["role_detail"] = final_role_detail

                for f in _make_list(existing_entry.get("source_frame")):
                    if f not in preferred_entry_base["source_frame"]:
                        preferred_entry_base["source_frame"].append(f)
                for n in _make_list(existing_entry.get("original_frame_number")):
                    if n not in preferred_entry_base["original_frame_number"]:
                        preferred_entry_base["original_frame_number"].append(n)

                dedup_credits_map[key] = preferred_entry_base

            elif is_existing_preferred_scene:
                existing_entry["role_detail"] = final_role_detail

                for f in current_source_frames:
                    if f not in existing_entry["source_frame"]:
                        existing_entry["source_frame"].append(f)
                for n in current_frame_numbers:
                    if n not in existing_entry["original_frame_number"]:
                        existing_entry["original_frame_number"].append(n)

            else:
                existing_entry["role_detail"] = final_role_detail

                for f in current_source_frames:
                    if f not in existing_entry["source_frame"]:
                        existing_entry["source_frame"].append(f)
                for n in current_frame_numbers:
                    if n not in existing_entry["original_frame_number"]:
                        existing_entry["original_frame_number"].append(n)

                if existing_scene_pos == "unknown" and current_scene_pos != "unknown":
                    existing_entry["scene_position"] = current_scene_pos

    final_credits_list = list(dedup_credits_map.values())

    for credit_entry in final_credits_list:
        name = (credit_entry.get("name") or "").strip()
        if name and len(name_to_role_groups.get(name, set())) > 1:
            credit_entry["Need revisioning for deduplication"] = True

        credit_entry.pop("source_image_batch_index", None)
        credit_entry.pop("source_image_index_issue", None)

    return final_credits_list


def save_credits(episode_id: str, credits_data: list[dict]) -> None:
    """Saves the list of credit dictionaries to the database for a specific episode.
    Deletes existing credits for the episode before inserting new ones."""
    conn = None
    try:
        credits_data = deduplicate_credits(credits_data)
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()

        cursor.execute(f"DELETE FROM {config.DB_TABLE_CREDITS} WHERE episode_id = ?", (episode_id,))
        logging.info(f"[{episode_id}] Deleted existing credits from DB.")

        insert_data = []
        for credit in credits_data:
            source_frame = credit.get('source_frame')
            if isinstance(source_frame, list):
                source_frame_db = ",".join(source_frame)
            else:
                source_frame_db = source_frame
            original_frame_number = credit.get('original_frame_number')
            if isinstance(original_frame_number, list):
                original_frame_number_db = ",".join(str(x) for x in original_frame_number if x is not None)
            else:
                original_frame_number_db = str(original_frame_number) if original_frame_number is not None else None
            
            scene_pos = credit.get('scene_position', None)
              # Handle is_person field - infer from role group if not provided
            is_person = credit.get('is_person')
            if is_person is None:
                # Fallback: detect based on role group or name patterns
                if is_company_role_group(credit.get('role_group')):
                    is_person = False
                elif detect_company_name_patterns(credit.get('name', '')):
                    is_person = False
                else:
                    is_person = True  # Default to person

            # Calculate normalized name for IMDB validation and operations
            name = credit.get('name', '')
            normalized_name = normalize_name(name) if name else None

            insert_data.append(
                (
                    episode_id,
                    source_frame_db,
                    credit.get('role_group'),
                    name,
                    credit.get('role_detail'),                    credit.get('role_group_normalized'),
                    original_frame_number_db,
                    scene_pos,
                    is_person,
                    normalized_name,
                )
            )

        cursor.executemany(
            f"""
        INSERT INTO {config.DB_TABLE_CREDITS}
        (episode_id, source_frame, role_group, name, role_detail, role_group_normalized, original_frame_number, scene_position, is_person, normalized_name)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            insert_data,
        )

        conn.commit()
        logging.info(f"[{episode_id}] Successfully saved {len(insert_data)} credits to DB.")
        
        # Invalidate cache after successful save
        invalidate_credits_cache(episode_id)
        
        return True, f"Saved {len(insert_data)} credits."

    except sqlite3.Error as e:
        logging.error(f"[{episode_id}] Database error saving credits: {e}", exc_info=True)
        if conn:
            conn.rollback()
        return False, f"Database error: {e}"
    except Exception as e:
        logging.error(f"[{episode_id}] Unexpected error saving credits: {e}", exc_info=True)
        if conn:
            conn.rollback()
        return False, f"Unexpected error: {e}"
    finally:
        if conn:
            conn.close()


def load_vlm_results_from_jsonl(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Loads VLM results from a JSON file (expected to be a list of dicts)."""
    results = []
    if not jsonl_path.is_file():
        logging.warning(f"VLM results file not found: {jsonl_path}")
        return results
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                logging.warning(f"VLM results file is empty: {jsonl_path}")
                return []
            data = json.loads(content)
            if isinstance(data, list):
                results = data
            else:
                logging.warning(
                    f"VLM results file {jsonl_path} does not contain a JSON list. Content type: {type(data)}"
                )

                if isinstance(data, dict):
                    results = [data]
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {jsonl_path}: {e}")
    except Exception as e:
        logging.error(f"Error reading VLM results from {jsonl_path}: {e}")
    return results


def load_processed_frames(jsonl_path: Path, episode_id: str) -> Set[str]:
    """Loads the set of already processed frame filenames from a JSONL file."""
    processed_set = set()
    if not jsonl_path.is_file():
        logging.info(f"[{episode_id}] Processed frames file not found ({jsonl_path.name}), starting fresh.")
        return processed_set

    logging.info(f"[{episode_id}] Loading previously processed frames from {jsonl_path.name}...")
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                try:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)

                    if isinstance(data, dict) and 'source_frame' in data:
                        processed_set.add(data['source_frame'])
                except json.JSONDecodeError as json_err:
                    logging.warning(
                        f"[{episode_id}] Skipping invalid JSON line in {jsonl_path.name}: {json_err} - Line: '{line[:100]}...'"
                    )
                except Exception as line_err:
                    logging.warning(
                        f"[{episode_id}] Error processing line in {jsonl_path.name}: {line_err} - Line: '{line[:100]}...'"
                    )
    except Exception as load_err:
        logging.error(
            f"[{episode_id}] Failed to load or read processed frames file {jsonl_path.name}: {load_err}", exc_info=True
        )

    logging.info(f"[{episode_id}] Loaded {len(processed_set)} processed frame filenames.")
    return processed_set


def normalize_text_for_comparison(text: str, user_stopwords: list[str]) -> str:
    """
    Normalizes OCR text for comparison by:
    1. Converting to lowercase.
    2. Removing punctuation and special characters (keeps alphanumeric and spaces).
    3. Removing user-defined stopwords.
    4. Removing ALL whitespace to create a single string for comparison.
    """
    if not text:
        return ""  # Convert to lowercase
    text = text.lower()

    # Remove punctuation and special characters - keep letters, numbers, and spaces
    # This will remove hyphens, apostrophes etc.
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # Split into words, remove stopwords, then rejoin
    # This also handles removal of extra whitespace effectively
    words = text.split()
    if user_stopwords:
        # Ensure stopwords are lowercase for comparison
        lower_stopwords = [stopword.lower() for stopword in user_stopwords]
        filtered_words = []
        for word in words:
            # Remove stopwords as substrings from each word
            cleaned_word = word
            for stopword in lower_stopwords:
                cleaned_word = cleaned_word.replace(stopword, '')
            # Only keep the word if there's something left after removing stopwords
            if cleaned_word.strip():
                filtered_words.append(cleaned_word)
        words = filtered_words

    # Join words WITHOUT spaces to create a single string for comparison
    return " ".join(words)


def init_global_text_hash_state(episode_id: str):
    """
    Initializes or resets the global (session-level) state for the last saved
    OCR text and frame hash for a specific episode.
    """
    session_key_text = f"global_last_ocr_text_{episode_id}"
    session_key_hash = f"global_last_frame_hash_{episode_id}"
    session_key_bbox = f"global_last_ocr_bbox_{episode_id}"

    if session_key_text in st.session_state:
        del st.session_state[session_key_text]
    if session_key_hash in st.session_state:
        del st.session_state[session_key_hash]
    if session_key_bbox in st.session_state:
        del st.session_state[session_key_bbox]

    st.session_state[session_key_text] = None
    st.session_state[session_key_hash] = None
    st.session_state[session_key_bbox] = None
    logging.debug(f"[{episode_id}] Initialized global text/hash/bbox tracking state in session_state.")


def correct_bbox_for_rotation(bbox, angle, image_width, image_height):
    """
    Correct bounding box coordinates for document orientation rotation.

    Args:
        bbox: Bounding box in format [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] or [x1,y1,x2,y2]
              Can be a numpy array or regular Python list
        angle: Rotation angle in degrees (0, 90, 180, 270)
        image_width: Original image width
        image_height: Original image height

    Returns:
        Corrected bounding box in same format as input
    """
    if angle == 0:
        return bbox

    # Handle numpy arrays by converting to list
    if hasattr(bbox, 'tolist'):
        bbox = bbox.tolist()

    # Check if bbox is empty or None after conversion
    if not bbox or (hasattr(bbox, '__len__') and len(bbox) == 0):
        return bbox

    # Convert to polygon format if needed
    if isinstance(bbox, list) and len(bbox) == 4 and not isinstance(bbox[0], (list, tuple)):
        # Convert [x1,y1,x2,y2] to [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        x1, y1, x2, y2 = bbox
        points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    else:
        points = bbox

    corrected_points = []

    for point in points:
        x, y = point[0], point[1]

        if angle == 90:
            # 90° clockwise: (x,y) -> (y, width-x)
            new_x = y
            new_y = image_width - x
        elif angle == 180:
            # 180°: (x,y) -> (width-x, height-y)
            new_x = image_width - x
            new_y = image_height - y
        elif angle == 270:
            # 270° clockwise: (x,y) -> (height-y, x)
            new_x = image_height - y
            new_y = x
        else:
            new_x, new_y = x, y

        corrected_points.append([new_x, new_y])

    # Return in same format as input
    if isinstance(bbox, list) and len(bbox) == 4 and not isinstance(bbox[0], (list, tuple)):
        # Convert back to [x1,y1,x2,y2] format
        xs = [p[0] for p in corrected_points]
        ys = [p[1] for p in corrected_points]
        return [min(xs), min(ys), max(xs), max(ys)]
    else:
        return corrected_points


def run_ocr(
    img_np: np.ndarray,
    ocr_reader: Any,
    ocr_engine_type: str,
    image_context_identifier: Optional[str] = None,
    apply_clahe: bool = True,
    try_both_clahe_and_original: bool = True,
    debug_image_name_prefix: Optional[str] = None,
) -> Tuple[
    Optional[str],  # OCR text
    Any,  # OCR details structure
    Optional[Tuple[int, int, int, int]],  # Bounding box
    Optional[str],  # Error message
]:
    """
    Runs OCR on the given image, trying different rotations and optionally CLAHE.
    Returns the best text found, its details, bounding box, and any error.

    Args:
        img_np: Image as a NumPy array.
        ocr_reader: The initialized OCR reader instance.
        ocr_engine_type: String identifier for the OCR engine (e.g., "paddleocr", "easyocr").
        image_context_identifier: Optional string to prepend to log messages for better context.
        apply_clahe: Whether to apply CLAHE by default.
        try_both_clahe_and_original: If True, will try with and without CLAHE if the first attempt fails or yields no text.
        debug_image_name_prefix: Optional prefix for saving debug images.
    """
    if img_np is None or img_np.size == 0:
        return None, None, None, "Input image is empty"

    best_text: str | None = None
    best_details: Any | None = None
    best_bbox: Tuple[int, int, int, int] | None = None
    best_score = -1
    final_error_message = None

    # Prepare processing configs (Original and optional CLAHE)
    processing_configs = [{"apply_clahe": False, "label": "Original"}]
    if try_both_clahe_and_original:
        processing_configs.append({"apply_clahe": True, "label": "CLAHE"})

    for proc in processing_configs:
        img_to_ocr = apply_clahe_filter(img_np) if proc["apply_clahe"] else img_np
        log_tag = f"{image_context_identifier} {proc['label']}" if image_context_identifier else proc['label']

        try:
            if ocr_engine_type == "paddleocr":
                logging.info(f"{log_tag} Starting PaddleOCR prediction...")
                success, result = paddleocr_predict_with_retry(ocr_reader, img_to_ocr)
                
                if not success:
                    final_error_message = f"PaddleOCR failed: {result}"
                    logging.error(f"{log_tag} {final_error_message}")
                    continue
                    
                raw_results = result
                logging.info(f"{log_tag} PaddleOCR returned {len(raw_results) if raw_results else 0} results")

                text_lines = []

                # Extract rotation angle from doc_preprocessor_res if available
                rotation_angle = 0
                image_height, image_width = img_to_ocr.shape[:2]
                logging.info(f"{log_tag} Processing PaddleOCR results, image size: {image_width}x{image_height}")

                if raw_results and len(raw_results) > 0:
                    logging.info(f"{log_tag} Processing {len(raw_results)} raw results...")
                    result = raw_results[0]
                    logging.info(f"{log_tag} First result type: {type(result)}")

                    if isinstance(result, dict):
                        # logging.info(f"{log_tag} Result is dict with keys: {list(result.keys())}")
                        # Check for document preprocessing rotation
                        doc_preprocess = result.get("doc_preprocessor_res", {})
                        if doc_preprocess and isinstance(doc_preprocess, dict):
                            rotation_angle = doc_preprocess.get("angle", 0)
                            logging.info(f"{log_tag} PaddleOCR detected rotation angle: {rotation_angle}°")

                        texts = result.get("rec_texts", [])
                        scores = result.get("rec_scores", [])
                        polys = result.get("rec_polys", [])
                        boxes = result.get("rec_boxes", [])

                        # logging.info(f"{log_tag} Extracted texts: {len(texts)}, scores: {len(scores)}, polys: {len(polys)}, boxes: {len(boxes)}")
                        # logging.debug(f"{log_tag} PaddleOCR extracted: {len(texts)} texts, {len(scores)} scores, {len(polys)} polys, {len(boxes)} boxes")

                        # Use polys if available, otherwise fall back to boxes
                        bbox_data = polys if polys else boxes
                        logging.info(
                            f"{log_tag} Using bbox data from: {'polys' if polys else 'boxes'}, length: {len(bbox_data)}"
                        )

                        # Process each text detection
                        logging.info(f"{log_tag} Starting to process {len(texts)} text detections...")
                        for i in range(len(texts)):
                            logging.debug(f"{log_tag} Processing text detection {i+1}/{len(texts)}...")
                            txt = texts[i] if i < len(texts) else ""
                            conf = scores[i] if i < len(scores) else 0.0

                            # logging.debug(f"{log_tag} Text {i+1}: '{txt}', confidence: {conf}")

                            if conf < constants.MIN_OCR_CONFIDENCE:
                                # logging.debug(f"{log_tag} Skipping text {i+1} due to low confidence: {conf} < {config.MIN_OCR_CONFIDENCE}")
                                bbox_ocr = None
                                continue
                            if i < len(bbox_data):
                                # logging.debug(f"{log_tag} Processing bbox {i+1}/{len(bbox_data)}...")
                                bbox_info = bbox_data[i]
                                try:
                                    # logging.debug(f"{log_tag} Original bbox_info type: {type(bbox_info)}, value: {bbox_info}")
                                    # Handle numpy arrays by converting to list
                                    if hasattr(bbox_info, 'tolist'):
                                        bbox_info = bbox_info.tolist()
                                        # logging.debug(f"{log_tag} Converted bbox_info to list: {bbox_info}")

                                    if isinstance(bbox_info, list):
                                        if len(bbox_info) >= 4:
                                            # Check if it's polygon format [[x,y], [x,y], ...] or box format [x1,y1,x2,y2]
                                            if isinstance(bbox_info[0], (list, tuple)):
                                                # Polygon format: [[x,y], [x,y], [x,y], [x,y]]
                                                bbox_ocr = bbox_info
                                                # logging.debug(f"{log_tag} Using polygon format bbox: {bbox_ocr}")
                                            elif len(bbox_info) == 4:
                                                # Box format: [x1, y1, x2, y2] -> convert to polygon
                                                x1, y1, x2, y2 = bbox_info
                                                bbox_ocr = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                                                # logging.debug(f"{log_tag} Converted box to polygon format: {bbox_ocr}")
                                            else:
                                                logging.warning(f"{log_tag} Unexpected bbox format: {bbox_info}")

                                            # Correct bbox for rotation if needed
                                            if rotation_angle != 0 and bbox_ocr:
                                                logging.debug(
                                                    f"{log_tag} Applying rotation correction for {rotation_angle}°..."
                                                )
                                                bbox_ocr = correct_bbox_for_rotation(
                                                    bbox_ocr, rotation_angle, image_width, image_height
                                                )
                                                logging.debug(
                                                    f"{log_tag} Corrected bbox for {rotation_angle}° rotation: {bbox_ocr}"
                                                )
                                        else:
                                            logging.warning(f"{log_tag} Unexpected bbox format: {bbox_info}")
                                except Exception as e:
                                    logging.warning(f"{log_tag} Error processing bbox {i}: {e}")
                                    bbox_ocr = None

                            text_lines.append((bbox_ocr, txt, float(conf)))
                            # logging.debug(f"{log_tag} Added text line {i+1}: '{txt}' with bbox: {bbox_ocr}")

                        # logging.info(f"{log_tag} Finished processing all text detections. Total processed: {len(text_lines)}")
                    else:
                        logging.warning(f"{log_tag} Unexpected PaddleOCR result format: {type(result)}")
                else:
                    logging.info(f"{log_tag} PaddleOCR returned no results")
            elif ocr_engine_type == "easyocr":
                text_lines: list = []  # maintain rotations for EasyOCR
                for angle in (0, 90, 180, 270):
                    img_rot = rotate_image(img_to_ocr, angle) if angle else img_to_ocr
                    raw = ocr_reader.readtext(img_rot)
                    logging.debug(f"{log_tag} EasyOCR raw results rot{angle}: {raw}")
                    for bbox, txt, conf in raw:
                        if conf >= constants.MIN_OCR_CONFIDENCE:
                            # bbox is list of points [[x,y],...]
                            formatted = [[int(pt[0]), int(pt[1])] for pt in bbox]
                            text_lines.append((formatted, txt, float(conf)))
            else:
                raise ValueError(f"Unsupported OCR engine: {ocr_engine_type}")

        except Exception as err:
            final_error_message = f"{ocr_engine_type} OCR Error: {err}"
            logging.error(f"{log_tag} {final_error_message}", exc_info=True)
            continue

        if not text_lines:
            final_error_message = f"No text found ({proc['label']})"
            logging.info(
                f"{image_context_identifier} OCR tried {proc['label']} processing but found no text (confidence threshold: {constants.MIN_OCR_CONFIDENCE})"
            )
            continue

        # Sort text lines by bbox position before combining
        sorted_text_lines = sort_text_lines_by_bbox(text_lines)

        # combine text_lines into one result
        combined = " ".join([ln[1] for ln in sorted_text_lines]).strip()
        # logging.info(f"{image_context_identifier} OCR {proc['label']} found {len(text_lines)} text lines, combined length: {len(combined)}")

        # Log each individual text line found (in sorted order)
        for i, (bbox, text, conf) in enumerate(sorted_text_lines):
            bbox_info = ""
            if bbox:
                try:
                    if isinstance(bbox, list) and len(bbox) >= 4:
                        if isinstance(bbox[0], (list, tuple)):
                            # Format: [[x1,y1], [x2,y2], ...]
                            ys = [pt[1] for pt in bbox]
                            xs = [pt[0] for pt in bbox]
                            center_y = sum(ys) / len(ys)
                            center_x = sum(xs) / len(xs)
                        else:
                            # Format: [x1, y1, x2, y2, ...]
                            ys = bbox[1::2]
                            xs = bbox[::2]
                            center_y = sum(ys) / len(ys)
                            center_x = sum(xs) / len(xs)
                        bbox_info = f" (center: {center_x:.0f},{center_y:.0f})"
                except Exception:
                    bbox_info = " (bbox parse error)"
            # logging.info(f"{image_context_identifier} OCR {proc['label']} line {i+1}: '{text}' (conf: {conf:.2f}){bbox_info}")

        if combined and len(combined) > best_score:
            xs = [pt[0] for ln in sorted_text_lines if ln[0] for pt in ln[0]]
            ys = [pt[1] for ln in sorted_text_lines if ln[0] for pt in ln[0]]
            best_text = combined
            best_details = sorted_text_lines  # Use sorted text lines
            best_bbox = (min(xs), min(ys), max(xs), max(ys)) if xs and ys else None
            best_score = len(combined)
            logging.info(
                f"{image_context_identifier} OCR {proc['label']} is new best result: '{combined}' (score: {best_score})"
            )
            logging.info(f"{image_context_identifier} OCR {proc['label']} FULL TEXT: '{combined}'")
            # break after first successful config
            break

    if best_text:
        logging.info(f"{image_context_identifier} OCR success: FULL TEXT: '{best_text}' bbox={best_bbox}")
        return best_text, best_details, best_bbox, None

    err = final_error_message or "OCR failed"
    logging.warning(f"{image_context_identifier} OCR final error: {err}")
    return None, None, None, err


def ocr_with_retry(
    img_np: np.ndarray,
    ocr_reader: Any,
    ocr_engine_type: str,
    image_context_identifier: Optional[str] = None,
    max_attempts: int = MAX_OCR_ATTEMPTS,
    retry_delay: float = 0.5,
) -> OCRResult:
    """
    Retry OCR up to max_attempts, returning first successful result or last error.
    """
    attempts = 0
    last_error = None
    while attempts < max_attempts:
        try:
            text, details, bbox, error = run_ocr(
                img_np, ocr_reader, ocr_engine_type, image_context_identifier=image_context_identifier
            )
            if error:
                last_error = error
            else:
                return OCRResult(text, details, bbox, None)
        except Exception as exc:
            last_error = str(exc)
            logging.error(f"[{image_context_identifier}] OCR exception on attempt {attempts+1}: {last_error}")
        attempts += 1
        if attempts < max_attempts:
            time.sleep(retry_delay)
    return OCRResult(None, None, None, last_error)


def apply_clahe_filter(
    img_np: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Applies CLAHE filter to a BGR image."""
    if img_np is None or img_np.size == 0:
        logging.warning("apply_clahe_filter: Input image is empty.")
        return img_np  # Or raise error

    if len(img_np.shape) == 3 and img_np.shape[2] == 3:  # Color image
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(img_lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l_channel)
        merged_channels = cv2.merge((cl, a_channel, b_channel))
        enhanced_img = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)
        return enhanced_img
    elif len(img_np.shape) == 2:  # Grayscale image
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_img = clahe.apply(img_np)
        return enhanced_img
    else:
        logging.warning(f"apply_clahe_filter: Unsupported image format with shape {img_np.shape}. Returning original.")
        return img_np


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    """Rotates an image by a given angle (90, 180, 270 degrees)."""
    if image is None or image.size == 0:
        logging.warning("rotate_image: Input image is empty.")
        return image  # Or raise error

    if angle == 0:
        return image
    elif angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        logging.warning(f"rotate_image: Unsupported angle {angle}. Returning original image.")
        return image


def is_valid_timecode_format(timecode_str: str) -> bool:
    """Checks if the timecode string is in HH:MM:SS or MM:SS format."""
    if not timecode_str:
        return False
    # Regex for HH:MM:SS or MM:SS
    # Allows for optional hours part
    pattern = r"^([0-5]?[0-9]:)?[0-5]?[0-9]:[0-5][0-9]$"
    return bool(re.match(pattern, timecode_str))


def timecode_to_frames(timecode_str: str, fps: float) -> int:
    """Converts HH:MM:SS or MM:SS timecode string to frame count."""
    if not is_valid_timecode_format(timecode_str):
        raise ValueError(f"Invalid timecode format: {timecode_str}")

    parts = list(map(int, timecode_str.split(':')))
    if len(parts) == 3:  # HH:MM:SS
        h, m, s = parts
    elif len(parts) == 2:  # MM:SS
        h = 0
        m, s = parts
    else:  # Should be caught by is_valid_timecode_format, but as a safeguard
        raise ValueError(f"Invalid timecode format: {timecode_str}")

    total_seconds = h * 3600 + m * 60 + s
    return int(total_seconds * fps)


def image_contrast(frame, method: str = None) -> float:
    """
    Calculate the contrast of an image using different methods.

    Args:
        frame: Image as numpy array (BGR or grayscale)
        method: "stddev", "laplacian", or None (uses config default)

    Returns:
        float: Contrast measure (higher = more contrast)
    """
    if method is None:
        method = getattr(config, 'CONTRAST_CALCULATION_METHOD', 'stddev')

    try:
        if frame is None:
            return 0.0

        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        if method == "stddev":
            # Standard deviation method (default)
            _, std_dev = cv2.meanStdDev(gray)
            return float(std_dev[0][0])
        elif method == "laplacian":
            # Laplacian variance method (good for focus/sharpness)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            return float(laplacian.var())
        else:
            logging.warning(f"Unknown contrast method '{method}', falling back to stddev")
            _, std_dev = cv2.meanStdDev(gray)
            return float(std_dev[0][0])

    except Exception as e:
        logging.error(f"Error calculating image contrast with method '{method}': {e}")
        return 0.0


def compare_frame_quality(
    frame1: np.ndarray, frame2: np.ndarray, text1: str, text2: str, user_stopwords: List[str]
) -> str:
    """
    Compare two frames and determine which has better quality for OCR.

    Args:
        frame1: First frame image
        frame2: Second frame image
        text1: OCR text from first frame
        text2: OCR text from second frame
        user_stopwords: Stopwords for text normalization

    Returns:
        "frame1", "frame2", or "similar" indicating which frame is better
    """  # Normalize texts for comparison
    norm_text1 = normalize_text_for_comparison(text1 or "", user_stopwords)
    norm_text2 = normalize_text_for_comparison(text2 or "", user_stopwords)

    # Calculate text similarity
    similarity = fuzz.token_sort_ratio(norm_text1, norm_text2)

    # Use dynamic threshold based on text length (longer texts need higher thresholds)
    avg_text_length = (len(norm_text1) + len(norm_text2)) // 2
    dynamic_threshold = calculate_dynamic_fuzzy_threshold(avg_text_length)

    if similarity < dynamic_threshold:
        return "different_text"  # Not comparable - different content

    # Calculate contrast for both frames
    contrast1 = image_contrast(frame1)
    contrast2 = image_contrast(frame2)

    contrast_diff = abs(contrast1 - contrast2)
    min_improvement = getattr(config, 'MIN_CONTRAST_IMPROVEMENT_THRESHOLD', 5.0)

    if contrast_diff < min_improvement:
        return "similar"  # Contrast difference too small to matter
    elif contrast1 > contrast2:
        return "frame1"
    else:
        return "frame2"


def sort_text_lines_by_bbox(text_lines: List[Tuple[Any, str, float]]) -> List[Tuple[Any, str, float]]:
    """
    Sort text lines by their bounding box positions.
    Priority: top-to-bottom first, then left-to-right for lines at similar heights.
    Now handles rotation-corrected bounding boxes properly.

    Args:
        text_lines: List of (bbox, text, confidence) tuples

    Returns:
        Sorted list of text lines
    """

    def get_sort_key(line_tuple):
        bbox, text, conf = line_tuple
        if not bbox:
            return (float('inf'), float('inf'))  # Put lines without bbox at the end

        try:
            # Calculate center Y coordinate for vertical ordering
            if isinstance(bbox, list) and len(bbox) >= 4:
                if isinstance(bbox[0], (list, tuple)):
                    # Format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    ys = [pt[1] for pt in bbox]
                    xs = [pt[0] for pt in bbox]
                    center_y = sum(ys) / len(ys)
                    center_x = sum(xs) / len(xs)
                else:
                    # Format: [x1, y1, x2, y2, ...]
                    coords = bbox
                    ys = coords[1::2]  # y coordinates
                    xs = coords[::2]  # x coordinates
                    center_y = sum(ys) / len(ys)
                    center_x = sum(xs) / len(xs)
            else:
                return (float('inf'), float('inf'))

            # Round center_y to group lines at similar heights
            # This helps handle slight variations in text baseline
            rounded_y = round(center_y / 10) * 10  # Group by 10-pixel intervals

            return (rounded_y, center_x)

        except Exception as e:
            logging.warning(f"Error calculating bbox sort key: {e}")
            return (float('inf'), float('inf'))

    try:
        sorted_lines = sorted(text_lines, key=get_sort_key)
        return sorted_lines
    except Exception as e:
        logging.warning(f"Error sorting text lines by bbox: {e}")
        return text_lines  # Return original if sorting fails


def format_problem_description(problem_types: List[str]) -> str:
    """
    Format problem types into a human-readable description.
    
    Args:
        problem_types: List of problem type strings
        
    Returns:
        Formatted description string
    """
    if not problem_types:
        return "No specific issues identified"
      # Map problem types to user-friendly descriptions
    problem_descriptions = {
        "empty_name": "❌ Empty or missing name",
        "unknown_role_group": "❓ Unknown or missing role group", 
        "missing_normalized_role": "⚠️ Missing normalized role group",
        "missing_is_person_flag": "🤖 Missing person/company classification",
        "concatenated_names": "🔗 Multiple names concatenated together",
        "imdb_name_not_found": "� Name not found in IMDB database",
        "invalid_imdb_name": "📋 Not found in IMDB database",
        "concatenated_names_detected": "🔗 Appears to be multiple names joined",
        "invalid_name_after_cleaning": "🧹 Name becomes invalid after cleaning",
        "no_valid_words": "📝 No valid words found in name",
        "too_short": "📏 Name too short to validate"
    }
    
    descriptions = []
    for problem_type in problem_types:
        description = problem_descriptions.get(problem_type, f"⚠️ {problem_type.replace('_', ' ').title()}")
        descriptions.append(description)
    
    if len(descriptions) == 1:
        return descriptions[0]
    elif len(descriptions) == 2:
        return f"{descriptions[0]} and {descriptions[1]}"
    else:
        return f"{', '.join(descriptions[:-1])}, and {descriptions[-1]}"


# ===========================
# Company Detection Functions
# ===========================

def is_company_role_group(role_group: Optional[str]) -> bool:
    """
    Check if a role group indicates a company rather than a person.
    
    Args:
        role_group: The role group to check
        
    Returns:
        True if the role group indicates a company
    """
    if not role_group:
        return False
    
    company_role_groups = {
        'Production Companies',
        'Distributors', 
        'Sales Representatives / ISA',
        'Special Effects Companies',
        'Miscellaneous Companies',
        'Miscellaneous Company',  # Handle singular form
    }
    
    # Check exact matches and variations
    role_group_normalized = role_group.strip()
    if role_group_normalized in company_role_groups:
        return True
    
    # Handle common variations
    company_keywords = [
        'companies', 'company', 'distributors', 'distributor', 
        'production', 'miscellaneous', 'effects'
    ]
    
    role_lower = role_group_normalized.lower()
    return any(keyword in role_lower for keyword in company_keywords)


def detect_company_name_patterns(name: str) -> bool:
    """
    Detect if a name looks like a company based on common patterns.
    
    Args:
        name: The name to analyze
        
    Returns:
        True if the name appears to be a company
    """
    if not name:
        return False
    
    name_lower = name.lower().strip()
    
    # Common company suffixes and patterns
    company_patterns = [
        r'\b(inc\.?|corp\.?|ltd\.?|llc|limited|corporation|incorporated)\b',
        r'\b(pictures|entertainment|studios?|films?|productions?)\b',
        r'\b(services?|group|media|television|tv|networks?)\b',
        r'\b(catering|rental|equipment|facilities)\b',
        r'\b(bros\.?|brothers|sisters|associates)\b',
    ]
    
    for pattern in company_patterns:
        if re.search(pattern, name_lower):
            return True
    
    return False


# ===========================
# Performance Caching Functions  
# ===========================

def get_cached_problematic_credits_count(episode_id: str) -> Optional[int]:
    """
    Get cached count of problematic credits for an episode.
    
    Args:
        episode_id: Episode to check
        
    Returns:
        Cached count or None if not cached/expired
    """
    cache_key = f"problem_count_{episode_id}"
    cache_time_key = f"problem_count_time_{episode_id}"
    
    if cache_key in st.session_state and cache_time_key in st.session_state:
        cache_time = st.session_state[cache_time_key]
        # Check if cache is still valid (30 seconds)
        if time.time() - cache_time < 30:
            return st.session_state[cache_key]
    
    return None


def cache_problematic_credits_count(episode_id: str, count: int) -> None:
    """
    Cache the count of problematic credits for an episode.
    
    Args:
        episode_id: Episode ID
        count: Number of problematic credits
    """
    cache_key = f"problem_count_{episode_id}"
    cache_time_key = f"problem_count_time_{episode_id}"
    
    st.session_state[cache_key] = count
    st.session_state[cache_time_key] = time.time()


def invalidate_credits_cache(episode_id: str) -> None:
    """
    Invalidate cached data for an episode when credits are saved.
    
    Args:
        episode_id: Episode ID to invalidate cache for
    """
    cache_keys_to_remove = [
        f"problem_count_{episode_id}",
        f"problem_count_time_{episode_id}",
        f"episode_stats_{episode_id}",
        f"episode_stats_time_{episode_id}"
    ]
    
    for key in cache_keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]
    
    logging.info(f"Invalidated cache for episode {episode_id}")


def identify_problematic_credits_fast(episode_id: str) -> int:
    """
    Fast version of problematic credits identification with caching.
    
    Args:
        episode_id: Episode to analyze
        
    Returns:
        Number of problematic credits
    """
    # Check cache first
    cached_count = get_cached_problematic_credits_count(episode_id)
    if cached_count is not None:
        logging.debug(f"Using cached problematic credits count for {episode_id}: {cached_count}")
        return cached_count
    
    try:
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        
        # Quick count query for performance
        cursor.execute(f"""
            SELECT COUNT(*) 
            FROM {config.DB_TABLE_CREDITS} 
            WHERE episode_id = ? 
            AND (reviewed_status IS NULL OR reviewed_status != 'kept')
            AND (
                role_group = 'Unknown' 
                OR role_group_normalized IS NULL 
                OR name IS NULL 
                OR name = ''
                OR (is_person IS NULL AND role_group NOT LIKE '%Compan%')
            )
        """, (episode_id,))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        # Cache the result
        cache_problematic_credits_count(episode_id, count)
        
        logging.info(f"Identified {count} problematic credits for episode {episode_id}")
        return count
        
    except Exception as e:
        logging.error(f"Error identifying problematic credits for {episode_id}: {e}", exc_info=True)
        return 0


def identify_problematic_credits(episode_id: str) -> List[Dict[str, Any]]:
    """
    Identify problematic credits for an episode that need manual review.
    Returns the actual credit objects (not just count).
    
    Args:
        episode_id: Episode to analyze
          Returns:
        List of problematic credit dictionaries
    """
    try:
        logging.info(f"[Problematic Credits] Starting identification for episode: {episode_id}")
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        
        # Get credits that need review
        cursor.execute(f"""
            SELECT id, episode_id, source_frame, role_group, name, role_detail, 
                   role_group_normalized, original_frame_number, scene_position, 
                   reviewed_status, is_person, normalized_name, is_present_in_imdb
            FROM {config.DB_TABLE_CREDITS} 
            WHERE episode_id = ? 
            AND (reviewed_status IS NULL OR reviewed_status != 'kept')
            ORDER BY role_group_normalized, name
        """, (episode_id,))
        
        credits_data = cursor.fetchall()
        conn.close()
        
        logging.info(f"[Problematic Credits] Found {len(credits_data)} credits that need review for episode: {episode_id}")
        if not credits_data:
            logging.info(f"[Problematic Credits] No credits to review for episode: {episode_id}")
            return []
        
        problematic_credits = []
        
        logging.info(f"[Problematic Credits] Processing {len(credits_data)} individual credits...")
        for i, credit in enumerate(credits_data):
            (credit_id, ep_id, source_frame, role_group, name, role_detail, 
             role_group_normalized, original_frame_number, scene_position, 
             reviewed_status, is_person, normalized_name, is_present_in_imdb) = credit
            
            logging.debug(f"[Problematic Credits] Processing credit {i+1}/{len(credits_data)}: ID={credit_id}, name='{name}', role='{role_group}', is_person={is_person}")
            
            # Skip company role groups from problematic credits identification
            if is_company_role_group(role_group):
                logging.debug(f"[Problematic Credits] Skipping company role group: '{role_group}' for name: '{name}'")
                continue
            
            problem_types = []
            
            # Check for various problematic conditions
            if not name or name.strip() == "":
                problem_types.append("empty_name")
            
            if not role_group or role_group == "Unknown":
                problem_types.append("unknown_role_group")
            
            if not role_group_normalized:
                problem_types.append("missing_normalized_role")
            # Check for potential company validation issues
            if is_person is None and not is_company_role_group(role_group):
                problem_types.append("missing_is_person_flag")
                
            # Check for potentially problematic names (very long names)
            if name and len(name.strip().split()) > 6:
                problem_types.append("concatenated_names")            # Check if name is NOT found in IMDB database (using pre-computed result from Step 4)
            # Only check for persons, not companies
            if name and name.strip() and is_person not in [False, 0]:
               
                if is_present_in_imdb == False:  # Use == instead of is for SQLite integer comparison
                    logging.debug(f"[IMDB Check] Name '{name}' NOT FOUND in IMDB database (pre-computed)")
                    problem_types.append("imdb_name_not_found")
                elif is_present_in_imdb == True:  # Use == instead of is for SQLite integer comparison
                    logging.debug(f"[IMDB Check] Name '{name}' is VALID in IMDB database (pre-computed)")
                elif is_present_in_imdb is None:
                    logging.warning(f"[IMDB Check] Name '{name}' not yet processed by Step 4 IMDB validation")
                    # Optionally add a flag to indicate this needs Step 4 processing
                    problem_types.append("imdb_not_processed")
            
            # If any problems found, add to problematic list
            if problem_types:
                problematic_credit = {
                    'id': credit_id,
                    'episode_id': ep_id,
                    'source_frame': source_frame,
                    'role_group': role_group,
                    'name': name,
                    'role_detail': role_detail,
                    'role_group_normalized': role_group_normalized,
                    'original_frame_number': original_frame_number,
                    'scene_position': scene_position,
                    'reviewed_status': reviewed_status,
                    'is_person': is_person,
                    'normalized_name': normalized_name,
                    'problem_types': problem_types,                    'priority_score': len(problem_types) * 10  # Higher score for more problems
                }
                problematic_credits.append(problematic_credit)
        
        # Sort by priority score (most problematic first)
        problematic_credits.sort(key=lambda x: x['priority_score'], reverse=True)
        
        logging.info(f"[Problematic Credits] FINAL RESULTS for episode {episode_id}:")
        logging.info(f"[Problematic Credits] Total processed: {len(credits_data)} credits")
        logging.info(f"[Problematic Credits] Found problematic: {len(problematic_credits)} credits")
        
        # Log summary of problem types
        problem_type_counts = {}
        for pc in problematic_credits:
            for pt in pc['problem_types']:
                problem_type_counts[pt] = problem_type_counts.get(pt, 0) + 1
        
        logging.info(f"[Problematic Credits] Problem type summary: {problem_type_counts}")
        
        return problematic_credits
        
    except Exception as e:
        logging.error(f"Error identifying problematic credits for {episode_id}: {e}", exc_info=True)
        return []


def get_best_frames_for_credit(credit: Dict[str, Any], max_frames: int = 2) -> List[str]:
    """
    Get the best frame filenames for displaying a credit.
    
    Args:
        credit: Credit dictionary containing frame information
        max_frames: Maximum number of frames to return
        
    Returns:
        List of frame filenames
    """
    try:
        source_frames = credit.get('source_frame', [])
        if isinstance(source_frames, str):
            source_frames = source_frames.split(',')
        elif not isinstance(source_frames, list):
            source_frames = [str(source_frames)] if source_frames else []
        
        # Clean and limit the frames
        frames = [frame.strip() for frame in source_frames if frame and frame.strip()]
        return frames[:max_frames]
        
    except Exception as e:
        logging.error(f"Error getting best frames for credit: {e}")
        return []


def find_frame_path(episode_id: str, frame_filename: str) -> Optional[Path]:
    """
    Find the full path to a frame file for an episode.
    
    Args:
        episode_id: Episode identifier
        frame_filename: Name of the frame file
        
    Returns:
        Path to the frame file or None if not found
    """
    try:
        # Common frame directories to search
        episode_dir = config.EPISODES_BASE_DIR / episode_id
        frame_dirs = [
            episode_dir / "analysis" / "frames",
            episode_dir / "analysis" / "step1_representative_frames", 
            episode_dir / "analysis" / "skipped_frames",
            episode_dir / "frames",  # Fallback
        ]
        
        for frame_dir in frame_dirs:
            if frame_dir.exists():
                frame_path = frame_dir / frame_filename
                if frame_path.exists():
                    return frame_path
        
        logging.warning(f"Frame file not found: {frame_filename} for episode {episode_id}")
        return None
        
    except Exception as e:
        logging.error(f"Error finding frame path for {frame_filename}: {e}")
        return None


import unicodedata

def normalize_name(name):
    """
    Normalize names by:
    - Converting to lowercase.
    - Removing titles.
    - Removing punctuation.
    - Removing accents.
    - Removing extra spaces.
    """
    if not isinstance(name, str):
        name = str(name)

    # Convert to lowercase
    name = name.lower()
    
    # Remove titles
    name = re.sub(
        r"\b("
        r"dr\.|dott\.|dott\.ssa|prof\.|prof\.ssa|ing\.|arch\.|avv\.|sig\.|sig\.ra|sig\.na|"
        r"mr\.|mrs\.|ms\.|mx\.|fr\.|rev\.|hon\.|sen\.|rep\.|gov\.|pres\.|vp\.|"
        r"capt\.|cmdr\.|lt\.|col\.|maj\.|gen\.|adm\.|"
        r"msgr\.|sr\.|sra\.|srta\.|srs\.|"
        r"mlle\.|mme\.|mons\.|pr\.|amb\.|pm\.|"
        r"ph\.?d|m\.?d|esq\.|emo\.|eccmo\.|p\.i|geom\."
        r")\s+",
        "", 
        name,
        flags=re.IGNORECASE
    )

    # Normalize unicode to decompose accented characters
    name = unicodedata.normalize('NFD', name)
    # Remove diacritical marks (accents)
    name = ''.join(char for char in name if unicodedata.category(char) != 'Mn')
    
    # Remove punctuation but preserve spaces
    name = re.sub(r"[.\-_,']", "", name)
    
    # Remove any extra whitespace and normalize to single spaces
    name = re.sub(r"\s+", " ", name).strip()
    
    return name

# ===========================
# End of Utils Module
# ===========================
