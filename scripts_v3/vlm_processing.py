import base64
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from mimetypes import guess_type
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

try:
    from openai import APIConnectionError, APIStatusError, AzureOpenAI, OpenAI, RateLimitError
except ImportError:
    AzureOpenAI = OpenAI = None
    APIStatusError = APIConnectionError = RateLimitError = None
    logging.warning("Azure OpenAI library not found. Azure VLM processing will not be available.")

try:
    from anthropic import AnthropicFoundry
    # Set Anthropic library logging to INFO to avoid debug spam
    import anthropic
    logging.getLogger('anthropic').setLevel(logging.INFO)
except ImportError:
    AnthropicFoundry = None
    logging.warning("Anthropic library not found. Claude VLM processing will not be available.")

import importlib.util

# llama_cpp is imported lazily (inside _get_gemma12b_client), not here: importing it
# eagerly loads its CUDA DLLs (cublas/cudart) into the process, which can shadow the
# differently-versioned copies PaddleOCR bundles (same DLL basenames, Windows resolves
# by name against whatever is already loaded) and break PaddleOCR init with WinError 127.
# find_spec only locates the module without executing it, so availability can still be
# checked here without paying that cost.
_LLAMA_CPP_AVAILABLE = importlib.util.find_spec("llama_cpp") is not None
if not _LLAMA_CPP_AVAILABLE:
    logging.warning("llama-cpp-python not found. gemma12b (local GGUF) VLM processing will not be available.")

from scripts_v3 import config, utils
# Note: Role corrections are now applied after DB save via utils.apply_role_group_corrections_to_database()

# Exponential backoff settings for Azure API retries
# 429 (NoCapacity) errors require longer waits during peak load
MAX_API_RETRIES = 3
BACKOFF_FACTOR = 4.0

# gemma12b: local gemma-4-12b vision GGUF run in-process via llama-cpp-python.
# Loading the model is expensive (~15-20s + several GB VRAM), so keep one instance
# per (model, mmproj, gpu_layers, ctx) alive across episodes in the same process.
_GEMMA12B_CLIENT_CACHE: Dict[tuple, Any] = {}


def _gemma12b_config() -> Tuple[str, str, int, int]:
    """Resolve gemma12b GGUF paths + runtime knobs from env, defaulting to bin/."""
    bin_dir = config.PROJECT_ROOT / "bin"
    model_path = os.getenv(config.AzureConfig.GEMMA12B_MODEL_GGUF_ENV) or str(bin_dir / "gemma-4-12b-it-qat-q4_0.gguf")
    mmproj_path = os.getenv(config.AzureConfig.GEMMA12B_MMPROJ_GGUF_ENV) or str(bin_dir / "mmproj-gemma-4-12b-it-qat-q4_0.gguf")
    n_gpu_layers = int(os.getenv(config.AzureConfig.GEMMA12B_N_GPU_LAYERS_ENV) or -1)  # -1 = all layers on GPU
    # ~11k-token prompt + image + output must fit; keep well above that but small
    # enough for the KV cache to fit 16 GB VRAM (paired with flash_attn below).
    n_ctx = int(os.getenv(config.AzureConfig.GEMMA12B_N_CTX_ENV) or 16384)
    return model_path, mmproj_path, n_gpu_layers, n_ctx


def _get_gemma12b_client(model_path: str, mmproj_path: str, n_gpu_layers: int, n_ctx: int):
    """Build (or reuse a cached) in-process llama.cpp gemma-4-12b vision model."""
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Gemma4ChatHandler

    key = (model_path, mmproj_path, n_gpu_layers, n_ctx)
    cached = _GEMMA12B_CLIENT_CACHE.get(key)
    if cached is not None:
        return cached
    chat_handler = Gemma4ChatHandler(clip_model_path=mmproj_path, verbose=False)
    client = Llama(
        model_path=model_path,
        chat_handler=chat_handler,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        flash_attn=True,  # uses reduced sliding-window KV cache; avoids OOM on 16 GB
        verbose=False,
    )
    _GEMMA12B_CLIENT_CACHE[key] = client
    return client

# GPT Terra / GPT Sol (gpt-5.6 family) use the Responses API with configurable reasoning.
# effort: low | medium | high | xhigh | max ; mode "pro" enables the deep-reasoning path,
# "standard" the lighter one.
VLM_TERRA_REASONING = {"mode": "pro", "effort": "high"}
# Lighter reasoning config shared by every "_standard" variant (Terra Standard,
# Sol Standard): same deployment as the pro/high sibling, default/lighter reasoning.
VLM_STANDARD_REASONING = {"mode": "standard", "effort": "medium"}

# Providers that talk to the Responses API, and the reasoning config each one uses.
VLM_REASONING_BY_PROVIDER = {
    "azure_gpt_terra": VLM_TERRA_REASONING,
    "azure_gpt_terra_standard": VLM_STANDARD_REASONING,
    "azure_gpt_sol": VLM_TERRA_REASONING,
    "azure_gpt_sol_standard": VLM_STANDARD_REASONING,
}
RESPONSES_API_PROVIDERS = tuple(VLM_REASONING_BY_PROVIDER.keys())


def _response_to_jsonable(response: Any) -> Any:
    """Best-effort conversion of an SDK response object into a JSON-serializable
    structure capturing the full payload returned by the LLM."""
    # llama-cpp-python (gemma12b) already returns a plain dict.
    if isinstance(response, (dict, list)):
        return response
    for attr, kwargs in (("model_dump", {"mode": "json"}), ("to_dict", {}), ("model_dump", {}), ("dict", {})):
        fn = getattr(response, attr, None)
        if callable(fn):
            try:
                return fn(**kwargs)
            except TypeError:
                try:
                    return fn()
                except Exception:
                    continue
            except Exception:
                continue
    dump_json = getattr(response, "model_dump_json", None)
    if callable(dump_json):
        try:
            return json.loads(dump_json())
        except Exception:
            pass
    return {"repr": str(response)}


def _chat_part_to_responses_part(part: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a Chat Completions content part into a Responses API content part."""
    ptype = part.get("type")
    if ptype == "text":
        return {"type": "input_text", "text": part.get("text", "")}
    if ptype == "image_url":
        img = part.get("image_url", {}) or {}
        out = {"type": "input_image", "image_url": img.get("url")}
        if img.get("detail"):
            out["detail"] = img["detail"]
        return out
    return part


@dataclass
class ChatCompletionRequest:
    model: str
    messages: List[Dict[str, Any]]
    max_tokens: int
    temperature: float


@dataclass
class ChatCompletionResponse:
    content: str


def local_image_to_data_url(image_path: Path) -> Optional[str]:
    """Encodes a local image file into a base64 data URL."""
    try:
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = "image/jpeg"
            logging.warning(f"Could not guess MIME type for {image_path}, defaulting to {mime_type}")

        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

        return f"data:{mime_type};base64,{base64_encoded_data}"
    except Exception as e:
        logging.error(f"Error encoding image {image_path} to data URL: {e}")
        return None


def resolve_vlm_provider(vlm_provider: str = "auto") -> str:
    """Resolve a requested VLM provider to a concrete provider id.

    Returns one of 'claude', 'azure_gpt_terra', 'azure_gpt_terra_standard',
    'azure_gpt_sol', 'azure_gpt_sol_standard', 'azure_gpt5',
    'gemma-4-26b'.
    Accepts the explicit ids plus 'auto' (preference order:
    GPT Terra -> GPT Sol -> Claude -> GPT 5.2 -> LM Studio; the "_standard"
    reasoning variants are explicit-select only, never picked by auto) and the legacy
    'azure' alias (GPT 5.2 only). Raises ValueError if the requested provider (or any
    provider, for 'auto') is not available. This is the single source of truth
    shared by run_azure_vlm_ocr_on_frames() and the app's folder resolution.
    """
    # Credentials
    claude_endpoint = os.getenv("CLAUDE_ENDPOINT")
    claude_api_key = os.getenv("CLAUDE_API_KEY")
    claude_model = os.getenv("CLAUDE_MODEL_DEPLOYMENT_NAME")

    azure_api_key = os.getenv(config.AzureConfig.API_KEY_ENV)
    gpt5_endpoint = os.getenv(config.AzureConfig.GPT5_ENDPOINT_ENV)
    gpt5_deployment = os.getenv(config.AzureConfig.GPT5_DEPLOYMENT_NAME_ENV)
    gpt_terra_endpoint = os.getenv(config.AzureConfig.GPT_TERRA_ENDPOINT_ENV)
    gpt_terra_deployment = os.getenv(config.AzureConfig.GPT_TERRA_DEPLOYMENT_NAME_ENV)
    gpt_terra_api_key = os.getenv(config.AzureConfig.GPT_TERRA_API_KEY_ENV)
    # GPT Sol shares Terra's key/endpoint; only the deployment name is its own
    gpt_sol_deployment = os.getenv(config.AzureConfig.GPT_SOL_DEPLOYMENT_NAME_ENV)

    # LM Studio (local OpenAI-compatible server)
    lmstudio_base_url = os.getenv(config.AzureConfig.LMSTUDIO_BASE_URL_ENV) or "http://localhost:1234/v1"
    lmstudio_model = os.getenv(config.AzureConfig.LMSTUDIO_MODEL_ENV) or "google/gemma-4-26b-a4b-qat"

    # Availability
    claude_available = bool(claude_endpoint and claude_api_key and claude_model and AnthropicFoundry)
    gpt5_available = bool(azure_api_key and gpt5_endpoint and gpt5_deployment and OpenAI)
    gpt_terra_available = bool(gpt_terra_api_key and gpt_terra_endpoint and gpt_terra_deployment and AzureOpenAI)
    gpt_sol_available = bool(gpt_terra_api_key and gpt_terra_endpoint and gpt_sol_deployment and AzureOpenAI)
    lmstudio_available = bool(lmstudio_model and OpenAI)
    gemma12b_model_path, gemma12b_mmproj_path, _, _ = _gemma12b_config()
    gemma12b_available = bool(_LLAMA_CPP_AVAILABLE and os.path.isfile(gemma12b_model_path) and os.path.isfile(gemma12b_mmproj_path))

    if vlm_provider == "claude":
        if not claude_available:
            raise ValueError("Claude provider requested but credentials not available or library not installed")
        return "claude"
    if vlm_provider == "azure_gpt_terra":
        if not gpt_terra_available:
            missing = []
            if not gpt_terra_api_key: missing.append("GPT_TERRA_AZURE_OPENAI_KEY")
            if not gpt_terra_endpoint: missing.append("GPT_TERRA_AZURE_OPENAI_ENDPOINT")
            if not gpt_terra_deployment: missing.append("GPT_TERRA_AZURE_OPENAI_DEPLOYMENT_NAME")
            if not AzureOpenAI: missing.append("openai library (AzureOpenAI class)")
            raise ValueError(f"Azure GPT Terra provider requested but credentials not available or library not installed. Missing: {', '.join(missing)}")
        return "azure_gpt_terra"
    if vlm_provider == "azure_gpt_terra_standard":
        if not gpt_terra_available:
            missing = []
            if not gpt_terra_api_key: missing.append("GPT_TERRA_AZURE_OPENAI_KEY")
            if not gpt_terra_endpoint: missing.append("GPT_TERRA_AZURE_OPENAI_ENDPOINT")
            if not gpt_terra_deployment: missing.append("GPT_TERRA_AZURE_OPENAI_DEPLOYMENT_NAME")
            if not AzureOpenAI: missing.append("openai library (AzureOpenAI class)")
            raise ValueError(f"Azure GPT Terra Standard provider requested but credentials not available or library not installed. Missing: {', '.join(missing)}")
        return "azure_gpt_terra_standard"
    if vlm_provider == "azure_gpt_sol":
        if not gpt_sol_available:
            missing = []
            if not gpt_terra_api_key: missing.append("GPT_TERRA_AZURE_OPENAI_KEY")
            if not gpt_terra_endpoint: missing.append("GPT_TERRA_AZURE_OPENAI_ENDPOINT")
            if not gpt_sol_deployment: missing.append("GPT_SOL_AZURE_OPENAI_DEPLOYMENT_NAME")
            if not AzureOpenAI: missing.append("openai library (AzureOpenAI class)")
            raise ValueError(f"Azure GPT Sol provider requested but credentials not available or library not installed. Missing: {', '.join(missing)}")
        return "azure_gpt_sol"
    if vlm_provider == "azure_gpt_sol_standard":
        if not gpt_sol_available:
            missing = []
            if not gpt_terra_api_key: missing.append("GPT_TERRA_AZURE_OPENAI_KEY")
            if not gpt_terra_endpoint: missing.append("GPT_TERRA_AZURE_OPENAI_ENDPOINT")
            if not gpt_sol_deployment: missing.append("GPT_SOL_AZURE_OPENAI_DEPLOYMENT_NAME")
            if not AzureOpenAI: missing.append("openai library (AzureOpenAI class)")
            raise ValueError(f"Azure GPT Sol Standard provider requested but credentials not available or library not installed. Missing: {', '.join(missing)}")
        return "azure_gpt_sol_standard"
    if vlm_provider == "azure_gpt5":
        if not gpt5_available:
            missing = []
            if not azure_api_key: missing.append("AZURE_OPENAI_KEY")
            if not gpt5_endpoint: missing.append("GPT_5_AZURE_OPENAI_ENDPOINT")
            if not gpt5_deployment: missing.append("GPT_5_AZURE_OPENAI_DEPLOYMENT_NAME")
            if not OpenAI: missing.append("openai library (OpenAI class)")
            raise ValueError(f"Azure GPT_5 provider requested but credentials not available or library not installed. Missing: {', '.join(missing)}")
        return "azure_gpt5"
    if vlm_provider == "gemma-4-26b":
        if not lmstudio_available:
            missing = []
            if not lmstudio_model: missing.append("LMSTUDIO_VLM_MODEL")
            if not OpenAI: missing.append("openai library (OpenAI class)")
            raise ValueError(f"gemma-4-26b (LM Studio) provider requested but model not available or library not installed. Missing: {', '.join(missing)}")
        return "gemma-4-26b"
    if vlm_provider == "gemma12b":
        if not gemma12b_available:
            missing = []
            if not _LLAMA_CPP_AVAILABLE: missing.append("llama-cpp-python")
            if not os.path.isfile(gemma12b_model_path): missing.append(f"model GGUF ({gemma12b_model_path})")
            if not os.path.isfile(gemma12b_mmproj_path): missing.append(f"mmproj GGUF ({gemma12b_mmproj_path})")
            raise ValueError(f"gemma12b (local GGUF) provider requested but not available. Missing: {', '.join(missing)}")
        return "gemma12b"
    if vlm_provider == "azure":  # Legacy/Generic Azure request -> GPT 5.2
        if gpt5_available:
            return "azure_gpt5"
        raise ValueError("Azure provider requested but GPT 5.2 credentials not available")
    # auto
    if gpt_terra_available:
        return "azure_gpt_terra"
    if gpt_sol_available:
        return "azure_gpt_sol"
    if claude_available:
        return "claude"
    if gpt5_available:
        return "azure_gpt5"
    if lmstudio_available:
        return "gemma-4-26b"
    raise ValueError("No VLM provider available (neither Claude, Azure nor LM Studio credentials set)")


def get_vlm_ocr_dir(episode_id: str, provider: str) -> Path:
    """Return the provider-specific OCR output directory for an episode."""
    subdir = config.VLM_OCR_DIR_BY_PROVIDER.get(provider, config.VLM_OCR_DIR_DEFAULT)
    return config.EPISODES_BASE_DIR / episode_id / subdir


def run_azure_vlm_ocr_on_frames(
    episode_id: str, max_new_tokens: int, vlm_provider: str = "auto", enable_role_correction: bool = True, include_previous_frame: bool = True
) -> Tuple[int, str, Optional[str]]:
    """
    Runs VLM OCR on selected frames for an episode, one frame at a time,
    processes results, and saves them.

    Args:
        episode_id: The ID of the episode.
        max_new_tokens: The maximum number of new tokens for generation.
        vlm_provider: VLM provider to use ('auto', 'claude', 'azure'). Default: 'auto'
        enable_role_correction: Whether to apply role group corrections. Default: True
        include_previous_frame: Whether to include previous frame as visual context. Default: True

    Returns:
        A tuple containing:
        - Count of newly processed credit entries added.
        - Status string ('completed', 'error', 'skipped_no_frames', etc.).
        - Error message string (or None if successful).
    """
    if AzureOpenAI is None:
        msg = "Azure OpenAI library not installed."
        logging.error(f"[{episode_id}] {msg}")
        return 0, "error_missing_dependency", msg

    logging.info(f"[{episode_id}] Starting Azure VLM OCR processing (Single Frame Mode).")
    episode_dir = config.EPISODES_BASE_DIR / episode_id

    # Resolve the concrete provider up-front so the OCR folder is provider-specific.
    # Each provider reads/writes its own folder (no cross-provider JSON reuse).
    try:
        selected_provider = resolve_vlm_provider(vlm_provider)
    except Exception as resolve_err:
        msg = f"Failed to resolve VLM provider: {resolve_err}"
        logging.error(f"[{episode_id}] {msg}")
        return 0, "error_vlm_client_init", msg
    vlm_provider = selected_provider

    ocr_dir = get_vlm_ocr_dir(episode_id, selected_provider)
    frames_dir = episode_dir / 'analysis' / 'frames'
    output_json_path = ocr_dir / f"{episode_id}_credits_azure_vlm.json"

    try:
        ocr_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        msg = f"Failed to create OCR output directory {ocr_dir}: {e}"
        logging.error(f"[{episode_id}] {msg}", exc_info=True)
        return 0, "error_creating_output_dir", msg

    all_frames_info = []
    if not frames_dir.is_dir():
        msg = f"Frames directory not found: {frames_dir}"
        logging.warning(f"[{episode_id}] {msg}")

    else:
        logging.info(f"[{episode_id}] Scanning for frames in {frames_dir}...")
        found_frame_files = sorted(list(frames_dir.glob('*.jpg')))
        for image_path in found_frame_files:
            frame_info = {"path": image_path, "filename": image_path.name}

            try:
                timecode_str = image_path.stem.split('_')[-1]
                if timecode_str.count('-') == 3:
                    frame_info["timecode"] = timecode_str.replace('-', ':', 2).replace('-', ',', 1)
                else:
                    frame_info["timecode"] = "N/A"
            except Exception:
                frame_info["timecode"] = "N/A"
            all_frames_info.append(frame_info)

    if not all_frames_info:
        msg = f"No frame image files found in {frames_dir}."
        logging.warning(f"[{episode_id}] {msg}")

        if not output_json_path.exists():
            return 0, "skipped_no_frames_found", msg
        else:
            logging.info(f"[{episode_id}] No new frames found, but existing output file present. Assuming completed.")
            return 0, "completed_no_new_frames", None

    logging.info(f"[{episode_id}] Found {len(all_frames_info)} total frames for potential processing.")

    # Check which frames are already fully processed in the database
    # A frame is fully processed if all its credits (name-frame pairs) exist in DB
    processed_frames_set = set()
    incomplete_frames_to_clean = set()
    
    # Load existing credits from JSON to know what should be in DB for each frame
    json_credits_by_frame = {}
    if output_json_path.exists():
        try:
            with open(output_json_path, 'r', encoding='utf-8') as f:
                existing_credits = json.load(f)
                for credit in existing_credits:
                    frames = credit.get('source_frame', [])
                    if isinstance(frames, str):
                        frames = [frames]
                    elif not isinstance(frames, list):
                        frames = []
                    
                    name = credit.get('name', '')
                    for frame_filename in frames:
                        if frame_filename not in json_credits_by_frame:
                            json_credits_by_frame[frame_filename] = []
                        json_credits_by_frame[frame_filename].append(name)
            
            logging.info(f"[{episode_id}] Loaded JSON with credits for {len(json_credits_by_frame)} frames")
            
            # Now check database to see which frames are fully processed
            try:
                conn = sqlite3.connect(config.DB_PATH)
                cursor = conn.cursor()
                
                for frame_filename, expected_names in json_credits_by_frame.items():
                    # Get all names in DB for this frame
                    cursor.execute(
                        f"""SELECT DISTINCT name FROM {config.DB_TABLE_CREDITS} 
                        WHERE episode_id = ? AND source_frame LIKE ?""",
                        (episode_id, f"%{frame_filename}%")
                    )
                    db_names = {row[0] for row in cursor.fetchall()}
                    
                    # Check if all expected names are in DB
                    if set(expected_names).issubset(db_names):
                        processed_frames_set.add(frame_filename)
                        logging.debug(f"[{episode_id}] Frame {frame_filename} fully processed in DB")
                    else:
                        missing = set(expected_names) - db_names
                        if db_names:  # Has some credits but incomplete
                            incomplete_frames_to_clean.add(frame_filename)
                            logging.info(f"[{episode_id}] Frame {frame_filename} incomplete in DB (missing: {missing}) - will clean and reprocess")
                        else:
                            logging.debug(f"[{episode_id}] Frame {frame_filename} not in DB - will process")
                
                # Clean incomplete frames from database
                if incomplete_frames_to_clean:
                    logging.info(f"[{episode_id}] Cleaning {len(incomplete_frames_to_clean)} incomplete frames from database")
                    for frame_filename in incomplete_frames_to_clean:
                        cursor.execute(
                            f"""DELETE FROM {config.DB_TABLE_CREDITS} 
                            WHERE episode_id = ? AND source_frame LIKE ?""",
                            (episode_id, f"%{frame_filename}%")
                        )
                        deleted = cursor.rowcount
                        logging.info(f"[{episode_id}] Deleted {deleted} incomplete credits for frame {frame_filename}")
                    
                    conn.commit()
                    logging.info(f"[{episode_id}] Database cleanup completed for incomplete frames")
                
                conn.close()
                logging.info(f"[{episode_id}] Found {len(processed_frames_set)} fully processed frames in database")
                
            except Exception as db_err:
                logging.warning(f"[{episode_id}] Could not check database for processed frames: {db_err}")
                # Fallback: treat all frames in JSON as processed
                processed_frames_set = set(json_credits_by_frame.keys())
            
            # Clean incomplete frames from JSON
            if incomplete_frames_to_clean:
                logging.info(f"[{episode_id}] Cleaning {len(incomplete_frames_to_clean)} incomplete frames from JSON")
                cleaned_credits = []
                for credit in existing_credits:
                    frames = credit.get('source_frame', [])
                    if isinstance(frames, str):
                        frames = [frames]
                    elif not isinstance(frames, list):
                        frames = []
                    
                    # Keep credits that don't belong to incomplete frames
                    if not any(frame in incomplete_frames_to_clean for frame in frames):
                        cleaned_credits.append(credit)
                
                # Save cleaned JSON
                try:
                    with open(output_json_path, 'w', encoding='utf-8') as f:
                        json.dump(cleaned_credits, f, ensure_ascii=False, indent=2)
                    logging.info(f"[{episode_id}] Cleaned JSON saved with {len(cleaned_credits)} credits (removed {len(existing_credits) - len(cleaned_credits)} incomplete credits)")
                except Exception as write_err:
                    logging.error(f"[{episode_id}] Failed to save cleaned JSON: {write_err}")
                
        except Exception as e:
            logging.warning(f"[{episode_id}] Could not load processed frames from {output_json_path}: {e}")
    
    logging.info(
        f"[{episode_id}] Found {len(processed_frames_set)} already fully processed frame filenames (verified in DB)."
    )

    # Initialize VLM client based on user preference or auto-detect
    try:
        # Get credentials
        claude_endpoint = os.getenv("CLAUDE_ENDPOINT")
        claude_api_key = os.getenv("CLAUDE_API_KEY")
        claude_model = os.getenv("CLAUDE_MODEL_DEPLOYMENT_NAME")
        
        # Azure Common
        azure_api_key = os.getenv(config.AzureConfig.API_KEY_ENV)
        azure_api_version = os.getenv(config.AzureConfig.API_VERSION_ENV, config.AzureConfig.DEFAULT_API_VERSION)
        
        # GPT 5.2
        gpt5_endpoint = os.getenv(config.AzureConfig.GPT5_ENDPOINT_ENV)
        gpt5_deployment = os.getenv(config.AzureConfig.GPT5_DEPLOYMENT_NAME_ENV)

        # GPT Terra (dedicated key + api_version)
        gpt_terra_endpoint = os.getenv(config.AzureConfig.GPT_TERRA_ENDPOINT_ENV)
        gpt_terra_deployment = os.getenv(config.AzureConfig.GPT_TERRA_DEPLOYMENT_NAME_ENV)
        gpt_terra_api_key = os.getenv(config.AzureConfig.GPT_TERRA_API_KEY_ENV)
        gpt_terra_api_version = os.getenv(config.AzureConfig.GPT_TERRA_API_VERSION_ENV, config.AzureConfig.DEFAULT_API_VERSION)

        # GPT Sol (shares Terra's key/endpoint/api_version; dedicated deployment)
        gpt_sol_deployment = os.getenv(config.AzureConfig.GPT_SOL_DEPLOYMENT_NAME_ENV)

        # LM Studio (local OpenAI-compatible server)
        lmstudio_base_url = os.getenv(config.AzureConfig.LMSTUDIO_BASE_URL_ENV) or "http://localhost:1234/v1"
        lmstudio_model = os.getenv(config.AzureConfig.LMSTUDIO_MODEL_ENV) or "google/gemma-4-26b-a4b-qat"

        # Check availability
        claude_available = bool(claude_endpoint and claude_api_key and claude_model and AnthropicFoundry)
        gpt5_available = bool(azure_api_key and gpt5_endpoint and gpt5_deployment and OpenAI)
        gpt_terra_available = bool(gpt_terra_api_key and gpt_terra_endpoint and gpt_terra_deployment and AzureOpenAI)
        gpt_sol_available = bool(gpt_terra_api_key and gpt_terra_endpoint and gpt_sol_deployment and AzureOpenAI)
        lmstudio_available = bool(lmstudio_model and OpenAI)
        gemma12b_model_path, gemma12b_mmproj_path, gemma12b_n_gpu_layers, gemma12b_n_ctx = _gemma12b_config()
        gemma12b_available = bool(_LLAMA_CPP_AVAILABLE and os.path.isfile(gemma12b_model_path) and os.path.isfile(gemma12b_mmproj_path))

        # Debug logging for availability
        logging.debug(f"[{episode_id}] VLM Availability Check:")
        logging.debug(f"  Claude: {claude_available} (Key: {bool(claude_api_key)}, Endpoint: {bool(claude_endpoint)}, Model: {bool(claude_model)}, Lib: {bool(AnthropicFoundry)})")
        logging.debug(f"  GPT 5.2: {gpt5_available} (Key: {bool(azure_api_key)}, Endpoint: {bool(gpt5_endpoint)}, Model: {bool(gpt5_deployment)}, Lib: {bool(OpenAI)})")
        logging.debug(f"  GPT Terra: {gpt_terra_available} (Key: {bool(gpt_terra_api_key)}, Endpoint: {bool(gpt_terra_endpoint)}, Model: {bool(gpt_terra_deployment)}, Lib: {bool(AzureOpenAI)})")
        logging.debug(f"  GPT Sol: {gpt_sol_available} (Key: {bool(gpt_terra_api_key)}, Endpoint: {bool(gpt_terra_endpoint)}, Model: {bool(gpt_sol_deployment)}, Lib: {bool(AzureOpenAI)})")
        logging.debug(f"  LM Studio: {lmstudio_available} (Model: {bool(lmstudio_model)}, Lib: {bool(OpenAI)})")
        logging.debug(f"  gemma12b: {gemma12b_available} (Model: {os.path.isfile(gemma12b_model_path)}, mmproj: {os.path.isfile(gemma12b_mmproj_path)}, Lib: {_LLAMA_CPP_AVAILABLE})")
        
        # Provider was already resolved via resolve_vlm_provider() above;
        # build the client for that concrete provider.

        # Initialize client
        # active_api_version is recorded in per-credit metadata (None for clients
        # that don't take an api_version, e.g. Claude and the GPT-5 base_url client).
        active_api_version = None
        if selected_provider == "claude":
            logging.info(f"[{episode_id}] Using Claude model: {claude_model}")
            client = AnthropicFoundry(api_key=claude_api_key, base_url=claude_endpoint)
            deployment_name = claude_model
            vlm_provider = "claude"

        elif selected_provider == "azure_gpt_terra":
            logging.info(f"[{episode_id}] Using Azure GPT Terra model: {gpt_terra_deployment}")
            # GPT Terra uses AzureOpenAI client with its own dedicated key + api_version
            client = AzureOpenAI(api_key=gpt_terra_api_key, api_version=gpt_terra_api_version, azure_endpoint=gpt_terra_endpoint)
            deployment_name = gpt_terra_deployment
            active_api_version = gpt_terra_api_version
            vlm_provider = "azure_gpt_terra"

        elif selected_provider == "azure_gpt_terra_standard":
            logging.info(f"[{episode_id}] Using Azure GPT Terra Standard model: {gpt_terra_deployment} (reasoning standard/medium)")
            # Same deployment/credentials as GPT Terra, only the reasoning config differs
            client = AzureOpenAI(api_key=gpt_terra_api_key, api_version=gpt_terra_api_version, azure_endpoint=gpt_terra_endpoint)
            deployment_name = gpt_terra_deployment
            active_api_version = gpt_terra_api_version
            vlm_provider = "azure_gpt_terra_standard"

        elif selected_provider == "azure_gpt_sol":
            logging.info(f"[{episode_id}] Using Azure GPT Sol model: {gpt_sol_deployment}")
            # GPT Sol reuses Terra's AzureOpenAI credentials (key/endpoint/api_version)
            client = AzureOpenAI(api_key=gpt_terra_api_key, api_version=gpt_terra_api_version, azure_endpoint=gpt_terra_endpoint)
            deployment_name = gpt_sol_deployment
            active_api_version = gpt_terra_api_version
            vlm_provider = "azure_gpt_sol"

        elif selected_provider == "azure_gpt_sol_standard":
            logging.info(f"[{episode_id}] Using Azure GPT Sol Standard model: {gpt_sol_deployment} (reasoning standard/medium)")
            # Same deployment/credentials as GPT Sol, only the reasoning config differs
            client = AzureOpenAI(api_key=gpt_terra_api_key, api_version=gpt_terra_api_version, azure_endpoint=gpt_terra_endpoint)
            deployment_name = gpt_sol_deployment
            active_api_version = gpt_terra_api_version
            vlm_provider = "azure_gpt_sol_standard"

        elif selected_provider == "azure_gpt5":
            logging.info(f"[{episode_id}] Using Azure GPT 5.2 model: {gpt5_deployment}")
            # GPT 5.2 uses OpenAI client with base_url
            client = OpenAI(api_key=azure_api_key, base_url=gpt5_endpoint)
            deployment_name = gpt5_deployment
            vlm_provider = "azure_gpt5"

        elif selected_provider == "gemma-4-26b":
            logging.info(f"[{episode_id}] Using LM Studio model: {lmstudio_model}")
            # LM Studio exposes an OpenAI-compatible server; talk to it with the
            # standard OpenAI client pointed at LMSTUDIO_BASE_URL. The api_key is
            # required by the client but ignored by LM Studio. The model is
            # identified by LMSTUDIO_VLM_MODEL on that server.
            client = OpenAI(base_url=lmstudio_base_url, api_key="lm-studio")
            deployment_name = lmstudio_model
            vlm_provider = "gemma-4-26b"

        elif selected_provider == "gemma12b":
            logging.info(
                f"[{episode_id}] Using local gemma-4-12b GGUF: {gemma12b_model_path} "
                f"(gpu_layers={gemma12b_n_gpu_layers}, n_ctx={gemma12b_n_ctx})"
            )
            # In-process llama.cpp vision model (GGUF + mmproj), cached across episodes.
            client = _get_gemma12b_client(
                gemma12b_model_path, gemma12b_mmproj_path, gemma12b_n_gpu_layers, gemma12b_n_ctx
            )
            deployment_name = os.path.basename(gemma12b_model_path)
            vlm_provider = "gemma12b"

    except Exception as client_err:
        msg = f"Failed to initialize VLM client: {client_err}"
        logging.error(f"[{episode_id}] {msg}", exc_info=True)
        return 0, "error_vlm_client_init", msg

    newly_added_credits_count = 0
    prompt_template = config.BASE_PROMPT_TEMPLATE
    previous_llm_output_json_str = "[]"

    # Provenance metadata attached to every credit extracted in this run.
    # Rides along in the output JSON so it survives the "reload from JSON" path
    # and lands in the credits.metadata DB column at save time.
    run_metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": deployment_name,
        "api_version": active_api_version,
    }

    manifest_path = episode_dir / "analysis" / "analysis_manifest.json"
    manifest_data = {}
    if manifest_path.is_file():
        try:
            with open(manifest_path, 'r', encoding='utf-8') as manifest_file:
                manifest_data = json.load(manifest_file)
            logging.info(f"[{episode_id}] Loaded analysis manifest with {len(manifest_data.get('scenes', {}))} scenes.")
        except Exception as e:
            logging.error(f"[{episode_id}] Error loading manifest file {manifest_path}: {e}", exc_info=True)
    else:
        logging.warning(f"[{episode_id}] Manifest file not found: {manifest_path}.")

    scenes_data = manifest_data.get("scenes", {})
    frames_to_process = []

    for scene_key, scene_result in scenes_data.items():
        output_files = scene_result.get("output_files", [])

        scene_position = scene_result.get("position", "unknown")
        
        # Extract scroll direction from SCENE metadata (not frame metadata)
        scene_scroll_direction = scene_result.get("scroll_direction", None)

        for frame_info_manifest in output_files:
            relative_path_str = frame_info_manifest.get("path")

            frame_num = frame_info_manifest.get("frame_num")

            if not relative_path_str:
                logging.warning(f"[{episode_id}] Frame info missing path: {frame_info_manifest}. Skipping.")
                continue

            full_image_path = frames_dir / Path(relative_path_str).name
            if not full_image_path.is_file():
                logging.warning(f"[{episode_id}] Frame file does not exist: {full_image_path}. Skipping.")
                continue

            frame_filename = full_image_path.name

            if frame_filename in processed_frames_set:
                logging.info(
                    f"[{episode_id}] Frame already processed based on existing output, skipping: {frame_filename}"
                )
                continue
            
            frames_to_process.append(
                {
                    "path": full_image_path,
                    "filename": frame_filename,
                    "frame_num": frame_num,
                    "scene_position": scene_position,
                    "scroll_direction": scene_scroll_direction,  # Use scene-level scroll direction
                }
            )

    logging.info(f"[{episode_id}] Total frames to process from manifest: {len(frames_to_process)}.")

    if not frames_to_process:
        msg = f"No new frames to process for episode {episode_id} after checking manifest and existing output."
        logging.warning(f"[{episode_id}] {msg}")

        if not output_json_path.exists() or not processed_frames_set:
            return 0, "skipped_no_frames_to_process", msg
        else:
            logging.info(
                f"[{episode_id}] No new frames found in manifest to process, and existing output file present. Assuming completed."
            )
            return 0, "completed_no_new_frames", None

    logging.info(f"[{episode_id}] Found {len(frames_to_process)} new frames to process from manifest.")

    try:
        all_parsed_credits = []
        name_to_role_groups = {}

        # Drop any previous raw calls for THIS model so a re-run replaces its own
        # rows (other models' calls for the episode are kept for comparison).
        utils.clear_raw_responses_for_model(episode_id, vlm_provider)

        if output_json_path.exists():
            try:
                with open(output_json_path, 'r', encoding='utf-8') as f_existing:
                    all_parsed_credits.extend(json.load(f_existing))
                logging.info(f"[{episode_id}] Loaded {len(all_parsed_credits)} existing credits to append to.")
            except Exception as e:
                logging.warning(
                    f"[{episode_id}] Could not load existing credits from {output_json_path} for appending: {e}"
                )

        for frame_idx, frame_data in enumerate(tqdm(frames_to_process, desc=f"Processing frames for {episode_id}")):
            frame_path = frame_data["path"]
            data_url = local_image_to_data_url(frame_path)

            if not data_url:
                logging.warning(f"[{episode_id}] Failed to encode image {frame_path} to data URL. Skipping frame.")
                continue
            
            # Get previous frame for visual context (if enabled and available)
            previous_frame_path = None
            previous_frame_data_url = None
            if include_previous_frame and frame_idx > 0:
                previous_frame_path = frames_to_process[frame_idx - 1]["path"]
                previous_frame_data_url = local_image_to_data_url(previous_frame_path)
                if not previous_frame_data_url:
                    logging.warning(f"[{episode_id}] Failed to encode previous frame {previous_frame_path} to data URL. Proceeding without it.")
                    previous_frame_path = None

            try:
                # Add scroll direction note only for DOWNWARD scrolling
                # In downward scroll, role titles appear BELOW the names (reversed order)
                scroll_note = ""
                scroll_direction = frame_data.get("scroll_direction")
                
                if scroll_direction == "downward":
                    scroll_note = "\n\nIMPORTANT: These credits are scrolling DOWNWARD (from top to bottom). The role title appears BELOW the names belonging to that role. For example, if you see names first and then 'Director' below them, those names are the directors.\n"
                # For upward or static credits, no special note needed (normal reading order)
                
                prompt_text = prompt_template.format(previous_credits_json=previous_llm_output_json_str) + scroll_note
                logging.debug(f"[{episode_id}] Using prompt for frame {frame_idx}: {frame_data['filename']} (scroll: {scroll_direction})...")

                # Prepare messages based on provider
                if vlm_provider == "claude":
                    # Claude format with base64 image and prompt caching
                    with open(frame_path, "rb") as image_file:
                        image_data = base64.b64encode(image_file.read()).decode("utf-8")
                    
                    # Build system message with the static prompt template for caching
                    # The prompt template is static across all frames and should be cached
                    system_message = [
                        {
                            "type": "text",
                            "text": prompt_template,
                            "cache_control": {"type": "ephemeral", "ttl": "1h"}  # Cache the static prompt
                        }
                    ]
                    
                    content_parts = []
                    
                    # Add previous frame context if available
                    if previous_frame_path and previous_frame_data_url:
                        with open(previous_frame_path, "rb") as prev_image_file:
                            prev_image_data = base64.b64encode(prev_image_file.read()).decode("utf-8")
                        
                        content_parts.append({
                            "type": "text",
                            "text": "CONTEXT: This is the PREVIOUS frame for reference (you already processed this one):"
                        })
                        content_parts.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": prev_image_data,
                            },
                            "cache_control": {"type": "ephemeral"}  # Cache previous frame
                        })
                        content_parts.append({
                            "type": "text",
                            "text": "---\n\nNow analyze the CURRENT frame below and extract NEW credits:\n"
                        })
                    
                    # Add current frame (not cached - changes every request)
                    content_parts.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data,
                        },
                    })
                    
                    # Add dynamic parts: previous credits JSON and scroll note
                    dynamic_prompt = f"previous_credits_json: {previous_llm_output_json_str}{scroll_note}"
                    content_parts.append({
                        "type": "text",
                        "text": dynamic_prompt
                    })
                    
                    messages = [
                        {
                            "role": "user",
                            "content": content_parts,
                        }
                    ]
                else:
                    # Azure OpenAI / GPT-5 format with data URL
                    # For prompt caching: put STATIC content first (minimum 1024 tokens)
                    # The static prompt template stays at the beginning for automatic caching
                    # GPT Terra/Sol must request the image with detail "original" (full resolution)
                    image_detail = "original" if vlm_provider in RESPONSES_API_PROVIDERS else "high"
                    content_parts = []
                    
                    # STATIC: Add the base prompt template first (gets cached automatically if >1024 tokens)
                    content_parts.append({
                        "type": "text",
                        "text": prompt_template
                    })
                    
                    # STATIC: Add previous frame context if available (also cacheable)
                    if previous_frame_path and previous_frame_data_url:
                        content_parts.append({
                            "type": "text",
                            "text": "\n\nCONTEXT: This is the PREVIOUS frame for reference (you already processed this one):"
                        })
                        content_parts.append({
                            "type": "image_url", 
                            "image_url": {
                                "url": previous_frame_data_url,
                                "detail": image_detail
                            }
                        })
                        content_parts.append({
                            "type": "text",
                            "text": "---\n\nNow analyze the CURRENT frame below and extract NEW credits:\n"
                        })
                    
                    # DYNAMIC: Add current frame (changes every request, not cached)
                    content_parts.append({
                        "type": "image_url", 
                        "image_url": {
                            "url": data_url,
                            "detail": image_detail
                        }
                    })
                    
                    # DYNAMIC: Add variable parts (previous credits JSON and scroll note)
                    dynamic_prompt = f"\n\nprevious_credits_json: {previous_llm_output_json_str}{scroll_note}"
                    content_parts.append({
                        "type": "text",
                        "text": dynamic_prompt
                    })
                    
                    # Note: data_url already contains "data:image/jpeg;base64," prefix from local_image_to_data_url
                    messages = [
                        {
                            "role": "user",
                            "content": content_parts,
                        }
                    ]

                logging.debug(f"[{episode_id}] Sending request for frame {frame_idx}: {frame_data['filename']}")

                # Retry with exponential backoff
                response = None
                for attempt in range(1, MAX_API_RETRIES + 1):
                    try:
                        if vlm_provider == "claude":
                            # Claude API call with prompt caching support
                            response = client.messages.create(
                                model=deployment_name,
                                system=system_message,  # System message with cached prompt template
                                messages=messages,
                                max_tokens=max_new_tokens
                            )
                            
                            # Log cache usage metrics if available
                            if hasattr(response, 'usage'):
                                usage = response.usage
                                cache_creation_tokens = getattr(usage, 'cache_creation_input_tokens', 0)
                                cache_read_tokens = getattr(usage, 'cache_read_input_tokens', 0)
                                if cache_creation_tokens > 0 or cache_read_tokens > 0:
                                    logging.info(
                                        f"[{episode_id}] Frame {frame_idx} cache metrics: "
                                        f"created={cache_creation_tokens}, read={cache_read_tokens}"
                                    )
                            
                            # Extract content from Claude response (it's a list of TextBlock)
                            if response.content and len(response.content) > 0:
                                generated_text = response.content[0].text.strip()
                            else:
                                generated_text = ""
                        elif vlm_provider in RESPONSES_API_PROVIDERS:
                            # GPT Terra/Sol (gpt-5.6 family) use the Responses API; the
                            # reasoning config (pro/high vs standard/medium) is per-provider.
                            responses_input = [{
                                "role": "user",
                                "content": [_chat_part_to_responses_part(pt) for pt in content_parts],
                            }]
                            response = client.responses.create(
                                model=deployment_name,
                                reasoning=VLM_REASONING_BY_PROVIDER[vlm_provider],
                                input=responses_input,
                            )
                            generated_text = (getattr(response, "output_text", None) or "").strip()

                            # Log usage/cache metrics if available (Responses API)
                            if hasattr(response, "usage") and response.usage is not None:
                                _det = getattr(response.usage, "input_tokens_details", None)
                                _cached = getattr(_det, "cached_tokens", 0) if _det else 0
                                _odet = getattr(response.usage, "output_tokens_details", None)
                                _reason = getattr(_odet, "reasoning_tokens", 0) if _odet else 0
                                if _cached or _reason:
                                    logging.info(
                                        f"[{episode_id}] Frame {frame_idx} {vlm_provider} usage: "
                                        f"cached={_cached}, reasoning={_reason}"
                                    )
                        elif vlm_provider == "gemma12b":
                            # Local gemma-4-12b vision GGUF via llama-cpp-python (in-process,
                            # GPU). Reuses the OpenAI-style `messages` (data-URL images) built
                            # above; the response is a plain dict.
                            response = client.create_chat_completion(
                                messages=messages,
                                max_tokens=max_new_tokens,
                                temperature=0.0,
                            )
                            generated_text = (response["choices"][0]["message"].get("content") or "").strip()
                        else:
                            # Azure OpenAI / GPT-5 / LM Studio (OpenAI-compatible) API call
                            # Automatic prompt caching: static content at the beginning gets cached

                            completion_kwargs = {
                                "model": deployment_name,
                                "messages": messages
                            }

                            # GPT 5.2 does not support max_tokens or temperature.
                            # LM Studio accepts both.
                            if vlm_provider != "azure_gpt5":
                                completion_kwargs["max_tokens"] = max_new_tokens
                                completion_kwargs["temperature"] = 0.0

                            response = client.chat.completions.create(**completion_kwargs)
                            generated_text = response.choices[0].message.content.strip()
                            
                            # Log cache usage metrics if available (Azure OpenAI Prompt Caching)
                            if hasattr(response, 'usage'):
                                usage = response.usage
                                cached_tokens = 0
                                if hasattr(usage, 'prompt_tokens_details'):
                                    details = usage.prompt_tokens_details
                                    if hasattr(details, 'cached_tokens'):
                                        cached_tokens = details.cached_tokens
                                if cached_tokens > 0:
                                    logging.info(
                                        f"[{episode_id}] Frame {frame_idx} Azure cache: "
                                        f"cached={cached_tokens} tokens (50% discount applied)"
                                    )
                        break
                    except (APIConnectionError, APIStatusError) as api_err:
                        logging.warning(
                            f"[{episode_id}] VLM API error (attempt {attempt}/{MAX_API_RETRIES}): {api_err}"
                        )
                        if attempt == MAX_API_RETRIES:
                            raise
                        sleep_time = BACKOFF_FACTOR**attempt
                        time.sleep(sleep_time)
                    except RateLimitError as rl_err:
                        logging.warning(
                            f"[{episode_id}] VLM rate limit (attempt {attempt}/{MAX_API_RETRIES}): {rl_err}"
                        )
                        if attempt == MAX_API_RETRIES:
                            raise
                        sleep_time = BACKOFF_FACTOR**attempt
                        time.sleep(sleep_time)
                    except Exception as e:
                        # Catch Claude-specific errors
                        logging.warning(
                            f"[{episode_id}] VLM error (attempt {attempt}/{MAX_API_RETRIES}): {e}"
                        )
                        if attempt == MAX_API_RETRIES:
                            raise
                        sleep_time = BACKOFF_FACTOR**attempt
                        time.sleep(sleep_time)
                        
                if response is None:
                    raise RuntimeError("VLM retry failed, no response received.")

                # Full raw LLM response for this frame (JSON-serializable). Saved as its
                # own row NOW, before parsing, so the call is persisted even when the
                # frame is empty and yields no names.
                try:
                    frame_raw_response = _response_to_jsonable(response)
                except Exception as raw_err:
                    logging.warning(f"[{episode_id}] Could not serialize raw response for frame {frame_idx}: {raw_err}")
                    frame_raw_response = {"repr": str(response)}
                utils.save_raw_response_llm_call(
                    episode_id, vlm_provider, frame_data["filename"], frame_raw_response
                )

                logging.debug(
                    f"[{episode_id}] VLM Raw Output for frame {frame_idx} ({frame_data['filename']}): {generated_text}"
                )

                cleaned_json_str = utils.clean_vlm_output(generated_text)

                current_frame_parsed_credits = utils.parse_vlm_json(
                    cleaned_json_str, frame_data['filename'], name_key="name"
                )

                # Note: Role group corrections are applied AFTER saving to DB
                # via apply_role_group_corrections_to_database() in utils.py

                if current_frame_parsed_credits:
                    for credit_entry in current_frame_parsed_credits:
                        raw_role_value = credit_entry.get("role_group")
                        # Role groups are now validated directly in config.py
                        # No need for external mapping file
                        credit_entry["role_group_normalized"] = raw_role_value if raw_role_value is not None else "Unknown"

                        credit_entry['source_frame'] = [frame_data["filename"]]
                        credit_entry['original_frame_number'] = [frame_data["frame_num"]]

                        credit_entry['scene_position'] = frame_data.get("scene_position", "unknown")

                        # Provenance: which model/api_version/time produced this credit
                        credit_entry['metadata'] = dict(run_metadata)

                        credit_entry.pop("source_image_index", None)

                        logging.debug(
                            f"[{episode_id}] Parsed credit entry for {frame_data['filename']}: {credit_entry}"
                        )
                        all_parsed_credits.append(credit_entry)

                previous_llm_output_json_str = cleaned_json_str
                logging.debug(
                    f"[{episode_id}] Context for next frame (from {frame_data['filename']}): {previous_llm_output_json_str[:200]}..."
                )

            except APIStatusError as e:
                logging.error(
                    f"[{episode_id}] Azure API error on frame {frame_idx} ({frame_data['filename']}): {e.status_code} - {e.response}",
                    exc_info=True,
                )
            except APIConnectionError as e:
                logging.error(
                    f"[{episode_id}] Azure connection error on frame {frame_idx} ({frame_data['filename']}): {e}",
                    exc_info=True,
                )

            except RateLimitError as e:
                logging.error(
                    f"[{episode_id}] Azure rate limit exceeded on frame {frame_idx} ({frame_data['filename']}): {e}. Waiting and retrying might be needed.",
                    exc_info=True,
                )

            except Exception as frame_err:
                logging.error(
                    f"[{episode_id}] Error processing frame {frame_idx} ({frame_data['filename']}): {frame_err}",
                    exc_info=True,
                )

        dedup_credits_map = {}

        for credit in all_parsed_credits:
            name = (credit.get("name") or "").strip()
            role_group = credit.get("role_group_normalized") or credit.get("role_group")
            if name and role_group:
                name_to_role_groups.setdefault(name, set()).add(role_group)

        for credit in all_parsed_credits:
            role_group = credit.get("role_group_normalized") or credit.get("role_group")
            name = (credit.get("name") or "").strip()
            key = (role_group, name)

            current_source_frames = credit.get("source_frame", [])
            if not isinstance(current_source_frames, list):
                current_source_frames = [current_source_frames] if current_source_frames is not None else []

            current_frame_numbers = credit.get("original_frame_number", [])
            if not isinstance(current_frame_numbers, list):
                current_frame_numbers = [current_frame_numbers] if current_frame_numbers is not None else []

            if key in dedup_credits_map:
                existing_entry = dedup_credits_map[key]
                for f in current_source_frames:
                    if f not in existing_entry["source_frame"]:
                        existing_entry["source_frame"].append(f)
                for n in current_frame_numbers:
                    if n not in existing_entry["original_frame_number"]:
                        existing_entry["original_frame_number"].append(n)

            else:
                new_entry = credit.copy()
                new_entry["source_frame"] = current_source_frames
                new_entry["original_frame_number"] = current_frame_numbers

                new_entry.pop("source_image_index_issue", None)
                dedup_credits_map[key] = new_entry

        final_dedup_credits_list = list(dedup_credits_map.values())

        for credit_entry in final_dedup_credits_list:
            name = (credit_entry.get("name") or "").strip()
            if name and len(name_to_role_groups.get(name, [])) > 1:
                credit_entry["Need revisioning for deduplication"] = True

        try:
            with open(output_json_path, 'w', encoding='utf-8') as f_out:
                json.dump(final_dedup_credits_list, f_out, ensure_ascii=False, indent=2)
            newly_added_credits_count = len(final_dedup_credits_list)
            logging.info(
                f"[{episode_id}] Saved/Updated {output_json_path.name} with {newly_added_credits_count} total deduplicated credit entries."
            )
        except Exception as write_err:
            logging.error(f"[{episode_id}] Failed to write dedup credits JSON: {write_err}", exc_info=True)
            return 0, "error_writing_json", str(write_err)

    except Exception as e:
        msg = f"Unexpected error during Azure VLM processing loop: {e}"
        logging.error(f"[{episode_id}] {msg}", exc_info=True)
        return 0, "error_vlm_loop", msg

    logging.info(f"[{episode_id}] Finished Azure VLM processing. Total credits in file: {newly_added_credits_count}.")
    return newly_added_credits_count, "completed", None
