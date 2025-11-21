import base64
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
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

from scripts_v3 import config, utils

# Exponential backoff settings for Azure API retries
MAX_API_RETRIES = 3
BACKOFF_FACTOR = 2.0


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


def run_azure_vlm_ocr_on_frames(
    episode_id: str, max_new_tokens: int, vlm_provider: str = "auto"
) -> Tuple[int, str, Optional[str]]:
    """
    Runs VLM OCR on selected frames for an episode, one frame at a time,
    processes results, and saves them.

    Args:
        episode_id: The ID of the episode.
        max_new_tokens: The maximum number of new tokens for generation.
        vlm_provider: VLM provider to use ('auto', 'claude', 'azure'). Default: 'auto'

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
    ocr_dir = episode_dir / 'ocr'
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
        
        # GPT 4.1
        gpt4_endpoint = os.getenv(config.AzureConfig.GPT4_ENDPOINT_ENV) or os.getenv(config.AzureConfig.ENDPOINT_ENV)
        gpt4_deployment = os.getenv(config.AzureConfig.GPT4_DEPLOYMENT_NAME_ENV) or os.getenv(config.AzureConfig.DEPLOYMENT_NAME_ENV)
        
        # GPT 5.1
        gpt5_endpoint = os.getenv(config.AzureConfig.GPT5_ENDPOINT_ENV)
        gpt5_deployment = os.getenv(config.AzureConfig.GPT5_DEPLOYMENT_NAME_ENV)
        
        # Check availability
        claude_available = bool(claude_endpoint and claude_api_key and claude_model and AnthropicFoundry)
        gpt4_available = bool(azure_api_key and gpt4_endpoint and gpt4_deployment and AzureOpenAI)
        gpt5_available = bool(azure_api_key and gpt5_endpoint and gpt5_deployment and OpenAI)
        
        # Debug logging for availability
        logging.debug(f"[{episode_id}] VLM Availability Check:")
        logging.debug(f"  Claude: {claude_available} (Key: {bool(claude_api_key)}, Endpoint: {bool(claude_endpoint)}, Model: {bool(claude_model)}, Lib: {bool(AnthropicFoundry)})")
        logging.debug(f"  GPT-4.1: {gpt4_available} (Key: {bool(azure_api_key)}, Endpoint: {bool(gpt4_endpoint)}, Model: {bool(gpt4_deployment)}, Lib: {bool(AzureOpenAI)})")
        logging.debug(f"  GPT-5.1: {gpt5_available} (Key: {bool(azure_api_key)}, Endpoint: {bool(gpt5_endpoint)}, Model: {bool(gpt5_deployment)}, Lib: {bool(OpenAI)})")
        
        # Select provider based on preference
        selected_provider = None
        
        if vlm_provider == "claude":
            if not claude_available:
                raise ValueError("Claude provider requested but credentials not available or library not installed")
            selected_provider = "claude"
        elif vlm_provider == "azure_gpt5":
            if not gpt5_available:
                missing = []
                if not azure_api_key: missing.append("AZURE_OPENAI_KEY")
                if not gpt5_endpoint: missing.append("GPT_5_1_AZURE_OPENAI_ENDPOINT")
                if not gpt5_deployment: missing.append("GPT_5_1_AZURE_OPENAI_DEPLOYMENT_NAME")
                if not OpenAI: missing.append("openai library (OpenAI class)")
                raise ValueError(f"Azure GPT-5.1 provider requested but credentials not available or library not installed. Missing: {', '.join(missing)}")
            selected_provider = "azure_gpt5"
        elif vlm_provider == "azure_gpt4":
            if not gpt4_available:
                raise ValueError("Azure GPT-4.1 provider requested but credentials not available or library not installed")
            selected_provider = "azure_gpt4"
        elif vlm_provider == "azure": # Legacy/Generic Azure request -> prefer GPT-5 if available, else GPT-4
             if gpt5_available:
                 selected_provider = "azure_gpt5"
             elif gpt4_available:
                 selected_provider = "azure_gpt4"
             else:
                 raise ValueError("Azure provider requested but neither GPT-5.1 nor GPT-4.1 credentials available")
        else:  # auto
            if claude_available:
                selected_provider = "claude"
            elif gpt5_available:
                selected_provider = "azure_gpt5"
            elif gpt4_available:
                selected_provider = "azure_gpt4"
            else:
                raise ValueError("No VLM provider available (neither Claude nor Azure credentials set)")
        
        # Initialize client
        if selected_provider == "claude":
            logging.info(f"[{episode_id}] Using Claude model: {claude_model}")
            client = AnthropicFoundry(api_key=claude_api_key, base_url=claude_endpoint)
            deployment_name = claude_model
            vlm_provider = "claude"
            
        elif selected_provider == "azure_gpt5":
            logging.info(f"[{episode_id}] Using Azure GPT-5.1 model: {gpt5_deployment}")
            # GPT-5.1 uses OpenAI client with base_url
            client = OpenAI(api_key=azure_api_key, base_url=gpt5_endpoint)
            deployment_name = gpt5_deployment
            vlm_provider = "azure_gpt5"
            
        elif selected_provider == "azure_gpt4":
            logging.info(f"[{episode_id}] Using Azure GPT-4.1 model: {gpt4_deployment}")
            # GPT-4.1 uses AzureOpenAI client
            client = AzureOpenAI(api_key=azure_api_key, api_version=azure_api_version, azure_endpoint=gpt4_endpoint)
            deployment_name = gpt4_deployment
            vlm_provider = "azure_gpt4"
            
    except Exception as client_err:
        msg = f"Failed to initialize VLM client: {client_err}"
        logging.error(f"[{episode_id}] {msg}", exc_info=True)
        return 0, "error_vlm_client_init", msg

    newly_added_credits_count = 0
    prompt_template = config.BASE_PROMPT_TEMPLATE
    previous_llm_output_json_str = "[]"

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

            try:
                prompt_text = prompt_template.format(previous_credits_json=previous_llm_output_json_str)
                logging.debug(f"[{episode_id}] Using prompt for frame {frame_idx}: {frame_data['filename']}...")

                # Prepare messages based on provider
                if vlm_provider == "claude":
                    # Claude format with base64 image
                    with open(frame_path, "rb") as image_file:
                        image_data = base64.b64encode(image_file.read()).decode("utf-8")
                    
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": image_data,
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": prompt_text
                                }
                            ],
                        }
                    ]
                else:
                    # Azure OpenAI / GPT-5 format with data URL
                    # Note: data_url already contains "data:image/jpeg;base64," prefix from local_image_to_data_url
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url", 
                                    "image_url": {
                                        "url": data_url,
                                        "detail": "high"
                                    }
                                },
                                {"type": "text", "text": prompt_text},
                            ],
                        }
                    ]

                logging.debug(f"[{episode_id}] Sending request for frame {frame_idx}: {frame_data['filename']}")

                # Retry with exponential backoff
                response = None
                for attempt in range(1, MAX_API_RETRIES + 1):
                    try:
                        if vlm_provider == "claude":
                            # Claude API call
                            response = client.messages.create(
                                model=deployment_name,
                                messages=messages,
                                max_tokens=max_new_tokens
                            )
                            # Extract content from Claude response (it's a list of TextBlock)
                            if response.content and len(response.content) > 0:
                                generated_text = response.content[0].text.strip()
                            else:
                                generated_text = ""
                        else:
                            # Azure OpenAI / GPT-5 API call
                            # Both use client.chat.completions.create with same signature
                            
                            completion_kwargs = {
                                "model": deployment_name,
                                "messages": messages
                            }
                            
                            # GPT-5.1 and GPT-4.1 (Preview) do not support max_tokens or temperature
                            if vlm_provider not in ["azure_gpt5", "azure_gpt4"]:
                                completion_kwargs["max_tokens"] = max_new_tokens
                                completion_kwargs["temperature"] = 0.0
                                
                            response = client.chat.completions.create(**completion_kwargs)
                            generated_text = response.choices[0].message.content.strip()
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

                logging.debug(
                    f"[{episode_id}] VLM Raw Output for frame {frame_idx} ({frame_data['filename']}): {generated_text}"
                )

                cleaned_json_str = utils.clean_vlm_output(generated_text)

                current_frame_parsed_credits = utils.parse_vlm_json(
                    cleaned_json_str, frame_data['filename'], name_key="name"
                )

                if current_frame_parsed_credits:
                    for credit_entry in current_frame_parsed_credits:
                        raw_role_value = credit_entry.get("role_group")
                        # Role groups are now validated directly in config.py
                        # No need for external mapping file
                        credit_entry["role_group_normalized"] = raw_role_value if raw_role_value is not None else "Unknown"

                        credit_entry['source_frame'] = [frame_data["filename"]]
                        credit_entry['original_frame_number'] = [frame_data["frame_num"]]

                        credit_entry['scene_position'] = frame_data.get("scene_position", "unknown")

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
