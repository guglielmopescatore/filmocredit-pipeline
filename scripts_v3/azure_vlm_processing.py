import base64
import json
import logging
import os
import time
from dataclasses import dataclass
from mimetypes import guess_type
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

try:
    from openai import APIConnectionError, APIStatusError, AzureOpenAI, RateLimitError
except ImportError:
    AzureOpenAI = None
    APIStatusError = APIConnectionError = RateLimitError = None
    logging.warning("Azure OpenAI library not found. Azure VLM processing will not be available.")

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
    episode_id: str, max_new_tokens: int
) -> Tuple[int, str, Optional[str]]:
    """
    Runs Azure VLM OCR on selected frames for an episode, one frame at a time,
    processes results, and saves them.

    Args:
        episode_id: The ID of the episode.
        max_new_tokens: The maximum number of new tokens for generation.

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

    processed_frames_set = set()
    if output_json_path.exists():
        try:
            with open(output_json_path, 'r', encoding='utf-8') as f:
                existing_credits = json.load(f)
                for credit in existing_credits:
                    frames = credit.get('source_frame', [])
                    if isinstance(frames, list):
                        processed_frames_set.update(frames)
                    elif isinstance(frames, str):
                        processed_frames_set.add(frames)
        except Exception as e:
            logging.warning(f"[{episode_id}] Could not load processed frames from {output_json_path}: {e}")
    logging.info(
        f"[{episode_id}] Found {len(processed_frames_set)} already processed unique frame filenames in {output_json_path.name}."
    )

    # Initialize Azure client with credentials from config.Env
    try:
        api_key = os.getenv(config.AzureConfig.API_KEY_ENV)
        api_version = os.getenv(config.AzureConfig.API_VERSION_ENV, "2024-02-15-preview")
        endpoint = os.getenv(config.AzureConfig.ENDPOINT_ENV)
        deployment_name = os.getenv(config.AzureConfig.DEPLOYMENT_NAME_ENV)
        if not all([api_key, endpoint, deployment_name]):
            raise ValueError("Azure endpoint/deployment/key environment variables not set.")
        client = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)
    except Exception as client_err:
        msg = f"Failed to initialize Azure OpenAI client: {client_err}"
        logging.error(f"[{episode_id}] {msg}", exc_info=True)
        return 0, "error_azure_client_init", msg

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

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]

                logging.debug(f"[{episode_id}] Sending request for frame {frame_idx}: {frame_data['filename']}")

                # Retry with exponential backoff
                response = None
                for attempt in range(1, MAX_API_RETRIES + 1):
                    try:
                        response = client.chat.completions.create(
                            model=deployment_name, messages=messages, max_tokens=max_new_tokens, temperature=0.0
                        )
                        break
                    except (APIConnectionError, APIStatusError) as api_err:
                        logging.warning(
                            f"[{episode_id}] Azure API error (attempt {attempt}/{MAX_API_RETRIES}): {api_err}"
                        )
                        if attempt == MAX_API_RETRIES:
                            raise
                        sleep_time = BACKOFF_FACTOR**attempt
                        time.sleep(sleep_time)
                    except RateLimitError as rl_err:
                        logging.warning(
                            f"[{episode_id}] Azure rate limit (attempt {attempt}/{MAX_API_RETRIES}): {rl_err}"
                        )
                        if attempt == MAX_API_RETRIES:
                            raise
                        sleep_time = BACKOFF_FACTOR**attempt
                        time.sleep(sleep_time)
                if response is None:
                    raise RuntimeError("Azure VLM retry failed, no response received.")

                generated_text = response.choices[0].message.content.strip()
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
