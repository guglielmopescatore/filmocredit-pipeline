import logging

logger = logging.getLogger(__name__)
import json
import logging
from datetime import datetime
from logging import LoggerAdapter
from pathlib import Path

import cv2
import numpy as np
from scenedetect import FrameTimecode, SceneManager, open_video

logger = logging.getLogger(__name__)
import json
import time
from enum import Enum
from typing import Any, List, Optional, Set, Tuple

from PIL import Image
from scenedetect import FrameTimecode
from scenedetect.detectors import ContentDetector, ThresholdDetector
from tqdm import tqdm

from . import config, utils, constants


class SceneDetectionStatus(Enum):
    PENDING = "pending"
    COMPLETED_NO_SCENES_IN_SEGMENTS = "completed_no_scenes_in_segments"
    COMPLETED_NO_VALID_LONG_SCENES = "completed_no_valid_long_scenes"
    COMPLETED_NO_TEXT_IN_VALID_SCENES = "completed_no_text_in_valid_scenes"
    COMPLETED_FOUND_CANDIDATE_SCENES = "completed_found_candidate_scenes"
    ERROR = "error"


MAX_OCR_ATTEMPTS = 3


def _filter_scenes_by_count(
    scenes: List[Tuple[FrameTimecode, FrameTimecode, int]], start_count: int, end_count: int
) -> List[Tuple[FrameTimecode, FrameTimecode, int]]:
    """
    Filter scenes to keep only the first start_count and last end_count scenes.
    Returns tuples of (start_tc, end_tc, original_scene_index).
    """
    if not scenes:
        return []

    total_scenes = len(scenes)
    filtered_scenes = []

    # Take first start_count scenes (keeping original indices)
    for i in range(min(start_count, total_scenes)):
        start_tc, end_tc, original_idx = scenes[i]
        filtered_scenes.append((start_tc, end_tc, original_idx))

    # Take last end_count scenes (if we have enough scenes and they don't overlap)
    if total_scenes > start_count:
        last_scenes_start_idx = max(start_count, total_scenes - end_count)
        for i in range(last_scenes_start_idx, total_scenes):
            start_tc, end_tc, original_idx = scenes[i]
            filtered_scenes.append((start_tc, end_tc, original_idx))

    return filtered_scenes


def _filter_scenes_by_time(
    scenes: List[Tuple[FrameTimecode, FrameTimecode, int]], start_minutes: float, end_minutes: float, video
) -> List[Tuple[FrameTimecode, FrameTimecode, int]]:
    """
    Filter scenes to keep only those within the first start_minutes and last end_minutes of the video.
    Returns tuples of (start_tc, end_tc, original_scene_index).
    """
    if not scenes:
        return []

    # Get video duration
    video_duration_seconds = video.duration.get_seconds()
    start_seconds = start_minutes * 60
    end_seconds = end_minutes * 60

    # Calculate time boundaries
    start_boundary_seconds = min(start_seconds, video_duration_seconds)
    end_boundary_seconds = max(0, video_duration_seconds - end_seconds)

    filtered_scenes = []

    for start_tc, end_tc, original_idx in scenes:
        scene_start_seconds = start_tc.get_seconds()
        scene_end_seconds = end_tc.get_seconds()

        # Include scene if it overlaps with start segment or end segment
        in_start_segment = scene_start_seconds < start_boundary_seconds
        in_end_segment = scene_end_seconds > end_boundary_seconds

        if in_start_segment or in_end_segment:
            filtered_scenes.append((start_tc, end_tc, original_idx))

    return filtered_scenes


def identify_candidate_scenes(
    video_path: Path,
    episode_id: str,
    ocr_reader: Any,
    ocr_engine_type: str,
    user_stopwords: List[str],
    scene_counts: Optional[Tuple[int, int]] = None,
    time_segments: Optional[Tuple[float, float]] = None,
    whole_episode: bool = False,
) -> Tuple[List[dict], SceneDetectionStatus, Optional[str]]:
    """
    Identifies candidate scenes in a video based on scene counts (first N, last N), time segments (first X minutes, last Y minutes), or the whole episode.

    Args:
        video_path: Path to the video file
        episode_id: Unique identifier for the episode
        ocr_reader: OCR reader instance
        ocr_engine_type: Type of OCR engine being used
        user_stopwords: List of words to filter out from OCR results
        scene_counts: Optional tuple of (start_scenes_count, end_scenes_count). If None, defaults are used.
        time_segments: Optional tuple of (start_minutes, end_minutes). Takes precedence over scene_counts if provided.
        whole_episode: If True, analyzes the entire episode without any filtering. Takes precedence over other parameters.
    """
    episode_dir = config.EPISODES_BASE_DIR / episode_id
    analysis_dir = episode_dir / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    output_file = analysis_dir / 'initial_scene_analysis.json'
    scenes_cache_file = analysis_dir / 'raw_scenes_cache.json'

    step1_frames_dir = analysis_dir / 'step1_representative_frames'
    step1_frames_dir.mkdir(parents=True, exist_ok=True)

    candidate_scenes_data: List[dict] = []
    all_detected_shots_data: List[dict] = []
    status: SceneDetectionStatus = SceneDetectionStatus.PENDING
    error_message: Optional[str] = None

    # Create a logger adapter that injects episode_id into log records
    log: LoggerAdapter = LoggerAdapter(logger, {"episode_id": episode_id})
    try:
        video = open_video(str(video_path))
        total_frames = video.duration.get_frames()
        fps = video.frame_rate
        midpoint_frame = total_frames // 2
        log.info(f"Video total frames: {total_frames}, midpoint frame: {midpoint_frame}")

        # Check if cached scenes exist
        scenes_in_video = None
        if scenes_cache_file.exists():
            try:
                with open(scenes_cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    if cached_data.get('total_frames') == total_frames and cached_data.get('fps') == fps:
                        # Convert cached data back to FrameTimecode objects
                        scenes_in_video = []
                        for scene_data in cached_data['scenes']:
                            start_tc = FrameTimecode(scene_data['start_frame'], fps)
                            end_tc = FrameTimecode(scene_data['end_frame'], fps)
                            scenes_in_video.append((start_tc, end_tc))
                        log.info(f"Loaded {len(scenes_in_video)} scenes from cache.")
                    else:
                        log.info("Cached scenes are outdated (different video specs), will regenerate.")
            except Exception as e:
                log.warning(f"Failed to load cached scenes: {e}, will regenerate.")        # If no valid cached scenes, detect them
        if scenes_in_video is None:
            log.info("Detecting scenes using PySceneDetect...")
            scene_manager = SceneManager()
            
            # Use more conservative thresholds for credits detection
            # Higher threshold = less sensitive = fewer cuts in scrolling credits
            content_threshold = config.SceneDetectionConfig.CONTENT_DETECTOR_THRESHOLD
            threshold_threshold = config.SceneDetectionConfig.THRESHOLD_DETECTOR_THRESHOLD
            
            log.info(f"Using ContentDetector threshold: {content_threshold}")
            log.info(f"Using ThresholdDetector threshold: {threshold_threshold}")
            
            scene_manager.add_detector(ContentDetector(threshold=content_threshold))
            scene_manager.add_detector(ThresholdDetector(threshold=threshold_threshold, fade_bias=1))
            video.seek(FrameTimecode(0, fps))
            scene_manager.detect_scenes(video=video, end_time=FrameTimecode(total_frames, fps), show_progress=True)
            scenes_in_video = scene_manager.get_scene_list()

            # Cache the detected scenes
            cache_data = {
                'total_frames': total_frames,
                'fps': fps,
                'timestamp': datetime.now().isoformat(),
                'scenes': [
                    {
                        'start_frame': start_tc.get_frames(),
                        'end_frame': end_tc.get_frames(),
                        'start_timecode': start_tc.get_timecode(),
                        'end_timecode': end_tc.get_timecode(),
                    }
                    for start_tc, end_tc in scenes_in_video
                ],
            }
            with open(scenes_cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            log.info(f"Detected and cached {len(scenes_in_video)} scenes.")        # Create tuples with original scene indices
        scenes_with_indices = [(start_tc, end_tc, idx) for idx, (start_tc, end_tc) in enumerate(scenes_in_video)]

        if whole_episode:
            # Whole episode analysis - use all scenes without filtering
            filtered_scenes = scenes_with_indices
            log.info(f"Analyzing whole episode with {len(filtered_scenes)} scenes (no filtering applied).")
        elif time_segments:
            # Time-based filtering (takes precedence over scene_counts)
            start_minutes, end_minutes = time_segments
            filtered_scenes = _filter_scenes_by_time(scenes_with_indices, start_minutes, end_minutes, video)
            log.info(f"Filtered to {len(filtered_scenes)} scenes (first {start_minutes} min + last {end_minutes} min).")
        elif scene_counts:
            # Scene count-based filtering
            start_count, end_count = scene_counts
            filtered_scenes = _filter_scenes_by_count(scenes_with_indices, start_count, end_count)
            log.info(f"Filtered to {len(filtered_scenes)} scenes (first {start_count} + last {end_count}).")
        else:
            # No filtering - use all scenes
            filtered_scenes = scenes_with_indices

        # Populate all_detected_shots_data
        if filtered_scenes:
            for start_tc, end_tc, original_scene_idx in filtered_scenes:
                temporal_position_tag = "first_half" if start_tc.get_frames() < midpoint_frame else "second_half"
                all_detected_shots_data.append(
                    {
                        "shot_index_in_segment_scan": original_scene_idx,  # Use original scene index
                        "start_frame": start_tc.get_frames(),
                        "end_frame": end_tc.get_frames(),
                        "start_timecode": start_tc.get_timecode(),
                        "end_timecode": end_tc.get_timecode(),
                        "duration_frames": end_tc.get_frames() - start_tc.get_frames(),
                        "position": temporal_position_tag,
                        "segment_origin_tag": "full_video",
                    }
                )
            log.info(f"Storing {len(all_detected_shots_data)} unique raw shots from full video.")
        else:
            log.info("No scenes detected in the video.")
            status = SceneDetectionStatus.COMPLETED_NO_SCENES_IN_SEGMENTS
            # No early return or save here; let it flow to the end of the try block.
            # candidate_scenes_data will remain empty.

        if status != SceneDetectionStatus.COMPLETED_NO_SCENES_IN_SEGMENTS:  # Only proceed if scenes were found
            # Filter by minimum length - now using all_detected_shots_data
            valid_scenes_for_ocr_with_tags = []
            if all_detected_shots_data:  # Check if there are any shots to process
                for shot_data in all_detected_shots_data:
                    if shot_data["duration_frames"] >= constants.SCENE_MIN_LENGTH_FRAMES:
                        start_tc = FrameTimecode(shot_data["start_timecode"], fps)
                        end_tc = FrameTimecode(shot_data["end_timecode"], fps)
                        # shot_data["position_tag"] is the temporal one ("first_half"/"second_half")
                        # shot_data["segment_origin_tag"] is the segment identifier
                        valid_scenes_for_ocr_with_tags.append(
                            (
                                start_tc,
                                end_tc,
                                shot_data["position"],  # was shot_data["position_tag"]
                                shot_data["segment_origin_tag"],
                            )
                        )

            log.info(
                f"Found {len(valid_scenes_for_ocr_with_tags)} scenes after min length filter ({constants.SCENE_MIN_LENGTH_FRAMES} frames). Processing these for OCR."
            )

            if not valid_scenes_for_ocr_with_tags:
                status = SceneDetectionStatus.COMPLETED_NO_VALID_LONG_SCENES
                # Save state and early return if no valid long scenes
                out = {
                    "episode_id": episode_id,
                    "status": status.value,
                    "error": None,
                    "candidate_scenes": [],
                    "raw_shots_detected": all_detected_shots_data,
                    "timestamp": datetime.now().isoformat(),
                }
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(out, f, indent=4, ensure_ascii=False)
                return [], status, None
            else:
                # OCR Sampling and Candidate Scene Identification
                # Loop signature changes to unpack four items
                for scene_idx, (start_time, end_time, temporal_tag_for_scene, origin_tag_for_scene) in enumerate(
                    tqdm(valid_scenes_for_ocr_with_tags, desc=f"[{episode_id}] OCR Sampling & Analysis", unit="scene")
                ):
                    current_scene_data = {
                        "scene_index": scene_idx,  # This is now an index for *valid* scenes for OCR
                        "original_start_frame": start_time.get_frames(),
                        "original_end_frame": end_time.get_frames(),
                        "original_start_timecode": start_time.get_timecode(),
                        "original_end_timecode": end_time.get_timecode(),
                        "duration_frames": end_time.get_frames() - start_time.get_frames(),
                        "position": temporal_tag_for_scene,  # was "position_tag"
                        "segment_origin_tag": origin_tag_for_scene,
                        "has_potential_text": False,
                        "text_found_in_samples": [],
                        "ocr_attempts": [],
                        "ocr_results_for_samples": [],
                        "representative_frames_saved": [],
                        "ocr_errors_in_scene": [],
                    }
                    # --- Start of OCR sampling logic for the current scene ---
                    num_sample_points = len(constants.INITIAL_FRAME_SAMPLE_POINTS)
                    scene_duration_frames = current_scene_data["duration_frames"]

                    for sample_idx, sample_point_ratio in enumerate(constants.INITIAL_FRAME_SAMPLE_POINTS):
                        target_frame_num = start_time.get_frames() + int(scene_duration_frames * sample_point_ratio)
                        if not (start_time.get_frames() <= target_frame_num < end_time.get_frames()):
                            log.debug(f"Sample frame {target_frame_num} out of bounds for scene {scene_idx}. Skipping.")
                            continue

                        frame_timecode = FrameTimecode(target_frame_num, fps)
                        video.seek(frame_timecode)
                        frame_img_np = video.read()

                        if frame_img_np is not None:
                            frame_img_rgb = cv2.cvtColor(frame_img_np, cv2.COLOR_BGR2RGB)
                            img_pil = Image.fromarray(frame_img_rgb)

                            # Convert PIL Image to NumPy BGR array for OCR
                            img_np_bgr_for_ocr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                            text, details, _bbox, error_ocr = None, None, None, None  # Initialize
                            ocr_attempt_count = 0
                            ocr_success = False

                            image_context_id = (
                                f"{episode_id}_scene{scene_idx}_sample{sample_idx}_frame{target_frame_num}"
                            )

                            while ocr_attempt_count < MAX_OCR_ATTEMPTS and not ocr_success:
                                try:
                                    # Pass the NumPy BGR array to run_ocr
                                    text, details, _bbox, error_ocr = utils.run_ocr(
                                        img_np_bgr_for_ocr,
                                        ocr_reader,
                                        ocr_engine_type,
                                        image_context_identifier=image_context_id,
                                    )
                                    if error_ocr:
                                        logging.warning(
                                            f"[{episode_id}] OCR error for scene {scene_idx}, frame {target_frame_num} (attempt {ocr_attempt_count + 1}): {error_ocr}"
                                        )
                                        current_scene_data["ocr_errors_in_scene"].append(
                                            f"Frame {target_frame_num}: {error_ocr}"
                                        )
                                    ocr_success = True  # Assume success if no exception, error string will be handled
                                except Exception as ocr_exc:
                                    logging.error(
                                        f"[{episode_id}] Exception during OCR for scene {scene_idx}, frame {target_frame_num}, attempt {ocr_attempt_count + 1}: {ocr_exc}",
                                        exc_info=False,
                                    )
                                    error_ocr = str(ocr_exc)
                                    current_scene_data["ocr_errors_in_scene"].append(
                                        f"Frame {target_frame_num} Exception: {error_ocr}"
                                    )

                                ocr_attempt_count += 1
                                if not ocr_success and ocr_attempt_count < MAX_OCR_ATTEMPTS:
                                    time.sleep(0.5)  # Shorter sleep for retries

                            current_scene_data["ocr_results_for_samples"].append(
                                {
                                    "frame_number": target_frame_num,
                                    "ocr_text": text,
                                    # "ocr_details": details, # Avoid storing large details unless necessary
                                    "ocr_error": error_ocr,
                                }
                            )

                            if text and not error_ocr:
                                # Strip the raw text before normalization and logging
                                cleaned_text_for_norm = text.strip()

                                # Log if stripping made a change, and always log details for debugging this specific issue
                                if cleaned_text_for_norm != text:
                                    logging.debug(
                                        f"[{episode_id}] Scene {scene_idx} SampleFrame {target_frame_num}: Stripped OCR raw text from {repr(text)} to {repr(cleaned_text_for_norm)}"
                                    )

                                normalized_text_for_check = utils.normalize_text_for_comparison(
                                    cleaned_text_for_norm, user_stopwords
                                )

                                logging.debug(
                                    f"[{episode_id}] Scene {scene_idx} SampleFrame {target_frame_num}: OCR_Raw={repr(text)}, CleanedForNorm={repr(cleaned_text_for_norm)}, Normalized={repr(normalized_text_for_check)}, LenNormalized={len(normalized_text_for_check)}, MinLength={constants.MIN_OCR_TEXT_LENGTH}, PassesCheck={(len(normalized_text_for_check) >= constants.MIN_OCR_TEXT_LENGTH)})"
                                )

                                if len(normalized_text_for_check) >= constants.MIN_OCR_TEXT_LENGTH:
                                    current_scene_data["has_potential_text"] = True
                                    # Append the cleaned text if it passes checks, not the raw 'text'.
                                    # This ensures that what's stored as a sample is actually valid text.
                                    current_scene_data["text_found_in_samples"].append(cleaned_text_for_norm)
                                    # Save representative frame
                                    try:
                                        # The filename uses temporal_tag_for_scene (formerly position_tag_for_scene in the loop)
                                        frame_save_name = f"scene_{scene_idx:03d}_{temporal_tag_for_scene}_frame_{target_frame_num}_ocr.jpg"
                                        img_pil.save(step1_frames_dir / frame_save_name)
                                        current_scene_data["representative_frames_saved"].append(frame_save_name)
                                    except Exception as e_save_repr:
                                        logging.error(
                                            f"[{episode_id}] Error saving representative frame {target_frame_num} for scene {scene_idx}: {e_save_repr}"
                                        )
                                    break  # Found text in this scene, move to next scene
                    # --- End of OCR sampling logic for the current scene ---

                    if current_scene_data["has_potential_text"]:
                        current_scene_data["selected"] = True  # Add selected flag by default
                        candidate_scenes_data.append(current_scene_data)
                    else:
                        logging.debug(
                            f"[{episode_id}] Scene {scene_idx} (Frames {start_time.get_frames()}-{end_time.get_frames()}, {temporal_tag_for_scene}/{origin_tag_for_scene}) did not yield potential text after sampling."
                        )

        # Determine final status based on results if not already set by specific conditions like "completed_no_scenes_in_segments"
        if status == SceneDetectionStatus.PENDING:  # Default initial status
            if not candidate_scenes_data:
                status = SceneDetectionStatus.COMPLETED_NO_TEXT_IN_VALID_SCENES
            else:
                status = SceneDetectionStatus.COMPLETED_FOUND_CANDIDATE_SCENES

        # Fallback if status somehow remained "pending" (should not happen with above logic)
        if status == SceneDetectionStatus.PENDING:
            log.warning("Status was unexpectedly 'pending'. Defaulting to 'completed_no_text_in_valid_scenes'.")
            status = SceneDetectionStatus.COMPLETED_NO_TEXT_IN_VALID_SCENES

        log.info(
            f"Step 1 processing. Final Status: {status.value}. Found {len(candidate_scenes_data)} candidate scenes."
        )

        output_data_to_save = {
            "episode_id": episode_id,
            "status": status.value,
            "error": None,  # error_message is for exceptions, handled in except block
            "candidate_scenes": candidate_scenes_data,
            "raw_shots_detected": all_detected_shots_data,
            "timestamp": datetime.now().isoformat(),
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data_to_save, f, indent=4, ensure_ascii=False)
        log.info(f"Successfully saved initial scene analysis to {output_file.name}")
        # error_message is None in the success path

    except Exception as e:
        status = SceneDetectionStatus.ERROR
        error_message = str(e)
        logging.error(f"[{episode_id}] Step 1 (identify_candidate_scenes) failed: {e}", exc_info=True)

        # Save error information
        error_output_data = {
            "episode_id": episode_id,
            "status": status.value,
            "error": error_message,
            "candidate_scenes": candidate_scenes_data,  # Save whatever was gathered, if anything
            "raw_shots_detected": all_detected_shots_data,  # Save whatever was gathered
            "timestamp": datetime.now().isoformat(),
        }
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(error_output_data, f, indent=4, ensure_ascii=False)
            log.info(f"Saved error state for initial analysis to {output_file.name}")
        except Exception as e_save:
            logging.error(f"[{episode_id}] Failed to save error status to {output_file.name}: {e_save}")
    finally:
        if 'video' in locals() and video is not None:
            del video

    return candidate_scenes_data, status, error_message
