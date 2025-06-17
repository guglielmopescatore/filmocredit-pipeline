import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import imagehash  # Add this missing import
import numpy as np
from PIL import Image
from scenedetect import FrameTimecode, open_video
from scenedetect.video_stream import VideoStream
from thefuzz import fuzz
from tqdm import tqdm

from . import config, utils, constants
from .utils import calculate_dynamic_fuzzy_threshold


def _compute_image_hash(img: Any) -> Optional[imagehash.ImageHash]:
    """
    Compute average hash for a BGR numpy array image. Returns None on failure.
    """
    try:
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return imagehash.average_hash(pil_img, hash_size=constants.HASH_SIZE)
    except Exception as e:
        logging.warning(f"Failed to compute image hash: {e}")
        return None


# string module already imported above for punctuation removal


def _process_and_save_frame(
    img: np.ndarray,
    episode_id: str,
    frame_num: int,
    frames_output_dir: Path,
    skipped_frames_dir: Path,
    prefix: str,
    ocr_reader: Any,
    ocr_engine_type: str,
    user_stopwords: List[str],
    prev_text: Optional[str],
    prev_hash: Optional[imagehash.ImageHash],
    # Enhanced parameters for dynamic/static processing
    processing_mode: str = "single",  # "single", "dynamic", "static"
    scene_index: Optional[int] = None,
    saved_frame_count: Optional[int] = None,
    scene_position: Optional[str] = None,
    prev_bbox: Optional[Tuple[int, int, int, int]] = None,
    frame_width: Optional[int] = None,
    current_hash: Optional[imagehash.ImageHash] = None,
    fps: Optional[float] = None,
    frame_idx_rel: Optional[int] = None,
    # NEW: Episode-level saved text cache
    episode_saved_texts_cache: Optional[List[str]] = None,
    # NEW: File-to-cache mapping for physical file cleanup
    episode_saved_files_cache: Optional[List[str]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[imagehash.ImageHash], Optional[Tuple[int, int, int, int]]]:
    """
    Run OCR on img, normalize text and apply save vs skip logic.
    Enhanced version that supports single, dynamic scroll, and static frame processing.
    Now includes episode-level cache to prevent duplicates across scenes.

    Args:
        img: Frame image
        episode_id: Episode identifier
        frame_num: Frame number
        frames_output_dir: Directory for saved frames
        skipped_frames_dir: Directory for skipped frames
        prefix: Base filename prefix
        ocr_reader: OCR engine instance
        ocr_engine_type: OCR engine type string
        user_stopwords: Stopwords for text normalization
        prev_text: Previous frame text for comparison
        prev_hash: Previous frame hash for comparison
        processing_mode: "single", "dynamic", or "static"
        scene_index: Scene index for naming (dynamic/static)
        saved_frame_count: Current count of saved frames (dynamic/static)
        scene_position: Scene position metadata
        prev_bbox: Previous frame bounding box
        frame_width: Frame width for edge detection
        current_hash: Current frame hash (static mode)
        fps: Frames per second (for checks)        frame_idx_rel: Relative frame index in scene (static mode)
        episode_saved_texts_cache: List of normalized texts that have been saved in this episode
        episode_saved_files_cache: List of file paths corresponding to episode_saved_texts_cache entries

    Returns:
        (metadata, new_text, new_hash, new_bbox) if saved, or (None, prev_text, prev_hash, prev_bbox) if skipped.

        IMPORTANT: The text/hash/bbox state should ONLY be updated when a frame is actually saved.
        When skipped, the previous state must be returned unchanged.
    """

    # Initialize episode cache if not provided
    if episode_saved_texts_cache is None:
        episode_saved_texts_cache = []

    # For ALL modes: check fade frame FIRST, before doing any OCR or text processing
    if utils.is_fade_frame(img):
        skip_reason = "fade_frame"
        mode_tag = f"{processing_mode}_" if processing_mode != "single" else ""
        skip_path = skipped_frames_dir / f"{prefix}{mode_tag}skipped_{skip_reason}_num{frame_num:05d}.jpg"

        # Ensure the skipped directory exists
        skipped_frames_dir.mkdir(parents=True, exist_ok=True)

        try:
            success = cv2.imwrite(str(skip_path), img)
            if success:
                logging.info(
                    f"[{episode_id}] Frame {frame_num} ({processing_mode}) SKIPPED: fade frame detected - saved to {skip_path.name} - preserving previous state"
                )
            else:
                logging.error(
                    f"[{episode_id}] Frame {frame_num} ({processing_mode}) SKIPPED: fade frame detected but failed to save to {skip_path}"
                )
        except Exception as e:
            logging.error(
                f"[{episode_id}] Frame {frame_num} ({processing_mode}) SKIPPED: fade frame detected but error saving: {e}"
            )

        # Return previous state unchanged - NO OCR was done, prev_text should remain as-is
        return None, prev_text, prev_hash, prev_bbox

    # OCR and normalize
    try:
        text_res, _, bbox_res, ocr_error = utils.run_ocr(
            img,
            ocr_reader,
            ocr_engine_type,
            image_context_identifier=f"{episode_id}_frame{frame_num}_{processing_mode}",
        )
    except Exception as e:
        logging.warning(f"[{episode_id}] OCR exception frame {frame_num}: {e}")
        text_res, bbox_res, ocr_error = None, None, str(e)

    # ALWAYS log OCR results - whether successful or not
    if ocr_error:
        logging.info(f"[{episode_id}] Frame {frame_num} ({processing_mode}) OCR FAILED: {ocr_error}")
    else:
        logging.info(
            f"[{episode_id}] Frame {frame_num} ({processing_mode}) OCR SUCCESS: '{text_res or 'EMPTY'}' (length: {len(text_res or '')})"
        )

    # Normalize text using utils function to avoid duplication
    norm_text = utils.normalize_text_for_comparison(text_res or "", user_stopwords)

    # ALWAYS log normalization results for comparison debugging
    logging.info(f"[{episode_id}] Frame {frame_num} ({processing_mode}) TEXT PROCESSING:")
    logging.info(f"  - Raw OCR: '{text_res or 'NONE'}'")
    logging.info(f"  - Normalized: '{norm_text}' (length: {len(norm_text)})")

    # Log previous frame context for comparison
    if prev_text:
        logging.info(f"[{episode_id}] Frame {frame_num} ({processing_mode}) PREVIOUS FRAME TEXT: '{prev_text}'")
        logging.info(
            f"[{episode_id}] Frame {frame_num} ({processing_mode}) TEXT COMPARISON: '{norm_text}' vs '{prev_text}'"
        )

    # Check for sufficient text length
    has_sufficient_text = len(norm_text) >= constants.MIN_OCR_TEXT_LENGTH

    # For static mode: hash difference check (after OCR but before text comparison)
    if processing_mode == "static" and current_hash is not None and prev_hash is not None:
        hash_diff = utils.calculate_hash_difference(current_hash, prev_hash)
        logging.info(
            f"[{episode_id}] Frame {frame_num} (static) HASH DIFF: {hash_diff} (threshold: {constants.HASH_DIFFERENCE_THRESHOLD})"
        )

        if hash_diff < constants.HASH_DIFFERENCE_THRESHOLD:
            # Insufficient visual difference - preserve previous state (do NOT update with current frame's OCR)
            skip_reason = f"similar_hash_{hash_diff}"
            skip_path = skipped_frames_dir / f"{prefix}static_skipped_{skip_reason}_num{frame_num:05d}.jpg"
            cv2.imwrite(str(skip_path), img)
            logging.info(
                f"[{episode_id}] Frame {frame_num} (static) SKIPPED: similar hash {hash_diff} < {constants.HASH_DIFFERENCE_THRESHOLD} - preserving previous state"
            )
            return None, prev_text, prev_hash, prev_bbox
        elif not has_sufficient_text:
            # Visual difference but no text - preserve previous state (do NOT update with current frame's OCR)
            skip_reason = f"diff_hash_{hash_diff}_no_text"
            skip_path = skipped_frames_dir / f"{prefix}static_skipped_{skip_reason}_num{frame_num:05d}.jpg"
            cv2.imwrite(str(skip_path), img)
            logging.info(
                f"[{episode_id}] Frame {frame_num} (static) SKIPPED: visual diff but no text - preserving previous state"
            )
            return None, prev_text, prev_hash, prev_bbox

    # Skip for insufficient text (applies to all modes) - preserve previous state (do NOT update with current frame's OCR)
    if not has_sufficient_text:
        skip_reason = f"no_text_len_{len(norm_text)}"
        mode_tag = f"{processing_mode}_" if processing_mode != "single" else ""
        skip_path = skipped_frames_dir / f"{prefix}{mode_tag}skipped_{skip_reason}_num{frame_num:05d}.jpg"
        cv2.imwrite(str(skip_path), img)
        logging.warning(
            f"[{episode_id}] Frame {frame_num} ({processing_mode}) SKIPPED: insufficient text length {len(norm_text)} < {constants.MIN_OCR_TEXT_LENGTH} - preserving previous state"
        )
        return None, prev_text, prev_hash, prev_bbox

    # Compare current frame text with the last saved text in the episode (if any)
    text_similarity = 0
    save_frame = True

    if episode_saved_texts_cache:  # Compare with the most recently saved text (last item in cache)
        last_saved_text = episode_saved_texts_cache[-1]
        text_similarity = fuzz.token_sort_ratio(norm_text, last_saved_text)

        # Calculate dynamic threshold based on average text length
        avg_text_length = (len(norm_text) + len(last_saved_text)) // 2
        dynamic_threshold = calculate_dynamic_fuzzy_threshold(avg_text_length)

        logging.info(f"[{episode_id}] Frame {frame_num} ({processing_mode}) EPISODE-LEVEL SIMILARITY:")
        logging.info(f"  - Current: '{norm_text}' (length: {len(norm_text)})")
        logging.info(f"  - Last saved: '{last_saved_text}' (length: {len(last_saved_text)})")
        logging.info(f"  - Similarity: {text_similarity}% (dynamic threshold: {dynamic_threshold}%)")

        # Decide whether to save or skip based on similarity to last saved frame
        if text_similarity > dynamic_threshold:
            # Texts are similar - decide based on length (keep the longer one)
            current_length = len(norm_text)
            last_saved_length = len(last_saved_text)

            if current_length > last_saved_length:
                # Current text is longer - save current and remove the shorter one from cache
                save_frame = True

                # Remove the last saved text from cache since we're replacing it with longer text
                episode_saved_texts_cache.pop()

                # Also remove the corresponding file if file cache exists
                if episode_saved_files_cache and len(episode_saved_files_cache) > 0:
                    removed_file_path = episode_saved_files_cache.pop()
                    try:
                        # Move the old file to skipped directory instead of deleting
                        removed_file = Path(removed_file_path)
                        if removed_file.exists():
                            replacement_name = f"replaced_shorter_{removed_file.name}"
                            replacement_path = skipped_frames_dir / replacement_name
                            skipped_frames_dir.mkdir(parents=True, exist_ok=True)
                            removed_file.rename(replacement_path)
                            logging.info(f"[{episode_id}] Moved replaced shorter frame to: {replacement_path.name}")
                    except Exception as e:
                        logging.warning(f"[{episode_id}] Could not move replaced file {removed_file_path}: {e}")

                logging.info(
                    f"[{episode_id}] Frame {frame_num} ({processing_mode}) DECISION: SAVE - replacing shorter text ({last_saved_length} chars) with longer text ({current_length} chars)"
                )
                logging.info(
                    f"[{episode_id}] Removed shorter text from episode cache. Cache size: {len(episode_saved_texts_cache)}"
                )
            else:
                # Last saved text is longer or equal - skip current
                save_frame = False
                skip_reason = f"similar_to_last_saved_shorter_{text_similarity}"
                logging.info(
                    f"[{episode_id}] Frame {frame_num} ({processing_mode}) DECISION: SKIP - current text ({current_length} chars) is shorter than or equal to last saved ({last_saved_length} chars)"
                )
        else:
            logging.info(
                f"[{episode_id}] Frame {frame_num} ({processing_mode}) DECISION: SAVE - sufficiently different from last saved frame"
            )
    else:
        logging.info(
            f"[{episode_id}] Frame {frame_num} ({processing_mode}) EPISODE-LEVEL SIMILARITY: No previous saved text - similarity: {text_similarity}% (first frame)"
        )
        logging.info(f"[{episode_id}] Frame {frame_num} ({processing_mode}) DECISION: SAVE - first frame in episode")

    # Skip frame if save_frame is False
    if not save_frame:
        mode_tag = f"{processing_mode}_" if processing_mode != "single" else ""
        skip_path = skipped_frames_dir / f"{prefix}{mode_tag}skipped_{skip_reason}_num{frame_num:05d}.jpg"
        skipped_frames_dir.mkdir(parents=True, exist_ok=True)

        try:
            success = cv2.imwrite(str(skip_path), img)
            if success:
                logging.info(
                    f"[{episode_id}] Frame {frame_num} ({processing_mode}) SKIPPED: {skip_reason} - saved to {skip_path.name}"
                )
            else:
                logging.error(
                    f"[{episode_id}] Frame {frame_num} ({processing_mode}) SKIPPED: {skip_reason} but failed to save to {skip_path}"
                )
        except Exception as e:
            logging.error(
                f"[{episode_id}] Frame {frame_num} ({processing_mode}) SKIPPED: {skip_reason} but error saving: {e}"
            )

        return None, prev_text, prev_hash, prev_bbox
    # If we reach here, the frame WILL be saved - so we can update the comparison state
    if save_frame:
        # Generate appropriate filename based on mode
        frame_count = saved_frame_count if saved_frame_count is not None else 0
        if processing_mode == "dynamic":
            out_path = (
                frames_output_dir / f"{prefix}scroll_{frame_count:03d}_num{frame_num:05d}_sim{int(text_similarity)}.jpg"
            )
        elif processing_mode == "static":
            out_path = (
                frames_output_dir / f"{prefix}static_{frame_count:03d}_num{frame_num:05d}_sim{int(text_similarity)}.jpg"
            )
        else:  # single
            out_path = frames_output_dir / f"{prefix}frame_num{frame_num:05d}_sim{int(text_similarity)}.jpg"

        try:
            success = cv2.imwrite(str(out_path), img)
            if success:
                # ONLY ADD TO EPISODE CACHE WHEN SUCCESSFULLY SAVED
                episode_saved_texts_cache.append(norm_text)
                logging.info(
                    f"[{episode_id}] Frame {frame_num} ({processing_mode}) SAVED and ADDED TO EPISODE CACHE: '{norm_text[:50]}...' (cache size: {len(episode_saved_texts_cache)})"
                )

                metadata = {
                    "path": str(out_path.relative_to(config.EPISODES_BASE_DIR)),
                    "frame_num": frame_num,
                    "ocr_text": text_res or "",
                    "ocr_bbox": bbox_res,
                    "scene_position": scene_position,
                }
                logging.info(f"[{episode_id}] Frame {frame_num} ({processing_mode}) SAVED: {out_path.name}")
                # Add the normalized text to episode cache for future comparisons
                if episode_saved_texts_cache is not None:
                    episode_saved_texts_cache.append(norm_text)
                    logging.info(
                        f"[{episode_id}] Added text to episode cache. Cache size: {len(episode_saved_texts_cache)}"
                    )

                # Add the file path to episode files cache for file management
                if episode_saved_files_cache is not None:
                    episode_saved_files_cache.append(str(out_path))
                    logging.info(
                        f"[{episode_id}] Added file path to episode files cache. Files cache size: {len(episode_saved_files_cache)}"
                    )

                # ONLY update comparison state when frame is successfully saved
                return metadata, norm_text, current_hash or _compute_image_hash(img), bbox_res
            else:
                logging.error(f"[{episode_id}] Failed to save frame {frame_num}: cv2.imwrite returned False")
        except Exception as e:
            logging.error(f"[{episode_id}] Failed to save frame {frame_num}: {e}")

    # Fallback: if save fails, preserve previous state (do NOT update with current frame's OCR)
    mode_tag = f"{processing_mode}_" if processing_mode != "single" else ""
    skip_path = skipped_frames_dir / f"{prefix}{mode_tag}skipped_save_failed_num{frame_num:05d}.jpg"

    # Ensure the skipped directory exists
    skipped_frames_dir.mkdir(parents=True, exist_ok=True)

    try:
        success = cv2.imwrite(str(skip_path), img)
        if success:
            logging.info(
                f"[{episode_id}] Frame {frame_num} ({processing_mode}) SKIPPED: save failed - saved to {skip_path.name} - preserving previous state"
            )
        else:
            logging.error(
                f"[{episode_id}] Frame {frame_num} ({processing_mode}) SKIPPED: save failed and couldn't save to skipped either"
            )
    except Exception as e:
        logging.error(
            f"[{episode_id}] Frame {frame_num} ({processing_mode}) SKIPPED: save failed and error saving to skipped: {e}"
        )

    # Return previous state unchanged since the save failed
    return None, prev_text, prev_hash, prev_bbox


def analyze_candidate_scene_frames(
    video_path: Path,
    episode_id: str,
    scene_info: Dict[str, Any],
    fps: float,
    frame_height: int,
    frame_width: int,
    ocr_reader: Any,
    ocr_engine_type: str,
    user_stopwords: List[str],
    global_last_saved_ocr_text_input: Optional[str],
    global_last_saved_frame_hash_input: Optional[imagehash.ImageHash],
    global_last_saved_ocr_bbox_input: Optional[Tuple[int, int, int, int]],
    # NEW: Episode-level saved text cache
    episode_saved_texts_cache: Optional[List[str]] = None,
    # NEW: File-to-cache mapping for physical file cleanup
    episode_saved_files_cache: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Optional[str], Optional[imagehash.ImageHash], Optional[Tuple[int, int, int, int]]]:
    """
    Analyze and save representative frames for a candidate scene.
    Now supports episode-level text deduplication cache.

    Args:
        video_path: Path to the video file.
        episode_id: Identifier for the episode.
        scene_info: Dict with scene indices and frame boundaries.
        fps: Frames per second of the video.
        frame_height: Height of video frames in pixels.
        frame_width: Width of video frames in pixels.
        ocr_reader: Initialized OCR engine.
        ocr_engine_type: String key for OCR engine type.
        user_stopwords: List of stopwords to ignore in OCR comparison.
        global_last_saved_ocr_text_input: Previously saved OCR text for deduplication.
        global_last_saved_frame_hash_input: Previously saved frame hash.        global_last_saved_ocr_bbox_input: Previously saved bounding box.
        episode_saved_texts_cache: List of normalized texts already saved in this episode
        episode_saved_files_cache: List of file paths corresponding to episode_saved_texts_cache entries

    Returns:
        analysis_info: Dict summarizing frame analysis results.
        last_saved_text_output: The OCR text of the last saved frame (or None).
        last_saved_hash_output: Image hash of the last saved frame (or None).
        last_saved_bbox_output: Bounding box of detected text (or None).
    """
    # Initialize episode cache if not provided
    if episode_saved_texts_cache is None:
        episode_saved_texts_cache = []

    # Initialize episode files cache if not provided
    if episode_saved_files_cache is None:
        episode_saved_files_cache = []

    episode_dir = config.EPISODES_BASE_DIR / episode_id
    analysis_dir = episode_dir / 'analysis'
    frames_output_dir = analysis_dir / 'frames'
    skipped_frames_dir = analysis_dir / 'skipped_frames'
    frames_output_dir.mkdir(parents=True, exist_ok=True)
    skipped_frames_dir.mkdir(parents=True, exist_ok=True)

    scene_index = scene_info['scene_index']
    base_filename_prefix = f"scene_{scene_index:03d}_"
    logging.info(
        f"[{episode_id}] Step 2: Analyzing frames for scene {scene_index} ({scene_info['start_frame']}-{scene_info['end_frame']}, Position: {scene_info.get('position', 'unknown')})..."
    )

    current_scene_position = scene_info.get('position', 'unknown')

    analysis_info = {
        "type": "unknown",
        "output_files": [],
        "frame_count": 0,
        "median_flow_px_per_frame": 0.0,
        "status": "pending",
        "position": current_scene_position,
    }

    current_comparator_text: str | None = global_last_saved_ocr_text_input
    current_comparator_hash: Optional[imagehash.ImageHash] = global_last_saved_frame_hash_input
    current_comparator_bbox: Optional[Tuple[int, int, int, int]] = global_last_saved_ocr_bbox_input

    last_saved_text_output: str | None = None
    last_saved_hash_output: Optional[imagehash.ImageHash] = None
    last_saved_bbox_output: Optional[Tuple[int, int, int, int]] = None

    frames_data = []
    video: VideoStream = None
    try:
        video = open_video(str(video_path))
        video.seek(FrameTimecode(scene_info['start_frame'], fps))

        pbar_read = tqdm(
            total=scene_info['end_frame'] - scene_info['start_frame'],
            desc=f"Read scene {scene_index}",
            unit="frame",
            leave=False,
        )
        current_frame_num = scene_info['start_frame']
        while current_frame_num < scene_info['end_frame']:
            frame_img = video.read(decode=True)
            if frame_img is False:
                break
            # Ensure np is always the global numpy, not shadowed
            import numpy as np

            if frame_img is None or not isinstance(frame_img, np.ndarray) or frame_img.size == 0:
                logging.warning(f"[{episode_id}] Skipping invalid frame at approx {current_frame_num}")
                current_frame_num = video.position.get_frames()
                pbar_read.n = min(
                    current_frame_num - scene_info['start_frame'], scene_info['end_frame'] - scene_info['start_frame']
                )
                pbar_read.refresh()
                continue
            frames_data.append({"num": current_frame_num, "img": frame_img.copy()})
            pbar_read.update(1)
            current_frame_num += 1
        pbar_read.close()

        if not frames_data:
            logging.warning(f"[{episode_id}] No valid frames read for scene {scene_index}.")
            analysis_info["type"] = "no_frames_read"
            analysis_info["status"] = "completed_no_frames"
            return analysis_info, last_saved_text_output, last_saved_hash_output, last_saved_bbox_output

        num_frames = len(frames_data)
        if num_frames < 2:
            analysis_info["type"] = "single_frame" if num_frames == 1 else "empty"
            if num_frames == 1:
                # Single-frame: use helper to process and save/skip
                frame_data = frames_data[0]
                frame_num = frame_data.get("num", -1)
                metadata, new_text, new_hash, new_bbox = _process_and_save_frame(
                    frame_data.get("img"),
                    episode_id,
                    frame_num,
                    frames_output_dir,
                    skipped_frames_dir,
                    base_filename_prefix,
                    ocr_reader,
                    ocr_engine_type,
                    user_stopwords,
                    current_comparator_text,
                    current_comparator_hash,
                    processing_mode="single",
                    scene_position=current_scene_position,
                )
                if metadata:
                    analysis_info["output_files"] = [metadata]
                    analysis_info["frame_count"] = 1
                    last_saved_text_output = new_text
                    last_saved_hash_output = new_hash
                    last_saved_bbox_output = new_bbox
            analysis_info["status"] = "completed_single_frame"
            return analysis_info, last_saved_text_output, last_saved_hash_output, last_saved_bbox_output

        apply_dynamic_scroll_logic = False
        median_v_flow = 0.0

        if scene_info.get('position', 'unknown') == "second_half":
            # Vectorized optical flow computation: precompute grayscale frames
            try:
                gray_frames = [cv2.cvtColor(fd["img"], cv2.COLOR_BGR2GRAY) for fd in frames_data]
                # Compute vertical flows between consecutive frames
                vertical_flows = [utils.calculate_vertical_flow(g1, g2) for g1, g2 in zip(gray_frames, gray_frames[1:])]
                # Compute and store median vertical flow for scroll detection
                median_v_flow = float(np.median(vertical_flows)) if vertical_flows else 0.0
                analysis_info["median_flow_px_per_frame"] = median_v_flow
            except Exception as e:
                logging.error(f"[{episode_id}] Error during optimized flow calc for scene {scene_index}: {e}")
                analysis_info["type"] = "error_flow_calculation"
                analysis_info["status"] = "error"
                return analysis_info, last_saved_text_output, last_saved_hash_output, last_saved_bbox_output

            if abs(median_v_flow) > constants.MIN_ABS_SCROLL_FLOW_THRESHOLD:
                apply_dynamic_scroll_logic = True
            else:
                pass
        elif scene_info.get('position', 'unknown') == "first_half":
            analysis_info["type"] = "static_first_half"

        else:
            logging.warning(
                f"[{episode_id}] Scene {scene_index} has unknown position '{scene_info.get('position', 'unknown')}'. Defaulting to static/slow logic."
            )

        output_files_data = []
        saved_frame_count = 0

        if apply_dynamic_scroll_logic:
            analysis_info["type"] = "dynamic_scroll"
            scroll_pixels_per_frame = abs(median_v_flow)
            pixels_to_scroll_before_save = frame_height * constants.SCROLL_FRAME_HEIGHT_RATIO
            frames_per_save = max(1, int(round(pixels_to_scroll_before_save / scroll_pixels_per_frame)))
            analysis_info["save_interval_frames"] = int(frames_per_save)
            analysis_info["target_scroll_ratio"] = float(constants.SCROLL_FRAME_HEIGHT_RATIO)

            # Iterate over frames at computed interval with progress bar
            for i in tqdm(
                range(0, num_frames, frames_per_save),
                desc=f"Select scroll frames scene {scene_index}",
                unit="frame",
                leave=False,
            ):
                logging.info("-" * 80)
                frame_data = frames_data[i]
                full_frame_img = frame_data.get("img")
                frame_num = frame_data.get("num", -1)

                if full_frame_img is None or full_frame_img.size == 0:
                    continue

                logging.info(
                    f"[{episode_id}] Scene {scene_index} Frame {frame_num} (Dynamic Scroll Sample) Processing..."
                )

                # Use unified frame processing WITH episode cache
                metadata, new_text, new_hash, new_bbox = _process_and_save_frame(
                    full_frame_img,
                    episode_id,
                    frame_num,
                    frames_output_dir,
                    skipped_frames_dir,
                    base_filename_prefix,
                    ocr_reader,
                    ocr_engine_type,
                    user_stopwords,
                    current_comparator_text,
                    current_comparator_hash,
                    processing_mode="dynamic",
                    scene_index=scene_index,
                    saved_frame_count=saved_frame_count,
                    scene_position=current_scene_position,
                    prev_bbox=current_comparator_bbox,
                    frame_width=frame_width,
                    episode_saved_texts_cache=episode_saved_texts_cache,  # Pass episode cache
                    episode_saved_files_cache=episode_saved_files_cache,  # Pass episode files cache
                )

                if metadata:
                    output_files_data.append(metadata)
                    saved_frame_count += 1
                    last_saved_text_output = new_text
                    last_saved_hash_output = new_hash
                    last_saved_bbox_output = new_bbox
                    current_comparator_text = new_text
                    current_comparator_hash = new_hash
                    current_comparator_bbox = new_bbox

        else:
            if analysis_info["type"] == "unknown":
                analysis_info["type"] = "static_or_slow"

            scene_duration_seconds = num_frames / fps if fps > 0 else 0

            if scene_duration_seconds > constants.HASH_SAMPLE_INTERVAL_SECONDS and num_frames > 1 and fps > 0:
                frame_step = max(1, int(round(fps * constants.HASH_SAMPLE_INTERVAL_SECONDS)))
                sampled_indices_rel = [0] + list(range(frame_step, num_frames, frame_step))
                if num_frames > 0 and (num_frames - 1) not in sampled_indices_rel:
                    sampled_indices_rel.append(num_frames - 1)
                sampled_indices_rel = sorted(list(set(sampled_indices_rel)))

                pbar_hash = tqdm(
                    sampled_indices_rel, desc=f"Hash/OCR(sq) scene {scene_index}", unit="sample", leave=False
                )

                for frame_idx_rel in pbar_hash:
                    if frame_idx_rel >= len(frames_data):
                        continue

                    logging.info("-" * 80)
                    frame_data = frames_data[frame_idx_rel]
                    full_frame_img = frame_data.get("img")
                    frame_num = frame_data.get("num", -1)

                    if full_frame_img is None or full_frame_img.size == 0:
                        continue

                    logging.info(f"[{episode_id}] Scene {scene_index} Frame {frame_num} (Static Sample) Processing...")

                    # Compute hash for static processing
                    current_frame_hash = _compute_image_hash(full_frame_img)

                    # Use unified frame processing WITH episode cache
                    metadata, new_text, new_hash, new_bbox = _process_and_save_frame(
                        full_frame_img,
                        episode_id,
                        frame_num,
                        frames_output_dir,
                        skipped_frames_dir,
                        base_filename_prefix,
                        ocr_reader,
                        ocr_engine_type,
                        user_stopwords,
                        current_comparator_text,
                        current_comparator_hash,
                        processing_mode="static",
                        scene_index=scene_index,
                        saved_frame_count=saved_frame_count,
                        scene_position=current_scene_position,
                        prev_bbox=current_comparator_bbox,
                        frame_width=frame_width,
                        current_hash=current_frame_hash,
                        fps=fps,
                        frame_idx_rel=frame_idx_rel,
                        episode_saved_texts_cache=episode_saved_texts_cache,  # Pass episode cache
                        episode_saved_files_cache=episode_saved_files_cache,  # Pass episode files cache
                    )

                    if metadata:
                        output_files_data.append(metadata)
                        saved_frame_count += 1
                        last_saved_text_output = new_text
                        last_saved_hash_output = new_hash
                        last_saved_bbox_output = new_bbox
                        current_comparator_text = new_text
                        current_comparator_hash = new_hash
                        current_comparator_bbox = new_bbox

            else:
                pass

        analysis_info["output_files"] = output_files_data
        analysis_info["frame_count"] = saved_frame_count
        analysis_info["status"] = "completed"

    except Exception as e:
        logging.error(f"[{episode_id}] Step 2 failed unexpectedly for scene {scene_index}: {e}", exc_info=True)
        analysis_info["status"] = "error"
        analysis_info["error_message"] = str(e)
    finally:
        del frames_data
        if 'video' in locals() and video is not None:
            del video

    return analysis_info, last_saved_text_output, last_saved_hash_output, last_saved_bbox_output
