import os
import json
import cv2
import easyocr
import scenedetect
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import time
import shutil  # Per pulizia directory

# === CONFIGURAZIONE ===
CONFIG_PATH = 'config.json'

with open(CONFIG_PATH, 'r', encoding='utf-8') as config_file:
    config = json.load(config_file)

RAW_VIDEO_DIR = config.get('raw_video_dir', 'data/raw/')
OUTPUT_BASE_DIR = config.get('ocr_base_dir', 'data/processed/')
OCR_LANGUAGES = config.get('ocr_languages', ['it', 'en'])
SCENE_DETECTION_THRESHOLD = config.get('scene_detection_threshold', 30.0)
MIN_SCENE_LENGTH_SEC = config.get('min_scene_length_sec', 2)
FRAME_SAMPLE_POINTS = config.get('frame_sample_points', [0.25, 0.5, 0.75])
ROTATION_ANGLES = config.get('rotation_angles', [0, 90, 270])
MAX_WORKERS_GPU = config.get('max_workers_gpu', 2)
MAX_WORKERS_CPU = config.get('max_workers_cpu', 8)

# === CREA CARTELLE BASE ===
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# === CONFIGURAZIONE GPU ===
try:
    import torch
    use_gpu = torch.cuda.is_available()
except ImportError:
    use_gpu = False

# === INIZIALIZZA OCR ===
reader = easyocr.Reader(OCR_LANGUAGES, gpu=use_gpu)

# === CONFIGURA NUMERO WORKER IN BASE ALLA GPU ===
MAX_WORKERS = MAX_WORKERS_GPU if use_gpu else MAX_WORKERS_CPU

def processa_videos_batch(progress_callback=None, progress_video=None, status_placeholder=None):
    """
    Processa tutti i video nella directory RAW_VIDEO_DIR:
    - Rileva scene
    - Esegue OCR su punti campione
    - Salva un solo frame per scena (migliore o centrale)
    - Genera ocr_results.json includendo tutte le scene
    """
    # Backup di tutte le cartelle già processate
    backup_root = os.path.join(OUTPUT_BASE_DIR, 'backup')
    os.makedirs(backup_root, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for name in os.listdir(OUTPUT_BASE_DIR):
        dir_path = os.path.join(OUTPUT_BASE_DIR, name)
        if name == 'backup' or not os.path.isdir(dir_path):
            continue
        shutil.move(dir_path, os.path.join(backup_root, f"{name}_{timestamp}"))

    video_files = [f for f in os.listdir(RAW_VIDEO_DIR)
                   if f.lower().endswith((".mp4", ".avi", ".mkv"))]

    for idx_video, video_file in enumerate(video_files):
        video_path = os.path.join(RAW_VIDEO_DIR, video_file)
        video_name = os.path.splitext(video_file)[0]

        # Backup o pulizia output per run pulito
        output_dir = os.path.join(OUTPUT_BASE_DIR, video_name)
        if os.path.exists(output_dir):
            # Sposta la vecchia cartella in backup con timestamp
            backup_root = os.path.join(OUTPUT_BASE_DIR, 'backup')
            os.makedirs(backup_root, exist_ok=True)
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = os.path.join(backup_root, f"{video_name}_{timestamp}")
            shutil.move(output_dir, backup_dir)
        # Ricrea struttura pulita
        frames_dir = os.path.join(output_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)

        if status_placeholder:
            status_placeholder.info(f"Elaborazione {idx_video+1}/{len(video_files)}: {video_file}")

        ocr_data = []
        start_total = time.perf_counter()

        # Scene detection
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=SCENE_DETECTION_THRESHOLD))
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()

        cap = cv2.VideoCapture(video_path)
        start_ocr = time.perf_counter()
        start_seek = start_ocr

        for idx_scene, (t0, t1) in enumerate(scene_list):
            start_sec = t0.get_seconds()
            end_sec = t1.get_seconds()
            duration = end_sec - start_sec

            # Calcola timestamp di sample (si includono tutte le scene)
            timestamps = [start_sec + duration * p for p in FRAME_SAMPLE_POINTS]

            best_text = ""
            best_frame = None
            best_variant = None

            # ESEGUE OCR sui frame campione, senza salvarli
            for j, ts in enumerate(timestamps):
                variant = int(FRAME_SAMPLE_POINTS[j] * 100)
                cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
                ret, frame = cap.read()
                if not ret:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # OCR
                results = reader.readtext(rgb, rotation_info=ROTATION_ANGLES)
                text = "\n".join([r[1] for r in results])
                if len(text.strip()) > len(best_text.strip()):
                    best_text = text
                    best_frame = rgb
                    best_variant = variant

            # Se nessun testo rilevato, fallback al frame centrale
            if best_frame is None and timestamps:
                mid = len(FRAME_SAMPLE_POINTS) // 2
                ts = timestamps[mid]
                cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
                ret, frame = cap.read()
                if ret:
                    best_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    best_variant = int(FRAME_SAMPLE_POINTS[mid] * 100)
                    best_text = ""

            # Salva un solo frame selezionato
            frame_path = None
            if best_frame is not None:
                fname = f"frame_{idx_scene}_{best_variant}.jpg"
                frame_path = os.path.join(frames_dir, fname)
                cv2.imwrite(frame_path, cv2.cvtColor(best_frame, cv2.COLOR_RGB2BGR))

            # Registra la scena sempre
            ocr_data.append({
                "scene_idx": idx_scene,
                "frame_variant": best_variant,
                "timestamp_sec": (start_sec + end_sec) / 2,
                "scene_start_sec": start_sec,
                "scene_end_sec": end_sec,
                "ocr_text": best_text,
                "frame_path": frame_path
            })

        # Rilascio risorse
        cap.release()
        video_manager.release()

        # Scrive JSON con tutte le scene
        out_path = os.path.join(output_dir, 'ocr_results.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(ocr_data, f, indent=2, ensure_ascii=False)

        end_total = time.perf_counter()
        print(f"✅ {video_file}: {len(ocr_data)} scene processate. JSON salvato in {out_path}")
        print(f"⏱️ Tempo totale: {end_total - start_total:.2f}s, OCR: {time.perf_counter()-start_ocr:.2f}s, frame seek: {time.perf_counter()-start_seek:.2f}s")

        if progress_callback and progress_video:
            progress_callback(idx_video+1, len(video_files))
