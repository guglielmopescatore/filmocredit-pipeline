import streamlit as st
import os
import tempfile
import subprocess
from PIL import Image
import pytesseract
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import easyocr
import numpy as np
import cv2
import torch
import re
import time


# Verifica disponibilità GPU
use_gpu = torch.cuda.is_available()

# Mostra stato GPU prima del titolo
if use_gpu:
    st.success("🟢 EasyOCR: GPU attiva e pronta!")
else:
    st.error("🔴 EasyOCR: GPU non disponibile, userò la CPU (più lento)")

# Crea il reader EasyOCR
reader = easyocr.Reader(['it', 'en'], gpu=use_gpu)

st.title("Analisi segmenti titoli di testa/coda")

# Upload del video
video_file = st.file_uploader("Carica un file video", type=["mp4", "mkv", "avi"])

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_path = tmp_file.name
        tmp_file.write(video_file.read())

    st.video(tmp_path)

    # Segmentazione delle scene
    st.write("## Segmentazione del video in scene")
    video_manager = VideoManager([tmp_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30.0))

    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    scene_list = scene_manager.get_scene_list()

    st.write(f"Sono state trovate {len(scene_list)} scene.")

    selected_scenes = []

    frame_position = st.selectbox(
    "Posizione del frame da analizzare per ciascuna scena",
    ["inizio", "centro", "fine"],
    index=1
    )

    # Avvio timer totale
    total_start_time = time.perf_counter()
    ocr_total_time = 0

    for i, (start_time, end_time) in enumerate(scene_list):
        # Calcola i tre timestamp al 25%, 50% e 75% della durata della scena
        start_sec = start_time.get_seconds()
        end_sec = end_time.get_seconds()
        duration = end_sec - start_sec

        sample_points = [0.25, 0.5, 0.75]
        timestamps = [start_sec + duration * p for p in sample_points]

        best_text = ""
        best_frame_path = None

        for j, ts in enumerate(timestamps):
            frame_path = os.path.join(tempfile.gettempdir(), f"frame_{i}_{int(sample_points[j]*100)}.jpg")
            subprocess.run([
                "ffmpeg", "-y", "-ss", str(ts), "-i", tmp_path,
                "-frames:v", "1", "-q:v", "2", frame_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            frame_ocr_start = time.perf_counter()
            image = Image.open(frame_path)
            image_np = np.array(image)

            # OCR diretto con EasyOCR con gestione titoli ruotati
            easy_results = reader.readtext(image_np, rotation_info=[90, 270])
            text = "\n".join([res[1] for res in easy_results])

            frame_ocr_end = time.perf_counter()
            ocr_total_time += frame_ocr_end - frame_ocr_start

            if re.search(r"[a-zA-ZÀ-ÿ]", text):
                best_text = text
                best_frame_path = frame_path
                break  # trovato testo valido, non servono altri frame

        # Pulizia e preparazione
        text = best_text.strip()
        word_list = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
        word_count = len(word_list)

        # Zona narrativa ampia
        force_show = i < 40 or i >= len(scene_list) - 40

        # Titoli brevi noti
        ok_short_titles = {"CAST", "SUONO", "FOTO", "AIUTO", "COST", "VOCE", "LUCI"}

        # Analisi parole
        has_meaningful_word = any(
            len(w) >= 4 and re.search(r"[a-zA-ZÀ-ÿ]", w) for w in word_list
        )
        short_caps = [w for w in word_list if len(w) <= 4 and w.isupper()]
        short_words = [w for w in word_list if len(w) <= 2]

        # Motivazione di scarto
        filter_reason = ""

        # Applicazione filtro
        if (
            word_count <= 4 and len(short_caps) == word_count and not any(w in ok_short_titles for w in short_caps)
        ) and not force_show:
            filter_reason = "Solo parole brevi maiuscole"
        elif (
            len(short_words) > word_count / 2 and not has_meaningful_word
        ) and not force_show:
            filter_reason = "Prevalenza di parole cortissime"

        # Se abbiamo un motivo valido di scarto, saltiamo la scena
        if filter_reason:
            with st.sidebar:
                st.write(f"⛔ Scena {i+1} scartata: {filter_reason}")
            continue

        # Filtro standard soft/severo
        if force_show:
            text_is_valid = word_count >= 2
        else:
            text_is_valid = word_count >= 4

        if not text_is_valid and not force_show:
            with st.sidebar:
                st.write(f"⛔ Scena {i+1} scartata: troppo poche parole ({word_count})")
            continue


        # Visualizzazione
        st.write(f"### Scena {i+1} ({start_time} - {end_time})")

        if best_frame_path:
            image = Image.open(best_frame_path)
            st.image(image, caption=f"Keyframe (migliore tra 25/50/75%)", use_container_width=True)
        else:
            st.text("⚠️ Nessun frame con testo rilevato")

        if best_text:
            st.text_area("Testo rilevato:", best_text, height=100, key=f"text_{i}")
        else:
            st.text("⚠️ OCR fallito su tutti i frame")

        if st.checkbox(f"Seleziona questa scena", key=f"scene_{i}"):
            selected_scenes.append((start_time, end_time))

    total_end_time = time.perf_counter()

    with st.sidebar:
        st.write(f"⏱️ Tempo totale analisi: {total_end_time - total_start_time:.2f} secondi")
        st.write(f"🧠 Tempo totale dedicato all'OCR: {ocr_total_time:.2f} secondi")


    # Esportazione
    if selected_scenes:
        st.write("## Esporta le scene selezionate")
        output_path = os.path.join(tempfile.gettempdir(), "output_segments.mp4")
        with open("segments.txt", "w") as f:
            for i, (start, end) in enumerate(selected_scenes):
                f.write(f"file '{tmp_path}'\n")
                f.write(f"inpoint {start.get_seconds()}\n")
                f.write(f"outpoint {end.get_seconds()}\n")

        st.warning("L'esportazione dei segmenti è abbozzata: serve una logica ffmpeg per creare video multipli o concatenati.")
