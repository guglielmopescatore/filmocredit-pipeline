import streamlit as st
import os
import sys
import json
import re
import time
from PIL import Image
import glob

# ====== PATH SCRIPTS ======
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.batch_title_detector import processa_videos_batch

# ====== CONFIG ======
CONFIG_PATH = 'config.json'
with open(CONFIG_PATH, 'r', encoding='utf-8') as config_file:
    config = json.load(config_file)

OUTPUT_BASE_DIR   = config.get('ocr_base_dir', 'data/processed/')
RAW_VIDEO_DIR     = config.get('raw_video_dir', 'data/raw/')
SAFE_SCENE_MARGIN = config.get('safe_scene_margin', 40)
OK_SHORT_TITLES   = set(config.get('ok_short_titles', []))
OCR_LANGUAGES     = config.get('ocr_languages', ['it', 'en'])

# ====== GPU CHECK ======
try:
    import torch
    use_gpu = torch.cuda.is_available()
except ImportError:
    use_gpu = False

# ====== REGEX ======
CLEANER_REGEX       = re.compile(r"[^A-Za-zÀ-ÿ0-9]")
CAPS_REGEX          = re.compile(r"[A-Za-zÀ-ÿ]{3,}")
NUM_SHORT_REGEX     = re.compile(r"^[0-9]{1,4}$")
SINGLE_LETTER_REGEX = re.compile(r"^[A-Za-z]$")
SHORT_CAPS_REGEX    = re.compile(r"\b[A-Z]{2,4}\b")

# ====== FUNZIONI ============================================================

def carica_ocr_data(base_dir):
    """Ritorna la lista dei file ocr_results.json nel sotto‑albero `base_dir`."""
    ocr_files = []
    backup_dir = os.path.join(base_dir, 'backup')
    for root, _, files in os.walk(base_dir):
        if backup_dir and os.path.commonpath([root, backup_dir]) == backup_dir:
            continue
        for file in files:
            if file.endswith('ocr_results.json'):
                ocr_files.append(os.path.join(root, file))
    return ocr_files


def scena_significativa(text, idx, total_scenes):
    """Filtro “soft” originale (≈32 scene per film)."""
    lines = [line.strip() for line in text.splitlines() if line and len(line.strip()) >= 5]
    if not lines:
        return False
    for line in lines:
        cleaned = CLEANER_REGEX.sub('', line)
        if NUM_SHORT_REGEX.match(cleaned) or SINGLE_LETTER_REGEX.match(cleaned):
            continue
        if CAPS_REGEX.search(cleaned):
            return True
        if any(sig in OK_SHORT_TITLES for sig in SHORT_CAPS_REGEX.findall(cleaned)):
            return True
    # Titoli di testa/coda sempre ammessi
    return idx < SAFE_SCENE_MARGIN or idx >= total_scenes - SAFE_SCENE_MARGIN


def filtra_scene(ocr_data, *, show_only_significant: bool, show_only_margins: bool):
    """Applica i filtri correnti e restituisce la lista di scene da esportare."""
    # Selezione frame migliori per ogni scena
    ocr_data.sort(key=lambda x: (x['scene_idx'], x.get('frame_variant', 50)))
    best_frame_per_scene = {}
    for scene in ocr_data:
        idx = scene['scene_idx']
        var = scene.get('frame_variant', 50)
        if idx not in best_frame_per_scene or abs(var - 50) < abs(best_frame_per_scene[idx].get('frame_variant', 50) - 50):
            best_frame_per_scene[idx] = scene

    # Applica i filtri
    selected = []
    total = len(best_frame_per_scene)
    for idx, scene in best_frame_per_scene.items():
        text = scene.get('ocr_text', '')
        if show_only_margins and not (
            idx < SAFE_SCENE_MARGIN or idx >= total - SAFE_SCENE_MARGIN
        ):
            continue
        if show_only_significant and not scena_significativa(text, idx, total):
            continue
        selected.append(scene)
    return selected

# ====== UI ==================================================================

st.set_page_config(page_title="Title Viewer", layout="wide")
st.title("🎬 Selezione Titoli da Scene")

# ----- SIDEBAR CONFIGURAZIONE -----------------------------------------------
st.sidebar.header("⚙️ Configurazione")
st.sidebar.success("✅ GPU disponibile e attiva" if use_gpu else "⚡ GPU non disponibile: uso CPU")

show_only_significant = st.sidebar.checkbox("Mostra solo scene significative", value=True)
show_only_margins     = st.sidebar.checkbox("Mostra solo scene di testa/coda", value=False)

# Checkbox smart per selezionare le scene mostrate
if 'select_all_prev' not in st.session_state:
    st.session_state['select_all_prev'] = False
select_all_displayed = st.sidebar.checkbox(
    "Seleziona tutte le scene mostrate", value=st.session_state['select_all_prev']
)

# 📦 PULSANTE PER ESPORTARE **TUTTI** GLI OCR -------------------------------
if st.sidebar.button("📦 Esporta selezione per tutti gli OCR"):
    ocr_files_all = carica_ocr_data(OUTPUT_BASE_DIR)
    progress_all  = st.sidebar.progress(0, text="Processing tutti gli OCR…")
    for done, ocr_path in enumerate(ocr_files_all, 1):
        with open(ocr_path, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
        scenes_ok = filtra_scene(ocr_data,
                                 show_only_significant=show_only_significant,
                                 show_only_margins=show_only_margins)
        export_path = os.path.join(os.path.dirname(ocr_path), "selected_scenes.json")
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(scenes_ok, f, indent=2, ensure_ascii=False)
        progress_all.progress(done / len(ocr_files_all))
    st.sidebar.success(f"Esportazione completata per {len(ocr_files_all)} file OCR")

# ----- BATCH OCR -------------------------------------------------------------
progress_videos = st.sidebar.empty()
status_text = st.sidebar.empty()
if st.sidebar.button("▶️ Avvia OCR batch"):
    progress_bar = st.sidebar.progress(0)
    def update_progress(current_scene, total_scenes):
        if total_scenes > 0:
            progress_bar.progress(min(current_scene / total_scenes, 1.0))
            time.sleep(0.01)

    with st.spinner('Elaborazione batch in corso...'):
        processa_videos_batch(progress_callback=update_progress,
                              progress_video=progress_bar,
                              status_placeholder=status_text)
    st.sidebar.success("✅ Batch completato!")

# ----- SELEZIONE OCR SINGOLO -------------------------------------------------
ocr_files = carica_ocr_data(OUTPUT_BASE_DIR)
selected_ocr_file = st.sidebar.selectbox("Seleziona OCR", ocr_files)

if selected_ocr_file:
    # Player video
    video_name = os.path.basename(os.path.dirname(selected_ocr_file))
    raw_videos = glob.glob(os.path.join(RAW_VIDEO_DIR, f"{video_name}.*"))
    if raw_videos:
        st.video(raw_videos[0])
    else:
        st.warning("Video originale non trovato.")

    # Carica OCR & filtra scene
    with open(selected_ocr_file, 'r', encoding='utf-8') as f:
        ocr_data = json.load(f)
    st.sidebar.success(f"Scene caricate: {len(ocr_data)}")

    displayed_scenes = filtra_scene(ocr_data,
                                    show_only_significant=show_only_significant,
                                    show_only_margins=show_only_margins)

    # ----- GESTIONE SELECT‑ALL ----
    prev_val = st.session_state['select_all_prev']
    toggle_changed = select_all_displayed != prev_val
    if toggle_changed:
        for scene in displayed_scenes:
            idx = scene['scene_idx']
            st.session_state[f"select_{idx}"] = select_all_displayed
        st.session_state['select_all_prev'] = select_all_displayed
        (st.rerun if hasattr(st, 'rerun') else st.experimental_rerun)()

    # ----- RENDERING --------------------------------------------------------
    total_displayed = 0
    for scene in displayed_scenes:
        idx  = scene['scene_idx']
        text = scene.get('ocr_text', '')
        total_displayed += 1

        start, end = scene['scene_start_sec'], scene['scene_end_sec']
        sm, ss = divmod(int(start), 60)
        em, es = divmod(int(end), 60)
        time_info = f"{sm:02d}:{ss:02d} - {em:02d}:{es:02d}"

        col1, col2 = st.columns([2, 2])
        with col1:
            path = scene.get('frame_path')
            if path and os.path.exists(path):
                st.image(Image.open(path), caption=f"Frame scena {idx+1} ({time_info})", use_container_width=True)
            else:
                st.warning("Frame non trovato.")
        with col2:
            if idx < SAFE_SCENE_MARGIN or idx >= len(displayed_scenes) - SAFE_SCENE_MARGIN:
                st.markdown(
                    f"<div style='background-color:#fff8dc; padding:10px; border-radius:8px; font-size:16px;'>"
                    f"{text.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
            else:
                st.code(text)

            if f"select_{idx}" in st.session_state:
                st.checkbox(f"Seleziona scena {idx}", key=f"select_{idx}")
            else:
                st.checkbox(f"Seleziona scena {idx}", key=f"select_{idx}", value=False)

        st.markdown("---")

    # ----- Scene selezionate + export ---------------------------------------
    selected_scenes = [scene for scene in displayed_scenes if st.session_state.get(f"select_{scene['scene_idx']}", False)]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Scene mostrate:** {total_displayed}")
    st.sidebar.markdown(f"**Scene selezionate:** {len(selected_scenes)}")
    if total_displayed:
        perc = len(selected_scenes) / total_displayed * 100
        st.sidebar.markdown(f"**Percentuale selezionata:** {perc:.1f}%")

    if selected_scenes:
        st.sidebar.subheader("💾 Esporta selezione")
        export_name = st.sidebar.text_input("Nome file di esportazione", "selected_scenes.json")
        if st.sidebar.button("Esporta"):
            export_path = os.path.join(os.path.dirname(selected_ocr_file), export_name)
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(selected_scenes, f, indent=2, ensure_ascii=False)
            st.sidebar.success(f"File esportato in {export_path}")
