# 📼 Rilevazione automatica dei titoli & OCR  
_Fase 1 del progetto di trascrizione automatica dei sottotitoli_

Questo README descrive i tre file introdotti con questo commit — `batch_title_detector.py`, `title_viewer.py` e `config.json` — e spiega come avviarli per individuare in modo (semi-)automatico i titoli/sottotitoli presenti nei video sorgente.

## Contenuto del commit
| File | Descrizione sintetica |
|------|-----------------------|
| **`batch_title_detector.py`** | Script batch che individua le scene, seleziona frame significativi, esegue OCR multilanguage (EasyOCR) e produce un report `ocr_results.json` per ogni video. |
| **`title_viewer.py`** | App Streamlit per: (1) lanciare il batch OCR con un click, (2) visualizzare video + frame OCR, (3) filtrare scene “interessanti”, (4) permettere la selezione manuale e l’esportazione di `selected_scenes.json`. |
| **`config.json`** | File di configurazione centrale con percorsi, lingue OCR, soglia di rilevazione scene, ecc. Tutti gli script lo caricano dinamicamente. |

## Requisiti
* Python ≥ 3.9  
* GPU CUDA consigliata (il codice degrada automaticamente su CPU)  
* Librerie principali:
  ```bash
  pip install opencv-python-headless easyocr scenedetect streamlit Pillow torch
  ```
  > Usa `requirements.txt` se presente nel repo.

## Installazione rapida
```bash
git clone <repo-url>
cd <repo>
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configurazione cartelle
```
data/
├── raw/          # ↳ inserisci qui i video sorgente (.mp4/.avi/.mkv)
└── processed/    # ← output automatico (generato dagli script)
```
Puoi cambiare questi percorsi — e molti altri parametri — modificando **`config.json`**.

## Flusso di lavoro

### 1) Elaborazione batch (CLI)
```bash
python scripts/batch_title_detector.py
```
Per ogni video in `data/raw/` verrà creata la struttura:
```
data/processed/<nome_video>/
├── frames/                # frame singolo per scena
└── ocr_results.json       # OCR + metadati scena
```
Le elaborazioni precedenti vengono auto-backupate in `data/processed/backup/`.

### 2) Revisione interattiva (GUI)
```bash
streamlit run app/title_viewer.py
```
Funzionalità principali dell’interfaccia:

* **Lancio OCR batch** direttamente dal pannello laterale.  
* **Filtri dinamici**  
  * “Mostra solo scene significative” (regex + euristica)  
  * “Mostra solo scene di testa/coda” (± `safe_scene_margin` scene)  
* **Anteprima sincronizzata**: video player + frame catturato + testo OCR.  
* **Selezione rapida** con “Seleziona tutte le scene mostrate”.  
* **Esportazione** delle scene spuntate in `selected_scenes.json` (una per video o in blocco).

## Parametri chiave (`config.json`)
| Chiave | Significato |
|--------|-------------|
| `scene_detection_threshold` | Sensibilità del detector di contenuto (`scenedetect`). |
| `frame_sample_points` | Percentili di scena usati per il campionamento frame (es.: 0.25, 0.5, 0.75). |
| `ocr_languages` | Codici lingua EasyOCR, es. `["it","en"]`. |
| `rotation_angles` | Angoli extra testati durante l’OCR (per testi ruotati). |
| `safe_scene_margin` | Quante scene iniziali/finali includere sempre nei filtri. |

## Personalizzazione & tuning
* **Lingue extra** → aggiungi ISO‑639‑1 in `ocr_languages`.  
* **Prestazioni** → regola `max_workers_gpu` / `max_workers_cpu`.  
* **Backup** → disattiva rimuovendo il blocco nel codice o cambiando percorso.  

## Troubleshooting
| Problema | Possibile causa |
|----------|-----------------|
| OCR lento | GPU non rilevata → controlla installazione CUDA / `torch.cuda.is_available()`. |
| Scene “rumorose” | Abbassa `scene_detection_threshold` o amplia `frame_sample_points`. |
| Output assente | Verifica estensioni video supportate e percorsi in `config.json`. |

## Contribuire
1. Fai fork e apri branch feature.  
2. Assicurati che gli script leggano i parametri da `config.json` per evitare hard‑coding.  
3. Descrivi chiaramente le modifiche nel messaggio di commit. Pull request benvenute!  

## Licenza
Distribuito con licenza **MIT** (vedi `LICENSE`).  
