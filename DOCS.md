# FilmOCredit Pipeline v3 - Documentazione Completa

## Panoramica

FilmOCredit è un sistema automatizzato per l'estrazione di crediti cinematografici e televisivi da file video. Il sistema utilizza tecniche avanzate di computer vision, OCR e intelligenza artificiale per identificare, analizzare ed estrarre automaticamente i crediti dai video.

## Architettura del Sistema

### Struttura Directory
```
filmocredit/
├── app.py                          # Interfaccia Streamlit principale
├── requirements.txt                # Dipendenze Python
├── how_to_install.md              # Istruzioni installazione
├── user_ocr_stopwords.txt         # Parole da escludere (loghi/watermark)
├── filmocredit_pipeline.log       # File di log
├── data/
│   ├── raw/                       # Video originali
│   ├── episodes/                  # Dati elaborati per episodio
│   │   └── [EPISODE_ID]/
│   │       ├── analysis/          # Analisi scene e frame
│   │       │   ├── analysis_manifest.json
│   │       │   ├── initial_scene_analysis.json
│   │       │   ├── raw_scenes_cache.json
│   │       │   ├── frames/        # Frame selezionati per OCR
│   │       │   ├── skipped_frames/ # Frame scartati
│   │       │   └── step1_representative_frames/
│   │       └── ocr/               # Risultati OCR
│   │           └── [EPISODE_ID]_credits_azure_vlm.json
│   └── processed/                 # Output finale
├── db/
│   └── tvcredits_v3.db           # Database SQLite
└── scripts_v3/                   # Moduli core
    ├── config.py                 # Configurazioni
    ├── scene_detection.py        # Rilevamento scene
    ├── frame_analysis.py         # Analisi frame
    ├── azure_vlm_processing.py   # OCR Azure AI
    ├── utils.py                  # Utilità
    ├── exceptions.py             # Eccezioni custom
    └── mapping_ruoli.json        # Mappatura ruoli
```

## Workflow Completo del Pipeline

### Step 1: Identificazione Scene Candidato

**Obiettivo**: Identificare le parti del video che potrebbero contenere crediti

**Processo**:
1. **Apertura Video**: Caricamento del file video usando `scenedetect`
2. **Selezione Segmenti**: 
   - **Modalità Scene Count**: Analizza le prime N e ultime N scene
   - **Modalità Time-based**: Analizza i primi X e ultimi Y minuti
3. **Rilevamento Scene**: Usa `ContentDetector` e `ThresholdDetector` per identificare cambi di scena
4. **Filtraggio Scene**: Applica criteri di durata minima e filtra scene troppo brevi
5. **Campionamento Frame**: Per ogni scena, estrae frame rappresentativi
6. **OCR Preliminare**: Esegue OCR su frame campione per identificare presenza di testo
7. **Classificazione**: Determina quali scene potrebbero contenere crediti basandosi su:
   - Presenza di testo
   - Densità di testo
   - Caratteristiche visuali (scorrimento, fade, etc.)

**Output**: 
- `initial_scene_analysis.json`: Lista scene candidate con metadati
- `step1_representative_frames/`: Frame rappresentativi per revisione

**Configurazioni Chiave**:
- `DEFAULT_START_SCENES_COUNT`: 100 scene iniziali
- `DEFAULT_END_SCENES_COUNT`: 100 scene finali
- `CONTENT_SCENE_DETECTOR_THRESHOLD`: 10.0
- `SCENE_MIN_LENGTH_FRAMES`: 10 frame minimi per scena

### Step 2: Analisi Frame delle Scene

**Obiettivo**: Analizzare in dettaglio i frame delle scene candidate selezionate

**Processo**:
1. **Selezione Scene**: L'utente può selezionare quali scene analizzare dall'interfaccia
2. **Modalità di Analisi**:
   - **Static Analysis**: Frame fissi a intervalli regolari
   - **Dynamic Analysis**: Rilevamento scroll automatico usando optical flow
   - **Single Frame**: Analisi frame singolo
3. **Elaborazione Frame**:
   - **Preprocessing**: Ridimensionamento, normalizzazione
   - **OCR**: Estrazione testo usando PaddleOCR
   - **Deduplicazione**: Rimozione frame duplicati usando hash perceptivi
   - **Filtri Qualità**: Scarta frame con fade, bassa qualità, o testo insufficiente
4. **Analisi Movimento**:
   - **Optical Flow**: Rilevamento direzione scorrimento crediti
   - **Edge Detection**: Identificazione bordi per crop automatico
   - **Similarity Check**: Confronto con frame precedenti
5. **Selezione Intelligente**: Mantiene solo frame con nuovo contenuto testuale

**Output**:
- `frames/`: Frame selezionati per OCR finale
- `skipped_frames/`: Frame scartati con motivo
- `analysis_manifest.json`: Metadati processo di analisi

**Algoritmi Chiave**:
- **Hash Perceptivo**: `imagehash.average_hash()` per deduplicazione
- **Optical Flow**: Lucas-Kanade per rilevamento movimento
- **Fuzzy Matching**: Confronto testo con soglia dinamica
- **Crop Intelligente**: Rilevamento margini e crop automatico

### Step 3: OCR Azure Vision Language Model

**Obiettivo**: Estrazione finale e strutturata dei crediti usando AI avanzata

**Processo**:
1. **Preparazione**: Codifica frame in base64 per API Azure
2. **Prompt Engineering**: Costruzione prompt specifico per estrazione crediti
3. **Elaborazione Sequenziale**: 
   - Processa un frame alla volta
   - Mantiene contesto frame precedente per evitare duplicati
   - Confronta con crediti già estratti
4. **Parsing Strutturato**: Estrazione crediti in formato JSON:
   ```json
   {
     "role_detail": "Regista",
     "name": "Nome Cognome", 
     "role_group": "Directors"
   }
   ```
5. **Mappatura Ruoli**: Normalizzazione ruoli usando `mapping_ruoli.json`
6. **Salvataggio Database**: Inserimento crediti in SQLite con metadati

**Output**:
- `[EPISODE_ID]_credits_azure_vlm.json`: Crediti estratti in formato JSON
- Record nel database `tvcredits_v3.db`

**Caratteristiche Azure VLM**:
- **Modello**: GPT-4 Vision
- **Context Window**: 8192 token
- **Retry Logic**: Gestione errori con backoff esponenziale
- **Rate Limiting**: Rispetto limiti API Azure

## Componenti Tecnici

### Motori OCR

#### PaddleOCR (Principale)
- **Lingue Supportate**: Italiano, Inglese, Cinese
- **GPU/CPU**: Supporto automatico CUDA se disponibile
- **Accuratezza**: Ottimizzato per testo in movimento
- **Performance**: ~2-3 secondi per frame

#### EasyOCR (Backup)
- **Lingue**: Multilinguaggio
- **Integrazione**: Fallback automatico se PaddleOCR fallisce

### Algoritmi Computer Vision

#### Scene Detection
- **ContentDetector**: Rileva cambi basati su contenuto
- **ThresholdDetector**: Rileva cambi basati su soglia luminosità
- **Parametri**:
  - `threshold`: 10.0 per content, 5.0 per threshold
  - `min_scene_len`: 10 frame minimi

#### Frame Analysis
- **Optical Flow**: Lucas-Kanade per rilevamento movimento
- **Edge Detection**: Sobel per identificazione bordi
- **Hash Perceptivo**: Average hash 16x16 per deduplicazione
- **Fade Detection**: Analisi istogramma per fade/dissolve

#### Preprocessing
- **Resize**: Mantenimento aspect ratio
- **Crop**: Rimozione bordi automatica basata su edge detection
- **Normalize**: Normalizzazione luminosità e contrasto

### Database Schema

#### Tabella Episodes
```sql
CREATE TABLE episodes (
    episode_id TEXT PRIMARY KEY,
    series_title TEXT,
    season_number INTEGER,
    episode_number INTEGER,
    video_filename TEXT,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

#### Tabella Credits
```sql
CREATE TABLE credits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT NOT NULL,
    source_frame TEXT NOT NULL,
    role_group TEXT,
    name TEXT,
    role_detail TEXT,
    role_group_normalized TEXT,
    source_image_index INTEGER,
    scene_position TEXT,
    original_frame_number TEXT,
    reviewed_status TEXT DEFAULT 'pending',
    reviewed_at TIMESTAMP,
    FOREIGN KEY (episode_id) REFERENCES episodes (episode_id)
)
```

## Sistema di Categorizzazione Ruoli

### Gruppi Principali
- **Directors**: Registi, Co-registi
- **Writers**: Sceneggiatori, Autori
- **Cast**: Attori, Interpreti
- **Producers**: Produttori, Produttori esecutivi
- **Composers**: Compositori, Autori musiche
- **Cinematography**: Direttori fotografia
- **Production Design**: Scenografi, Costumisti
- **Sound Department**: Tecnici audio, Mixaggio
- **Music Department**: Orchestra, Cori
- **Production Management**: Organizzatori, Assistenti

### Mappatura Intelligente
Il sistema usa `mapping_ruoli.json` per normalizzare automaticamente:
- Varianti linguistiche (IT/EN)
- Sinonimi e abbreviazioni
- Ruoli specifici vs generici
- Gerarchie produttive

## Interfaccia Utente (Streamlit)

### Tab 1: Setup & Run Pipeline
- **Selezione Video**: Checkbox multipla per video batch
- **Configurazione Scene**: 
  - Scene count vs time-based
  - Margini inizio/fine personalizzabili
- **Controlli Pipeline**: Pulsanti per Step 1, 2, 3 e "Run All"
- **Revisione Scene**: Interfaccia per selezionare scene candidate
- **Preview Video**: Player integrato per anteprima

### Tab 2: Review & Edit Credits
- **Filtri**: Solo episodi con problemi vs tutti
- **Modalità Review**:
  - **Focus Mode**: Revisione uno alla volta
  - **Overview Mode**: Vista tabellare completa
- **Azioni**: Keep/Delete con motivazione
- **Navigazione**: Avanti/Indietro nella coda problematici

### Tab 3: Logs
- **Live Logging**: Visualizzazione real-time logs
- **Filtering**: Per livello e modulo
- **Export**: Salvataggio log completi

## Configurazioni Avanzate

### Parametri Scene Detection
```python
CONTENT_SCENE_DETECTOR_THRESHOLD = 10.0   # Sensibilità rilevamento
THRESH_SCENE_DETECTOR_THRESHOLD = 5       # Soglia luminosità  
SCENE_MIN_LENGTH_FRAMES = 10              # Durata minima scene
```

### Parametri Frame Analysis
```python
OPTICAL_FLOW_PARAMS = {
    'pyr_scale': 0.5,     # Scala piramide
    'levels': 3,          # Livelli piramide
    'winsize': 15,        # Dimensione finestra
    'iterations': 3,      # Iterazioni
    'poly_n': 5,         # Ordine polinomio
    'poly_sigma': 1.2    # Deviazione gaussiana
}

HASH_SIZE = 16                    # Dimensione hash perceptivo
FADE_FRAME_THRESHOLD = 20.0       # Soglia rilevamento fade
HASH_SAMPLE_INTERVAL_SECONDS = 0.5  # Intervallo campionamento
```

### Parametri OCR
```python
MIN_OCR_CONFIDENCE = 0.75         # Confidenza minima OCR
MIN_OCR_TEXT_LENGTH = 4           # Lunghezza minima testo
OCR_TIMEOUT_SECONDS = 3           # Timeout OCR per frame
FUZZY_TEXT_SIMILARITY_THRESHOLD = 60  # Soglia similarità testo
```

### Parametri Azure VLM
```python
DEFAULT_VLM_MAX_NEW_TOKENS = 8192  # Token massimi generazione
MAX_API_RETRIES = 3               # Tentativi massimi API
BACKOFF_FACTOR = 2.0              # Fattore backoff esponenziale
```

## Gestione Errori e Logging

### Sistema Logging
- **Console Handler**: Output real-time su console
- **File Handler**: Rotating logs in JSON format
- **Streamlit Handler**: Integrazione UI per monitoring live
- **Livelli**: DEBUG, INFO, WARNING, ERROR

### Gestione Eccezioni
- **ConfigError**: Errori configurazione
- **OCRError**: Fallimenti OCR
- **SceneDetectionError**: Errori rilevamento scene
- **DatabaseError**: Problemi database
- **Azure API Errors**: Rate limiting, timeout, autenticazione

### Recovery e Retry
- **Checkpoint System**: Salvataggio stato intermedio
- **Resume Capability**: Ripresa elaborazione da interruzioni
- **Graceful Degradation**: Fallback automatici
- **Progress Tracking**: Monitoraggio avanzamento per video lunghi

## Requisiti Sistema

### Hardware Raccomandato
- **CPU**: Intel i7/AMD Ryzen 7 o superiore
- **RAM**: 16GB+ (32GB per video 4K)
- **GPU**: NVIDIA RTX 3060+ con 8GB+ VRAM (opzionale ma raccomandato)
- **Storage**: SSD con 100GB+ spazio libero
- **Network**: Connessione stabile per Azure API

### Software Dependencies
- **Python**: 3.9+
- **CUDA**: 12.6+ (per GPU)
- **FFmpeg**: Per elaborazione video
- **PaddlePaddle**: 3.0.0
- **OpenCV**: 4.0+
- **Azure OpenAI**: API key richiesta

### Formati Video Supportati
- **Container**: MP4, MKV, AVI, MOV
- **Codec**: H.264, H.265, VP9, AV1
- **Risoluzione**: 480p - 4K
- **Framerate**: 24-60 FPS

## Performance e Scalabilità

### Metriche Tipiche
- **Step 1**: ~2-5 minuti per ora di video
- **Step 2**: ~10-30 minuti per episodio (dipende da scene selezionate)
- **Step 3**: ~5-15 minuti per episodio (dipende da frame estratti)
- **Throughput**: 2-4 episodi/ora su hardware raccomandato

### Ottimizzazioni
- **Parallel Processing**: Elaborazione multi-thread quando possibile
- **Intelligent Caching**: Cache OCR e hash per evitare riprocessing
- **Smart Sampling**: Riduzione frame ridondanti
- **Progressive Enhancement**: Qualità incrementale basata su risorse

### Monitoring
- **Progress Bars**: Indicatori dettagliati per ogni step
- **Memory Usage**: Monitoraggio consumo RAM
- **Error Tracking**: Cattura e reporting errori dettagliati
- **Performance Metrics**: Timing per ottimizzazione

## Manutenzione e Troubleshooting

### Problemi Comuni

#### OCR Inaccurato
- **Causa**: Qualità video bassa, testo piccolo, movimento veloce
- **Soluzione**: Regolare soglie confidence, migliorare preprocessing

#### Scene Detection Impreciso  
- **Causa**: Video con pochi cambi scena, soglie inadeguate
- **Soluzione**: Regolare threshold detector, usare modalità time-based

#### Azure API Limits
- **Causa**: Rate limiting, quota esaurita
- **Soluzione**: Implement backoff, monitorare usage, ottimizzare batch size

#### Memory Issues
- **Causa**: Video 4K, frame cache eccessivo
- **Soluzione**: Ridurre dimensioni frame, clearing cache periodico

### Backup e Recovery
- **Database Backup**: Export automatico SQLite
- **Config Backup**: Versioning configurazioni
- **Data Recovery**: Sistema checkpoint per resume elaborazione
- **Log Analysis**: Tools per debugging basato su log

## Estensioni Future

### Planned Features
- **Multi-language OCR**: Supporto linguaggi aggiuntivi
- **Batch Processing**: Elaborazione automatica directory
- **Cloud Integration**: Deploy su Azure/AWS
- **Advanced AI**: Modelli specializzati per crediti
- **Quality Assurance**: Validazione automatica crediti estratti

### API Integration
- **REST API**: Endpoint per integrazione esterna
- **Webhook Support**: Notifiche eventi processing
- **External Databases**: Connettori IMDB, TMDB
- **Export Formats**: XML, CSV, JSON-LD

## Licenze e Credits

### Dependencies
- **PaddleOCR**: Apache 2.0
- **OpenCV**: Apache 2.0  
- **Streamlit**: Apache 2.0
- **SceneDetect**: BSD 3-Clause
- **Azure OpenAI**: Proprietario Microsoft

### Codice
- **Licenza**: Da definire
- **Autori**: Team di sviluppo FilmOCredit
- **Contributi**: Benvenuti via pull request

---

*Documentazione aggiornata: Giugno 2025 - Version 3.0*
