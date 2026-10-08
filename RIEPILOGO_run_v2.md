---
title: "Filmocredit — Run da eseguire, passo per passo"
subtitle: "v2 — 29 settembre 2026 — per Roberto (sostituisce la v1)"
lang: it
---

## 0. Prima di tutto, una volta sola

1. Il commit giusto è f9c9bb6.
2. Controlla che lo snapshot di GPT-5.6 Sol Standard usato a luglio (9 luglio 2026) sia ancora disponibile su Azure, con lo stesso deployment e lo stesso reasoning effort. Se non lo è, fermati e avvisami.
3. Da qui in poi **ogni run è una cartella di lavoro separata** (`git worktree add runs/<ID> <hash>`), con il suo `data/`, il suo `db/` vuoto e il suo log. Nessuna run riprende da un'altra o la sovrascrive.
4. In ogni run, alla fine, conserva: export CSV; database (con le risposte grezze); JSON per fotogramma prima della deduplicazione; log con timestamp, token in ingresso e uscita e costo per chiamata; `run_info.json` con hash, parametri, lista stopword usata ed elenco dei fotogrammi sottoposti al modello.

5. Cache del prompt: in **tutte** le run (A, B, C, D) nessuna modifica al codice sulla cache; resta il comportamento predefinito di Azure, così tutto gira su `f9c9bb6`. Da settembre il default Azure è `prompt_cache_retention = 24h` (a luglio era `in_memory`) e le letture dalla cache sono quasi nulle: il costo per chiamata sale a circa 0,041–0,046 USD. In `run_info.json` si registrano la `prompt_cache_retention` effettiva e i token letti e scritti in cache.

Lista stopword: in **tutti** i blocchi (A, B, C, D) quella di luglio, cioè il file `user_ocr_stopwords.txt` attuale (34 voci, **senza** NETFLIX).

OCR dello Stage I e II: PaddleOCR con `lang = en`, come negli Stage II di riferimento (dicembre 2025 e luglio 2026); `it` carica un altro modello e seleziona altri fotogrammi (vedi `revision_checks/`).

Costo per chiamata stimato su luglio: circa 0,027 USD.

Il modello da usare è sempre gpt sol 5.6 con reasoning effort standard, come a luglio. Non cambiare il modello o il reasoning effort.

---

## A. Curva degli intervalli — 2 run

**A cosa serve.** Il revisore dice che «cost-optimized» non è dimostrato perché abbiamo provato un solo naive (0,8 s). Facciamo il naive a due intervalli più radi (2,4 s e 4,8 s), per vedere se un campionamento uniforme meno fitto costa meno della pipeline e rende altrettanto.

**Corpus.** I 20 titoli di selezione, cioè queste 25 cartelle: `8_e_mezzo`, `Amelie`, `Apocalypse_Now`, `Chernobyl_S01E01_End`, `Chernobyl_S01E01_Opening`, `Dark_1x9`, `El_desorden_que_dejas_S01E03_End`, `El_desorden_que_dejas_S01E03_Opening`, `Eternal_Sunshine_of_the_Spotless_Mind`, `Fight_Club`, `Hill_Street_Blues_1x13`, `La_grande_bellezza`, `La_piovra_1x2`, `Maigret_S03E01_End`, `Maigret_S03E01_Opening`, `Planet_Earth_S01E10`, `Prime_Suspect_1x1`, `Psycho`, `Romanzo_criminale_S01E01_End`, `Romanzo_criminale_S01E01_Opening`, `Se7en`, `The_World_At_War_S01E03_End`, `The_World_At_War_S01E03_Opening`, `Twin_Peaks_1x3`, `Yes,_Prime_Minister_1x8`.

**Passi, per ciascuna delle 2 run:**

1. Crea il worktree `runs/NAIVE_<intervallo>`.
2. Per ogni cartella, copia in `data/episodes/<cartella>/naive_analysis/frames/` **solo** i fotogrammi naive di luglio il cui numero di sequenza (il `XXXXX` di `naive_XXXXX_numYYYYYY.jpg`) è divisibile per *k*: con *k* = 3 tieni `00000, 00003, 00006…`; con *k* = 6 tieni `00000, 00006, 00012…`. Nomi dei file invariati. Il conteggio riparte da zero in ogni cartella.
3. Stage III in modalità naive (`naive_mode=True`), provider GPT-5.6 Sol Standard, prompting incrementale come a luglio: i crediti JSON del fotogramma precedente vanno nel prompt come contesto testuale; l'immagine del fotogramma precedente **non** viene inviata.
4. Stage IV (IMDbizzazione) a soglia 88, come a luglio, così l'export ha lo stesso formato.
5. Export CSV con nome `NAIVE_<intervallo>_FUZZY88_GPT_SOL_STANDARD_20products.csv`.

| ID | *k* | Intervallo | Fotogrammi = chiamate | Costo |
|---|---:|---|---:|---:|
| `NAIVE_2.4s` | 3 | 2,4 s | 2.201 | ~59 USD |
| `NAIVE_4.0s` | 5 | 4,0 s | 1.328 | ~36 USD (reale 66,64 USD, aggiunta il 7/10/2026) |
| `NAIVE_4.8s` | 6 | 4,8 s | 1.106 | ~30 USD |

Controllo: il numero di fotogrammi sottoposti di ogni run deve coincidere con la colonna «Fotogrammi» (conteggio cartella per cartella, partendo da 0). Il punto a 0,8 s è la run di luglio e non si rifà. Le due run possono girare in parallelo: lo Stage III di ciascuna procede in contemporanea, lo Stage IV parte una run alla volta.

---

## B. Cap sui rulli — 3 run

**A cosa serve.** Il revisore dice che anche il cap di 150 non è stato esplorato. Il cap è l'intervallo massimo fra due catture sui rulli (`SCROLL_MAX_FRAMES_PER_SAVE`, `config.py`, r. 63). Abbassandolo si catturano più fotogrammi nei rulli e si vede se la recall sale abbastanza da giustificare le chiamate in più.

**Corpus.** Solo i 14 titoli con rulli, cioè queste 17 cartelle (963 fotogrammi sottoposti a luglio): `Amelie`, `Dark_1x9`, `El_desorden_que_dejas_S01E03_End`, `El_desorden_que_dejas_S01E03_Opening`, `Eternal_Sunshine_of_the_Spotless_Mind`, `Fight_Club`, `Hill_Street_Blues_1x13`, `La_grande_bellezza`, `La_piovra_1x2`, `Maigret_S03E01_End`, `Maigret_S03E01_Opening`, `Planet_Earth_S01E10`, `Prime_Suspect_1x1`, `Psycho`, `Romanzo_criminale_S01E01_End`, `Romanzo_criminale_S01E01_Opening`, `Se7en`. Gli altri sei titoli non hanno rulli e darebbero lo stesso risultato.

**Passi, per ciascuna delle 3 run:**

1. Crea il worktree `runs/PIPE_cap<valore>`.
2. In `scripts_v3/config.py` del worktree imposta `SCROLL_MAX_FRAMES_PER_SAVE  = <valore>`. È l'unica modifica al codice, e va scritta in `run_info.json`.
3. Metti in `data/raw/` le clip di luglio delle 17 cartelle (collegamenti ai file di `corpus20_cut/`).
4. **Non rifare lo Stage I**: copia in ogni `data/episodes/<cartella>/analysis/` i file di luglio `raw_scenes_cache.json` e `initial_scene_analysis.json`. Non copiare `frames/` né `analysis_manifest.json`, che vanno rigenerati.
5. Stage II (selezione dei fotogrammi) con il cap impostato.
6. Stage III con GPT-5.6 Sol Standard, come a luglio.
7. Stage IV a soglia 88.
8. Export CSV `PIPE_cap<valore>_FUZZY88_GPT_SOL_STANDARD_14products.csv`.

| ID | Cap | Chiamate stimate | Costo | Nota |
|---|---:|---:|---:|---|
| `PIPE_cap150` | 150 | 963 | ~26 USD | riferimento: stesso cap di luglio ma codice attuale |
| `PIPE_cap100` | 100 | ~1.130 | ~31 USD | |
| `PIPE_cap75` | 75 | ~1.200 | ~32 USD | |

Controllo per `PIPE_cap150`: i fotogrammi selezionati devono essere gli stessi di luglio (963), perché cambia solo il codice di Stage III (la regola su «M.»), non la selezione. Se non coincidono, fermati e avvisami: vorrebbe dire che lo Stage II non è deterministico o che qualcosa nel caricamento è cambiato.

`PIPE_cap220` solo se lo decidiamo **prima** di vedere questi risultati (~25 USD). Alzare il cap toglie al massimo 12–20 catture su 1.864, e possiamo riportarlo calcolato.

---

## C. Ripetizioni sull'hold-out — 5 run Sol (+ 5 Gemma)

**A cosa serve.** Il revisore dice che ogni configurazione è stata eseguita una volta sola e che il modello non è deterministico: non sappiamo quanto il risultato cambierebbe rifacendo la stessa run. Rifacciamo **la stessa identica cosa** più volte, sugli stessi fotogrammi, e misuriamo la dispersione. Usiamo i cinque hold-out perché sono stati scelti prima di ogni misura, come chiede il revisore.

**Corpus.** Le 8 cartelle dei cinque hold-out (207 fotogrammi trattenuti a luglio): `3_Percent_S01E06_End`, `3_Percent_S01E06_Opening`, `Blue_Eye_Samurai_S01E01_End`, `Honeyland_End`, `Persepolis_End`, `Persepolis_Opening`, `Wild_Strawberries_End`, `Wild_Strawberries_Opening`.

**Passi, per ciascuna ripetizione:**

1. Crea il worktree `runs/REP_SOL_r<n>` (n = 1…5).
2. Copia in ogni `data/episodes/<cartella>/analysis/` la cartella `frames/` e `analysis_manifest.json` di luglio. **Non rifare** Stage I e II: i fotogrammi devono essere gli stessi 207 in tutte le ripetizioni.
3. Stage III con GPT-5.6 Sol Standard, come a luglio.
4. Stage IV a soglia 88.
5. Export CSV `REP_SOL_r<n>_FUZZY88_GPT_SOL_STANDARD_5products.csv`.

Costo: 207 chiamate per ripetizione, ~6 USD; cinque ripetizioni ~28 USD.

Stesso procedimento con Gemma 4 12B in locale (`REP_GEMMA_r1…r5`), costo zero. Se il tempo stringe, tre ripetizioni per modello bastano.

---

## D. Film interi (end-to-end) — 1 run

**A cosa serve.** Il revisore dice che abbiamo sempre dato alla pipeline i titoli già tagliati a mano, e che quindi non sappiamo se sappia trovarli in un film intero. Facciamo girare la pipeline **completa** (Stage I, II, III e IV) sui film interi. Poi confrontiamo i segmenti trovati dallo Stage I con i confini annotati da Greta e Gabriele (precision, recall, IoU) e i crediti estratti con il gold (F1), come in Tabella 3.

**Passi:**

1. **Passo 0**: controlla i venti file interi per sottotitoli impressi (apri il file, guarda due o tre scene di dialogo). Mandami una tabella: titolo, nome del file, durata, sottotitoli impressi sì/no. Chi ha sottotitoli impressi è escluso. Nessuno annota confini prima di questo.
2. Crea il worktree `runs/FULL_pipeline`.
3. Metti i film interi che passano il passo 0 in `data/raw/main_corpus_full/` con identificativo `<Titolo>_FULL` (per esempio `Se7en_FULL.mp4`), così nulla finisce nelle cartelle di luglio.
4. Lista stopword: quella di luglio (34 voci, senza NETFLIX).
5. Stage I con modalità di selezione «Run on the whole episode», **senza alcuna deselezione manuale** delle scene candidate.
6. Stage II con il cap di luglio (150).
7. Stage III con GPT-5.6 Sol Standard su tutti i fotogrammi selezionati.
8. Stage IV a soglia 88.
9. Export CSV `FULL_FUZZY88_GPT_SOL_STANDARD_20products.csv`, più per ogni titolo `raw_scenes_cache.json`, `initial_scene_analysis.json` e l'elenco dei fotogrammi selezionati con il loro timecode nel film. In `run_info.json` anche l'MD5 e la durata di ogni file video e il tempo macchina per stadio.

Costo: dentro i titoli circa quanto luglio (~1.254 chiamate, ~34 USD), più i fotogrammi selezionati fuori dai titoli, che oggi non sappiamo: stima 35–60 USD in tutto.

---

## Riepilogo

| Blocco | Run | Chiamate | Costo |
|---|---|---:|---:|
| A — intervalli | 2 | ~3.300 | ~89 USD |
| B — cap | 3 (+1 facoltativa) | ~3.300 | ~89 USD |
| C — ripetizioni | 5 Sol (+3 Gemma) | ~1.000 | ~47 USD |
| D — film interi | 1 | ~1.300–2.200 | ~35–60 USD |
| **Totale** | | **~8.900–9.800** | **~241–266 USD** |

Ordine consigliato: C (piccolo e indipendente), poi A e B in parallelo, poi D appena i film interi hanno passato il passo 0. Prima di lanciare ciascun blocco, mandami la struttura di una run preparata (elenco dei file nel worktree e `run_info.json`): la controllo prima che parta qualsiasi chiamata.
