# Filmocredit — risultati delle run per la second submission

Riepilogo di tutte le run e le analisi chieste dalla seconda revisione (`RIEPILOGO_run_v2.md`, blocchi A–D),
con i numeri principali, le verifiche fatte lungo la strada e dove si trova ogni file. Ottobre 2026.

## Condizioni comuni

- **Codice**: commit `f9c9bb6` per tutte le run (ognuna in un worktree separato `runs/<ID>/`). Uniche modifiche
  dichiarate: il valore di `SCROLL_MAX_FRAMES_PER_SAVE` nelle run a cap 100 e 75 (blocco B).
- **Modello**: GPT-5.6 Sol Standard (`gpt-5.6-sol`, reasoning standard/medium), senza immagine del frame
  precedente (il contesto incrementale è il JSON dei crediti del frame precedente). Gemma 4 12B locale per le
  ripetizioni del blocco C (temperature 0).
- **Stage IV**: IMDbizzazione a soglia fuzzy 88. **Stopword**: lista di luglio (34 voci, senza NETFLIX), in tutti i blocchi.
- **OCR di Stage I e II**: PaddleOCR `lang=en`, come negli Stage II di riferimento (vedi Verifiche).
- **Cache del prompt**: nessuna modifica al codice. Da settembre Azure usa `prompt_cache_retention = 24h` e quasi
  non legge dalla cache: il costo per chiamata è circa 0,045–0,052 USD (a luglio circa 0,023). Ogni `run_info.json`
  registra la retention effettiva e i token letti/scritti in cache. Nelle tabelle `cost_usd_uncached` prezza tutti i
  token di input a tariffa piena, per confrontare luglio e settembre.
- **Valutazione**: exact match contro il gold umano (persone; "No Role" = nome, "With Role" = nome + ruolo),
  con le funzioni di `compare_llm_human_metrics.py` (importate, non copiate). Gold: 20 prodotti principali
  (`human_data_to_be_imdbized/credits_human_corrected_merged_to_be_imdbized_20_products.csv`) e 5 hold-out
  (`credits_human_corrected_to_be_imdbized_validation_5.csv`).

## Costi

| Blocco | Run | Chiamate | Costo |
|---|---|---:|---:|
| A — naive | `NAIVE_2.4s`, `NAIVE_4.8s` | 3.307 | 166,21 USD |
| B — cap | `PIPE_cap150/100/75`, `PIPE_noscroll_cap150/100/75` | 3.574 | 187,52 USD |
| C — ripetizioni Sol | `REP_SOL_r1…r5` | 1.035 | 41,63 USD |
| C — ripetizioni Gemma | `REP_GEMMA_r1…r5` | 1.035 | 0 (locale) |
| D — film interi | `FULL_pipeline` | 4.628 | 213,68 USD |
| **Totale** | | **13.579** | **≈ 609 USD** |

---

## A. Curva degli intervalli (naive)

Naive a 2,4 s e 4,8 s = sottocampionamento esatto (k = 3 e 6) dei frame naive di luglio a 0,8 s, cartella per
cartella. Confronto sui 20 prodotti con la pipeline della prima revisione e il naive a 0,8 s.

| Run | F1 nomi | Recall | F1 nome+ruolo | Chiamate | Costo a tariffa piena |
|---|---:|---:|---:|---:|---:|
| Pipeline (prima revisione) | **0,948** | 0,938 | **0,904** | 1.254 | 65 USD |
| Naive 0,8 s | 0,937 | 0,988 | 0,883 | 6.582 | 327 USD |
| Naive 2,4 s | 0,915 | 0,889 | 0,853 | 2.201 | 112 USD |
| Naive 4,8 s | 0,843 | 0,750 | 0,780 | 1.106 | 56 USD |

Diradando il naive il costo scende al livello della pipeline, ma la recall crolla (a 4,8 s: Chernobyl 0,26,
Hill Street 0,39, Twin Peaks 0,48): la pipeline resta la configurazione con il miglior rapporto qualità/costo.

## B. Cap sui rulli

Run rifatte con `f9c9bb6` su tutti i 20 prodotti: 14 titoli con rulli a cap 150/100/75; i 6 senza rulli una
volta a cap 150, tranne *Yes, Prime Minister*, dove oggi viene riconosciuto un rullo (effetto di `5fec349`) e che
quindi è stato rifatto anche a cap 100 e 75 (le altre 5 cartelle sono riprese da cap 150: il contesto incrementale
non passa da una cartella all'altra).

| Run (20 prodotti) | F1 nomi | F1 nome+ruolo | Chiamate | Costo |
|---|---:|---:|---:|---:|
| Pipeline prima revisione (cap 150) | 0,948 | 0,904 | 1.254 | 65 USD* |
| Rifatta, cap 150 | 0,956 | 0,911 | 1.262 | 65 USD |
| Rifatta, cap 100 | **0,957** | **0,917** | 1.371 | 72 USD |
| Rifatta, cap 75 | 0,954 | 0,908 | 1.491 | 79 USD |

\* a tariffa piena. Sui 15 prodotti in cui il cap conta il quadro è lo stesso (cap 100: 0,953 / 0,911). Il cap
migliore è 100 (+9% di chiamate rispetto a 150), a 75 cala la precisione. La rifatta a cap 150 supera la prima
revisione soprattutto per *Yes, Prime Minister* (recall 0,65 → 0,96, rullo ora riconosciuto) e *La grande bellezza*.

## C. Ripetizioni sull'hold-out

Stessi 207 frame (Stage II di luglio) in ogni ripetizione.

| | F1 nomi | F1 nome+ruolo | Jaccard medio tra ripetizioni (nome+ruolo) |
|---|---:|---:|---:|
| Sol, pubblicata (luglio) | 0,965 | 0,925 | — |
| Sol, 5 ripetizioni | **0,970 ± 0,003** (0,967–0,974) | **0,901 ± 0,009** (0,893–0,915) | 0,893 |
| Gemma, 5 ripetizioni | 0,727 ± 0 | 0,474 ± 0 | 1,000 (identiche) |

Non esiste una run Gemma originale sull'hold-out (quella di luglio copre solo i 20 prodotti principali).

**Perché lo 0,925 pubblicato sta sopra tutte le ripetizioni** (`results/C_holdout_check.md`):
- non è il calcolo: ricalcolato con lo stesso codice dà ancora 0,9246;
- non è il codice né il prompt: la run pubblicata (Stage III 18–23 luglio, prima di `5fec349`) aveva già
  praticamente il prompt di oggi (token in ingresso del primo frame uguali o ±8 su 10–12 mila);
- la differenza è concentrata: 16 crediti con ruolo giusto nella pubblicata e sbagliato in tutte e cinque le
  ripetizioni (3 nel verso opposto), su pochi cartelli con ruoli ambigui (*Persepolis* storyboard: Art → Animation
  Department; *Blue Eye Samurai* heads of studio: Production Managers → Additional Crew; stagisti → Additional Crew).
  Il modello legge lo stesso `role_detail` e assegna un altro `role_group`;
- la pubblicata è sotto tutte le ripetizioni sui nomi e sopra di 3,4 deviazioni standard sui ruoli; nel frattempo il
  servizio Azure dietro lo stesso deployment è stato aggiornato. Sembra un cambio di comportamento del modello più
  che una run fortunata, ma il servizio di luglio non è più interrogabile.
- Indicazione: per l'hold-out usare la media delle ripetizioni; lo 0,925 come esecuzione precedente sul servizio di luglio.

## D. Film interi (end to end)

20 film interi (`<Titolo>_FULL`, 29,3 ore di video), Stage I «Run on the whole episode» senza deselezione manuale,
Stage II a cap 150, Stage III e IV completi.

| | Clip tagliate (riferimento) | Film interi |
|---|---:|---:|
| Minuti di video | 87,6 | 1.755 |
| Inquadrature (scene detection) | 705 | 19.042 |
| Scene candidate (Stage I) | 516 | 2.930 |
| Frame selezionati (Stage II = chiamate Stage III) | 1.254 | 4.628 |
| Tempo macchina Stage I / II | — | 45,3 h / 25,2 h |

**Frame delle clip ritrovati nei film** (pHash dopo ritaglio delle bande nere + copertura delle parole OCR, poi
revisione manuale dei casi dubbi con le immagini affiancate): **99,0%** dei frame delle clip è presente tra i frame
dei film. 53 segnalati come probabilmente mancanti: 14 presenti, 26 frame di clip senza crediti (watermark, oggetti
di scena, transizioni), **13 mancanti davvero** — 8 crediti di persone (*Prime Suspect*: parte del rullo finale,
Director of Photography, Producer; *Amelie*: montatore; *Fight Club*: cartello iniziale di Edward Norton, il cui nome è
comunque nel cast dei titoli di coda), 4 loghi (*Fight Club*), 1 titolo (*El desorden que dejas*).

**Crediti estratti dai film contro il gold** (20 prodotti):

| | F1 nomi | Precision | Recall | F1 nome+ruolo | Chiamate | Costo a tariffa piena |
|---|---:|---:|---:|---:|---:|---:|
| Clip, prima revisione | 0,948 | 0,958 | 0,938 | 0,904 | 1.254 | 65 USD |
| Clip rifatte (stesso codice dei film) | 0,956 | 0,962 | 0,951 | 0,911 | 1.262 | 65 USD |
| **Film interi** | **0,943** | 0,944 | 0,943 | **0,901** | 4.628 | 214 USD |

La pipeline completa sui film interi resta vicina alle clip tagliate a mano (−1,3 punti sui nomi, −1,0 su nome+ruolo)
a circa 3,3 volte il costo. Il calo è concentrato: *Prime Suspect* (recall 0,96 → 0,71, rullo finale perso da
Stage II), falsi positivi da testo dentro le scene in *Dark*, *La grande bellezza*, *The World At War*; su altri
titoli i film vanno meglio (*Fight Club*, *Maigret*, *Romanzo criminale*, *Yes, Prime Minister*).

Note sulla run: lo Stage I di *La grande bellezza* è stato rifatto (la prima volta 3 processi in parallelo sulla GPU
avevano esaurito la memoria e 31.177 chiamate OCR erano fallite in silenzio, compresi i titoli di coda); i file
interi di *Eternal Sunshine* e *Yes, Prime Minister* erano sbagliati e sono stati sostituiti, con Stage I–IV rifatti
da zero. Il confronto dei segmenti di Stage I con i confini annotati (precision, recall, IoU) richiede le annotazioni
e non è ancora fatto: i dati sono pronti (vedi sotto).

---

## Verifiche fatte lungo la strada

- **Stage II di riferimento.** I frame «di luglio» del corpus principale vengono dallo Stage II del 4 dicembre 2025
  (commit `00f0cbc5`); quelli dell'hold-out dallo Stage II del 18–23 luglio 2026. Tra `00f0cbc5` e `f9c9bb6` lo Stage
  II cambia solo in `5fec349` (percentile del flusso scelto in base alla direzione di scorrimento).
- **Lingua dell'OCR.** Entrambi gli Stage II di riferimento sono stati fatti con PaddleOCR `lang=en`
  (`en_PP-OCRv5_mobile_rec`); `it` carica il modello `latin` e seleziona altri frame. Con `en` le scene statiche si
  riproducono esattamente (es. *El desorden* 50/50, *Blue Eye Samurai* 44/44, *Wild Strawberries* 18/18).
- **Determinismo dello Stage II** con `en`: 274 scene non a scorrimento su 275 identiche tra le tre run a cap
  diverso; una seconda esecuzione su 3 episodi riproduce la stessa selezione.
- **Hold-out con il codice di oggi**: 204 frame su 207 uguali a luglio; l'unica differenza è una scena di *Persepolis*
  riclassificata da `5fec349`.
- **Errori OCR**: dopo il caso di *La grande bellezza* il runner non segna completato un episodio se il suo log
  contiene anche un solo `PaddleOCR failed`; tutti i log finali dei 20 film ne hanno 0.

---

## Dove trovare i file

### Analisi (questa cartella, `analysis_2nd_revision/`)

| File | Contenuto |
|---|---|
| `README.md` | descrizione tecnica di dati e script, comandi per rigenerare tutto |
| `build_datasets.py` | costruisce `data/` dalle run (CSV dei crediti + provenienza e costi) |
| `metrics_lib.py` | funzioni comuni: valutazione contro il gold (da `compare_llm_human_metrics.py`), tabelle |
| `section_A_naive.py` | sezione A → `results/A_naive.md`, `A_naive.csv`, `A_naive_per_product.csv` |
| `section_B_cap.py` | sezione B → `results/B_cap.md`, `B_cap.csv`, `B_cap_per_product.csv` |
| `section_C_repetitions.py` | sezione C → `results/C_repetitions.md`, `C_repetitions.csv`, `C_dispersion.csv` |
| `section_C_holdout_check.py` | indagine sullo 0,925 → `results/C_holdout_check.md`, `C_holdout_check_credits.csv` (confronto credito per credito) |
| `section_D_frame_overlap.py` | frame clip vs film → `results/D_frame_overlap.md`, `D_frame_overlap.csv` (per titolo), `D_frame_overlap_frames.csv` (per frame), `D_likely_missing_review.csv` (revisione manuale con verdetti), `D_likely_missing/` (immagini affiancate, solo in locale: non nel repository né negli zip), `D_review_present_in_film.csv`, `D_review_missing_in_film.csv`, `D_review_no_credit_in_clip.csv` |
| `section_D_full_films.py` | report sezione D → `D_full_films_stage1_stage2.md` (Stage I/II per titolo, frame ritrovati, crediti vs gold), `results/D_full_films.csv`, `D_credits_vs_gold.csv`, `D_credits_per_product.csv` |
| `data/*.csv` | un CSV dei crediti per dataset: `ORIG_SOL_20products`, `ORIG_SOL_holdout5`, `A_NAIVE_0.8s/2.4s/4.8s`, `B_CAP150/100/75`, `C_SOL_r1…r5`, `C_GEMMA_r1…r5`, `D_FULL` |
| `data/datasets.json` | per ogni dataset: run, CSV e DB di origine, prodotti presi da ciascuno, chiamate, token, costo |

Ordine per rigenerare: `build_datasets.py`, poi le sezioni A, B, C, `section_C_holdout_check.py`,
`section_D_frame_overlap.py`, `section_D_full_films.py` (tutti dalla radice del repository).

### Verifiche sullo Stage II (`revision_checks/`)

| File | Contenuto |
|---|---|
| `stage2_ocr_language_check.py` + `_output.txt` | Stage II rifatto con `en` e `it` contro i riferimenti |
| `stage2_determinism.py` + `_output.txt` | determinismo dello Stage II (scene statiche tra le run, seconda esecuzione) |

### Run (`runs/<ID>/`, worktree su `f9c9bb6`, esclusi da git)

| Run | Blocco |
|---|---|
| `NAIVE_2.4s`, `NAIVE_4.8s` | A |
| `PIPE_cap150`, `PIPE_cap100`, `PIPE_cap75`, `PIPE_noscroll_cap150`, `PIPE_noscroll_cap100`, `PIPE_noscroll_cap75` | B |
| `STAGE2_holdout_cap150` | verifica dello Stage II sull'hold-out (senza Stage III) |
| `REP_SOL_r1…r5`, `REP_GEMMA_r1…r5` | C |
| `FULL_pipeline` | D |

In ogni run:

| File | Contenuto |
|---|---|
| `run_info.json` | commit, modifiche al codice, parametri, stopword (con sha256), frame sottoposti, cache effettiva, tempi; per D anche MD5/durata/fps dei video, rifacimenti (`redone_episodes`) e file sostituiti (`replaced_inputs`) |
| `run.log` | log completo con timestamp (token e costo per chiamata, retry, errori) |
| `run_output/<ID>_FUZZY88_….csv` | export dei crediti |
| `run_output/calls.csv` | una riga per chiamata: timestamp, frame, token (input, cache letta/scritta, output, reasoning), costo |
| `run_output/raw_responses/` | risposta grezza del modello per ogni frame (prima della deduplicazione) |
| `run_output/summary.json` | controllo «ogni frame ha la sua risposta», totali, tempi per stadio |
| `run_output/logs/` | log per episodio degli Stage I/II eseguiti in processi separati (B, D) |
| `run_output/stage2_frames_timecodes.csv` | frame selezionati con timecode nel video (B, D) |
| `db/tvcredits_v3.db` | database della run (crediti, risposte grezze, tempi) |
| `data/episodes/<episodio>/analysis/` | Stage I (`raw_scenes_cache.json`, `initial_scene_analysis.json`) e Stage II (`frames/`, `analysis_manifest.json`) |

### Script delle run (radice del repository)

| File | Contenuto |
|---|---|
| `prepare_naive_run.py` | prepara una run del blocco A (`--k`) |
| `prepare_pipe_run.py` | prepara una run del blocco B (`--cap`, `--set scroll|noscroll|ypm|holdout`) |
| `prepare_rep_run.py` | prepara una ripetizione del blocco C (`--model sol|gemma --rep n`) |
| `prepare_full_run.py` | prepara la run del blocco D |
| `run_headless.py` | esegue Stage I–IV senza GUI con le stesse funzioni dei pulsanti dell'interfaccia; lock per gli stadi pesanti, processi separati per episodio, ripresa dopo interruzione, export e riepilogo |
| `RIEPILOGO_run_v2.md` | il piano delle run, aggiornato con le decisioni prese |
