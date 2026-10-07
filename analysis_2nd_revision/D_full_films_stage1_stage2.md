# Section D - full-length films: Stage I and Stage II

**Clips**: hand-cut credit clips of the paper (Opening + End summed where there are two), Stage I/II of 2025-12-04 (commit 00f0cbc5, PaddleOCR `lang=en`); their frames are the ones Stage III used in July 2026. **Films**: the 20 full-length files (`runs/FULL_pipeline`, commit f9c9bb6, `lang=en`), Stage I with "Run on the whole episode" and no manual deselection of candidate scenes, Stage II with cap 150. Stage III/IV were run on the films afterwards (4,628 calls).

Shots = scenes found by scene detection; candidate scenes = scenes kept by Stage I (text found by OCR); frames = frames selected by Stage II (= Stage III calls). Film Stage I/II hours are machine time (Stage I partly ran 3 films in parallel on one GPU).

| Title | Clips | Clip min | Film min | Clip shots | Film shots | Clip cand. scenes | Film cand. scenes | Clip cand. min | Film cand. min | Clip frames | Film frames | Frames film/clip | Film Stage I h | Film Stage II h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8_e_mezzo | 1 | 3.7 | 139.1 | 10 | 159 | 8 | 25 | 3.3 | 38.5 | 36 | 113 | 3.1 | 1.0 | 2.4 |
| Amelie | 1 | 7.8 | 121.6 | 96 | 1,494 | 65 | 363 | 6.2 | 38.1 | 103 | 540 | 5.2 | 2.5 | 2.0 |
| Apocalypse_Now | 1 | 5.9 | 152.9 | 1 | 1,305 | 1 | 220 | 5.9 | 38.2 | 66 | 496 | 7.5 | 4.6 | 3.6 |
| Chernobyl_S01E01 | 2 | 3.0 | 56.3 | 7 | 728 | 7 | 71 | 3.0 | 8.0 | 79 | 141 | 1.8 | 3.5 | 0.9 |
| Dark_1x9 | 1 | 4.9 | 55.3 | 48 | 703 | 38 | 93 | 4.6 | 8.6 | 51 | 103 | 2.0 | 0.5 | 0.1 |
| El_desorden_que_dejas | 2 | 6.8 | 41.3 | 66 | 508 | 18 | 90 | 4.0 | 11.2 | 71 | 142 | 2.0 | 0.9 | 0.6 |
| Eternal_Sunshine_of_the_Spotless_Mind | 1 | 5.6 | 107.9 | 30 | 1,529 | 28 | 195 | 5.6 | 17.0 | 80 | 189 | 2.4 | 1.9 | 0.9 |
| Fight_Club | 1 | 4.9 | 139.1 | 39 | 3,112 | 39 | 607 | 4.9 | 36.2 | 85 | 673 | 7.9 | 11.3 | 3.5 |
| Hill_Street_Blues_1x13 | 1 | 3.2 | 48.1 | 43 | 525 | 29 | 72 | 2.1 | 8.7 | 44 | 111 | 2.5 | 0.4 | 0.1 |
| La_grande_bellezza | 1 | 9.4 | 141.5 | 20 | 1,833 | 8 | 104 | 8.9 | 19.1 | 103 | 213 | 2.1 | 1.4 | 0.9 |
| La_piovra_1x2 | 1 | 2.1 | 58.4 | 13 | 264 | 5 | 27 | 1.9 | 6.5 | 31 | 60 | 1.9 | 0.2 | 0.1 |
| Maigret_S03E01 | 2 | 4.9 | 95.9 | 39 | 565 | 38 | 85 | 4.9 | 14.6 | 83 | 171 | 2.1 | 1.7 | 1.0 |
| Planet_Earth_S01E10 | 1 | 0.8 | 58.9 | 4 | 724 | 3 | 23 | 0.8 | 3.5 | 11 | 36 | 3.3 | 4.5 | 0.6 |
| Prime_Suspect_1x1 | 1 | 3.4 | 108.2 | 15 | 739 | 15 | 121 | 3.4 | 27.5 | 35 | 211 | 6.0 | 1.7 | 1.7 |
| Psycho | 1 | 1.8 | 108.9 | 22 | 81 | 14 | 33 | 1.7 | 70.0 | 33 | 367 | 11.1 | 0.1 | 1.9 |
| Romanzo_criminale_S01E01 | 2 | 4.8 | 62.3 | 76 | 1,534 | 58 | 238 | 4.4 | 12.4 | 83 | 213 | 2.6 | 3.5 | 0.9 |
| Se7en | 1 | 7.2 | 126.8 | 141 | 2,128 | 124 | 360 | 7.0 | 26.2 | 150 | 430 | 2.9 | 2.8 | 1.1 |
| The_World_At_War_S01E03 | 2 | 2.2 | 54.8 | 8 | 244 | 4 | 47 | 2.2 | 19.5 | 32 | 122 | 3.8 | 1.4 | 1.4 |
| Twin_Peaks_1x3 | 1 | 3.7 | 48.3 | 19 | 539 | 10 | 124 | 3.3 | 13.0 | 62 | 240 | 3.9 | 1.2 | 1.3 |
| Yes,_Prime_Minister_1x8 | 1 | 1.5 | 29.5 | 8 | 328 | 4 | 32 | 1.4 | 4.7 | 16 | 57 | 3.6 | 0.1 | 0.1 |
| **Total** | 25 | 87.6 | 1,755.1 | 705 | 19,042 | 516 | 2,930 | 79.5 | 421.5 | 1,254 | 4,628 | 3.7 | 45.3 | 25.2 |

## Notes

- Film Stage I/II logs were checked for OCR failures (`PaddleOCR failed`): 0 in the final logs of all 20 films.
- La_grande_bellezza_FULL: Stage I and II redone on 2026-10-03 - Stage I OCR failed from scene 170 (CUDA out of memory with 3 parallel Stage I workers): 31177 OCR calls failed, end credits never OCRed. The failed attempt's logs are kept as `*.failed_<timestamp>.log`.
- Eternal_Sunshine_of_the_Spotless_Mind_FULL, Yes,_Prime_Minister_1x8_FULL: wrong full-length file supplied; replaced by the correct file on 2026-10-06, Stage I and II redone (old and new MD5 in `run_info.json`, `replaced_inputs`); everything Stage I/II had produced from the wrong files was deleted before the rerun.
- Stage II keeps every frame of a candidate scene in RAM; the longest film candidate scenes needed up to ~38 GB (Apocalypse Now) and ran through the page file.
- Segment boundaries vs the annotated credit boundaries (precision/recall/IoU) need the annotations; the film candidate scenes are in `runs/FULL_pipeline/data/episodes/<film>/analysis/initial_scene_analysis.json` and the selected frames with timecodes in `runs/FULL_pipeline/run_output/stage2_frames_timecodes.csv`.

## Clip frames found among the film frames

Are the frames the paper's Stage III received from the hand-cut clips also selected by Stage II on the full films? Clips and films are different encodes (frame numbers, resolution, letterbox), so frames are matched by content (`section_D_frame_overlap.py`): (1) perceptual hash (pHash, 64 bits) after cropping black borders, closest film frame of the same title; (2) share of the clip frame's Stage II OCR words found among the film frames' OCR words, which also covers credit rolls sampled a few frames apart. A clip frame is *likely missing* when it has no visual match within distance 16 and less than 50% of its OCR words are found.

Of 1,254 clip frames, 50.6% have a visual match (pHash <= 10; 65.0% within 16) and 53 (4.2%) were flagged as likely missing.

**Manual check of the flagged frames** (each looked at side by side with the closest film frame by pHash and by shared OCR words, and the credit's names searched, also fuzzily, in the OCR text of every film frame; images in `results/D_likely_missing/`, kept locally only, not in the repository):

- **13 present** in the film frames (v) and **1 present but badly extracted** (b: the film frame was taken during a cross-fade). Most were false alarms of the clip's own OCR (handwriting, noisy reading), e.g. *Brad Pitt*, the cast and screenplay cards of *Romanzo criminale*, *Narrated by Laurence Olivier*.
- **26 clip frames show no credit at all** (n): the NAIJAPREY watermark of the Eternal Sunshine clip encode, props in Se7en, the Psycho title animation mid-transition, the Sky Atlantic ident. They are Stage III calls the clip pipeline spent on non-credit frames.
- **13 missing** (x): 8 person credits; 4 company logos; 1 titles (series / episode).

**After the check, 1,241 of 1,254 clip frames (99.0%) are present among the full-film frames.** The person credits truly lost on the films are concentrated in *Prime Suspect* (part of the end roll, Director of Photography, Producer) plus two single cards (*Amelie*: editor; *Fight Club*: the opening *Edward Norton* card, whose name is still in the end-credit cast list).

### Missing clip frames

| n | Title | Type | Why |
|---:|---|---|---|
| 3 | Amelie | persona | opening card 'Montage Herve Schneid' not among the film frames (the next card 'Montage son' is) |
| 6 | El_desorden_que_dejas | titolo | series title card 'El desorden que dejas' not among the film frames |
| 7 | Fight_Club | logo | Regency logo not in the full film opening |
| 8 | Fight_Club | logo | Regency logo not in the full film opening |
| 9 | Fight_Club | logo | Regency logo not in the full film opening |
| 10 | Fight_Club | logo | 'Fox 2000 Pictures and Regency Enterprises present' card not among the film frames |
| 12 | Fight_Club | persona | opening card 'Edward Norton' not among the film frames; the name is in the end-credit cast list |
| 17 | Prime_Suspect_1x1 | persona | end-roll cast names (Rod Arthur, Rosy Clayton, Julian Firth, Ian Hastings, Moyra Ruskin...) not found in the film frames' OCR; neighbouring roll lines are |
| 18 | Prime_Suspect_1x1 | persona | end-roll cast names (Rod Arthur, Rosy Clayton, Julian Firth, Ian Hastings, Moyra Ruskin...) not found in the film frames' OCR; neighbouring roll lines are |
| 19 | Prime_Suspect_1x1 | persona | end-roll cast names (Rod Arthur, Rosy Clayton, Julian Firth, Ian Hastings, Moyra Ruskin...) not found in the film frames' OCR; neighbouring roll lines are |
| 20 | Prime_Suspect_1x1 | persona | end-roll cast names (Rod Arthur, Rosy Clayton, Julian Firth, Ian Hastings, Moyra Ruskin...) not found in the film frames' OCR; neighbouring roll lines are |
| 21 | Prime_Suspect_1x1 | persona | 'Director of Photography Ken Morgan' not found in the film frames |
| 22 | Prime_Suspect_1x1 | persona | 'Producer Don Leaver' not found in the film frames |

### Per title

| Title | Clip frames | Film frames | Visual match % (pHash <= 10) | Visual match % (<= 16) | Mean OCR-word coverage % | Likely missing | Missing after check | Present % after check |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8_e_mezzo | 36 | 113 | 88.9 | 91.7 | 85.6 | 1 | 0 | 100.0 |
| Amelie | 103 | 540 | 28.2 | 41.7 | 89.3 | 3 | 1 | 99.0 |
| Apocalypse_Now | 66 | 496 | 4.5 | 56.1 | 95.5 | 0 | 0 | 100.0 |
| Chernobyl_S01E01 | 79 | 141 | 89.9 | 91.1 | 98.5 | 0 | 0 | 100.0 |
| Dark_1x9 | 51 | 103 | 47.1 | 70.6 | 95.8 | 0 | 0 | 100.0 |
| El_desorden_que_dejas | 71 | 142 | 63.4 | 77.5 | 93.8 | 2 | 1 | 98.6 |
| Eternal_Sunshine_of_the_Spotless_Mind | 80 | 189 | 46.2 | 57.5 | 93.9 | 0 | 0 | 100.0 |
| Fight_Club | 85 | 673 | 3.5 | 20.0 | 71.1 | 7 | 5 | 94.1 |
| Hill_Street_Blues_1x13 | 44 | 111 | 90.9 | 95.5 | 94.4 | 0 | 0 | 100.0 |
| La_grande_bellezza | 103 | 213 | 67.0 | 87.4 | 93.5 | 0 | 0 | 100.0 |
| La_piovra_1x2 | 31 | 60 | 93.5 | 93.5 | 100.0 | 0 | 0 | 100.0 |
| Maigret_S03E01 | 83 | 171 | 86.7 | 95.2 | 93.1 | 0 | 0 | 100.0 |
| Planet_Earth_S01E10 | 11 | 36 | 18.2 | 36.4 | 96.8 | 0 | 0 | 100.0 |
| Prime_Suspect_1x1 | 35 | 211 | 25.7 | 31.4 | 68.4 | 9 | 6 | 82.9 |
| Psycho | 33 | 367 | 12.1 | 51.5 | 74.2 | 3 | 0 | 100.0 |
| Romanzo_criminale_S01E01 | 83 | 213 | 44.6 | 53.0 | 88.8 | 5 | 0 | 100.0 |
| Se7en | 150 | 430 | 28.7 | 42.0 | 77.4 | 19 | 0 | 100.0 |
| The_World_At_War_S01E03 | 32 | 122 | 75.0 | 84.4 | 59.7 | 2 | 0 | 100.0 |
| Twin_Peaks_1x3 | 62 | 240 | 75.8 | 88.7 | 84.6 | 1 | 0 | 100.0 |
| Yes,_Prime_Minister_1x8 | 16 | 57 | 93.8 | 93.8 | 87.1 | 1 | 0 | 100.0 |

Lists: `results/D_review_present_in_film.csv`, `results/D_review_missing_in_film.csv`, `results/D_review_no_credit_in_clip.csv`; full review with the verdicts in `results/D_likely_missing_review.csv` (edit `controllo` and rerun `section_D_frame_overlap.py` then this script to update the numbers). Limit: a credit marked missing could still sit in a film frame whose OCR is completely unreadable. This compares frames; the credits themselves are compared with the gold set in the next section.

## Credits extracted from the full films vs the gold set

Stage III (GPT Sol Standard) and Stage IV (fuzzy 88) on the 4,628 frames Stage II selected on the full films (213.68 USD), exact match against the human gold set of the 20 products, compared with the hand-cut clips: the first-revision run and the rerun of the clips with the same code as the films (f9c9bb6, OCR en, cap 150). `cost_usd_uncached` prices every input token at full rate (July read most input from the cache).

### No Role

| label | mode | products | TP | FP | FN | precision | recall | F1 | calls | cost_usd | cost_usd_uncached |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Clips, first revision (July) | No Role | 20 | 4563 | 198 | 302 | 0.9584 | 0.9379 | 0.9481 | 1254 | 29.23 | 65.09 |
| Clips, rerun (f9c9bb6) | No Role | 20 | 4624 | 184 | 241 | 0.9617 | 0.9505 | 0.9561 | 1262 | 65.30 | 65.30 |
| Full films (f9c9bb6) | No Role | 20 | 4588 | 274 | 277 | 0.9436 | 0.9431 | 0.9434 | 4628 | 213.68 | 214.22 |

### With Role

| label | mode | products | TP | FP | FN | precision | recall | F1 | calls | cost_usd | cost_usd_uncached |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Clips, first revision (July) | With Role | 20 | 4437 | 423 | 519 | 0.9130 | 0.8953 | 0.9040 | 1254 | 29.23 | 65.09 |
| Clips, rerun (f9c9bb6) | With Role | 20 | 4497 | 420 | 459 | 0.9146 | 0.9074 | 0.9110 | 1262 | 65.30 | 65.30 |
| Full films (f9c9bb6) | With Role | 20 | 4478 | 502 | 478 | 0.8992 | 0.9035 | 0.9014 | 4628 | 213.68 | 214.22 |

### Per product (No Role): clips rerun vs full films

| product | clips recall | films recall | clips FP | films FP | clips F1 | films F1 |
|---|---|---|---|---|---|---|
| 8 e mezzo | 0.9718 | 0.9437 | 1 | 3 | 0.9787 | 0.9504 |
| amelie | 0.9306 | 0.9167 | 31 | 30 | 0.9229 | 0.9167 |
| apocalypse now | 0.9361 | 0.9393 | 28 | 18 | 0.9243 | 0.9408 |
| chernobyl s01e01 | 0.9964 | 0.9909 | 1 | 5 | 0.9973 | 0.9909 |
| dark s01e09 | 0.8917 | 0.8339 | 17 | 51 | 0.9277 | 0.8660 |
| el desorden que dejas s01e03 | 0.9551 | 0.9633 | 6 | 5 | 0.9710 | 0.9762 |
| eternal sunshine of the spotless mind | 0.9650 | 0.9679 | 13 | 13 | 0.9636 | 0.9651 |
| fight club | 0.9449 | 0.9633 | 8 | 3 | 0.9613 | 0.9774 |
| hill street blues s01e13 | 0.9859 | 0.9718 | 2 | 2 | 0.9790 | 0.9718 |
| la grande bellezza | 0.9293 | 0.9293 | 31 | 58 | 0.9313 | 0.9051 |
| la piovra s01e02 | 1.0000 | 1.0000 | 1 | 1 | 0.9831 | 0.9831 |
| maigret s03e01 | 0.9574 | 0.9681 | 3 | 1 | 0.9626 | 0.9785 |
| planet earth s01e10 | 0.9828 | 0.9655 | 0 | 2 | 0.9913 | 0.9655 |
| prime suspect s01e01 | 0.9615 | 0.7051 | 2 | 28 | 0.9677 | 0.6832 |
| psycho | 0.9688 | 1.0000 | 2 | 5 | 0.9538 | 0.9275 |
| romanzo criminale s01e01 | 0.9436 | 0.9763 | 12 | 15 | 0.9535 | 0.9662 |
| se7en | 0.9784 | 0.9784 | 18 | 20 | 0.9717 | 0.9698 |
| the world at war s01e03 | 0.9231 | 0.9231 | 4 | 12 | 0.8276 | 0.6486 |
| twin peaks s01e03 | 0.9773 | 0.9886 | 1 | 0 | 0.9829 | 0.9943 |
| yes prime minister s01e08 | 0.9615 | 1.0000 | 3 | 2 | 0.9259 | 0.9630 |

False positives on the films are names the model read outside the credits (signs, documents, newspapers and other on-screen text in the scenes Stage I kept as candidates) or credits absent from the gold set.
