# Section D - clip frames found among the full-film frames

For every frame the paper's Stage III received from a hand-cut clip, the closest frame Stage II selected on the full film of the same title. **Visual match**: perceptual hash (pHash, 64 bits) after cropping black borders; matched if the Hamming distance is <= 10 (16 = loose). **OCR-word coverage**: share of the clip frame's OCR words (Stage II OCR, >= 3 characters) found among the OCR words of the film's frames; it also counts credit rolls sampled a few frames apart, where the image differs but the text is the same. **Likely missing**: clip frames with no loose visual match (distance > 16) and less than 50% of their OCR words found; OCR noise differs between encodes, so this is an upper bound to check by eye.

| Title | Clip frames | Film frames | Visual match % (pHash <= 10) | Visual match % (<= 16) | Median distance | Mean OCR-word coverage % | Frames with all words found % | Unique clip words found % | Likely missing frames | Checked: present (v/b) | Checked: no credit in clip frame (n) | Checked: missing (x) | Not checked | Clip frames present % (after check) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8_e_mezzo | 36 | 113 | 88.9 | 91.7 | 2.0 | 85.6 | 61.1 | 87.8 | 1 | 1 | 0 | 0 | 0 | 100.0 |
| Amelie | 103 | 540 | 28.2 | 41.7 | 18.0 | 89.3 | 32.0 | 82.1 | 3 | 2 | 0 | 1 | 0 | 99.0 |
| Apocalypse_Now | 66 | 496 | 4.5 | 56.1 | 16.0 | 95.5 | 60.6 | 88.3 | 0 | 0 | 0 | 0 | 0 | 100.0 |
| Chernobyl_S01E01 | 79 | 141 | 89.9 | 91.1 | 0.0 | 98.5 | 86.1 | 99.2 | 0 | 0 | 0 | 0 | 0 | 100.0 |
| Dark_1x9 | 51 | 103 | 47.1 | 70.6 | 12.0 | 95.8 | 51.0 | 95.8 | 0 | 0 | 0 | 0 | 0 | 100.0 |
| El_desorden_que_dejas | 71 | 142 | 63.4 | 77.5 | 8.0 | 93.8 | 73.2 | 98.4 | 2 | 0 | 1 | 1 | 0 | 98.6 |
| Eternal_Sunshine_of_the_Spotless_Mind | 80 | 189 | 46.2 | 57.5 | 13.0 | 93.9 | 52.5 | 90.7 | 0 | 0 | 0 | 0 | 0 | 100.0 |
| Fight_Club | 85 | 673 | 3.5 | 20.0 | 18.0 | 71.1 | 0.0 | 96.7 | 7 | 1 | 1 | 5 | 0 | 94.1 |
| Hill_Street_Blues_1x13 | 44 | 111 | 90.9 | 95.5 | 1.0 | 94.4 | 79.5 | 95.1 | 0 | 0 | 0 | 0 | 0 | 100.0 |
| La_grande_bellezza | 103 | 213 | 67.0 | 87.4 | 6.0 | 93.5 | 70.9 | 89.7 | 0 | 0 | 0 | 0 | 0 | 100.0 |
| La_piovra_1x2 | 31 | 60 | 93.5 | 93.5 | 0.0 | 100.0 | 100.0 | 100.0 | 0 | 0 | 0 | 0 | 0 | 100.0 |
| Maigret_S03E01 | 83 | 171 | 86.7 | 95.2 | 0.0 | 93.1 | 81.7 | 90.8 | 0 | 0 | 0 | 0 | 0 | 100.0 |
| Planet_Earth_S01E10 | 11 | 36 | 18.2 | 36.4 | 20.0 | 96.8 | 54.5 | 95.4 | 0 | 0 | 0 | 0 | 0 | 100.0 |
| Prime_Suspect_1x1 | 35 | 211 | 25.7 | 31.4 | 20.0 | 68.4 | 37.1 | 68.8 | 9 | 0 | 3 | 6 | 0 | 82.9 |
| Psycho | 33 | 367 | 12.1 | 51.5 | 16.0 | 74.2 | 63.3 | 87.3 | 3 | 0 | 3 | 0 | 0 | 100.0 |
| Romanzo_criminale_S01E01 | 83 | 213 | 44.6 | 53.0 | 16.0 | 88.8 | 53.7 | 91.2 | 5 | 3 | 2 | 0 | 0 | 100.0 |
| Se7en | 150 | 430 | 28.7 | 42.0 | 18.0 | 77.4 | 34.5 | 84.0 | 19 | 5 | 14 | 0 | 0 | 100.0 |
| The_World_At_War_S01E03 | 32 | 122 | 75.0 | 84.4 | 2.0 | 59.7 | 34.4 | 66.7 | 2 | 2 | 0 | 0 | 0 | 100.0 |
| Twin_Peaks_1x3 | 62 | 240 | 75.8 | 88.7 | 4.0 | 84.6 | 72.6 | 93.3 | 1 | 0 | 1 | 0 | 0 | 100.0 |
| Yes,_Prime_Minister_1x8 | 16 | 57 | 93.8 | 93.8 | 1.0 | 87.1 | 68.8 | 82.2 | 1 | 0 | 1 | 0 | 0 | 100.0 |
| **Total** | 1254 | 4628 | 50.6 | 65.0 | 10.0 | 87.4 | 55.3 |  | 53 | 14 | 26 | 13 | 0 | 99.0 |

Per-frame details (closest film frame, distance, coverage) in `results/D_frame_overlap_frames.csv`.
