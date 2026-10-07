# Section C - repetitions on the hold-out

Same 207 frames (July hold-out Stage II) in every run, exact match against the hold-out gold set. Gemma runs locally with temperature 0; there is no first-revision Gemma run on the hold-out.

## Runs - No Role

| label | mode | products | TP | FP | FN | precision | recall | F1 | calls | cost_usd | cost_usd_uncached |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GPT Sol Standard - first revision | No Role | 5 | 958 | 46 | 24 | 0.9542 | 0.9756 | 0.9647 | 207 | 10.06 | 12.30 |
| GPT Sol Standard - r1 | No Role | 5 | 962 | 45 | 20 | 0.9553 | 0.9796 | 0.9673 | 207 | 9.7150 | 12.27 |
| GPT Sol Standard - r2 | No Role | 5 | 963 | 43 | 19 | 0.9573 | 0.9807 | 0.9688 | 207 | 6.8140 | 12.18 |
| GPT Sol Standard - r3 | No Role | 5 | 963 | 43 | 19 | 0.9573 | 0.9807 | 0.9688 | 207 | 10.35 | 12.18 |
| GPT Sol Standard - r4 | No Role | 5 | 963 | 43 | 19 | 0.9573 | 0.9807 | 0.9688 | 207 | 7.4765 | 12.29 |
| GPT Sol Standard - r5 | No Role | 5 | 966 | 35 | 16 | 0.9650 | 0.9837 | 0.9743 | 207 | 7.2586 | 12.22 |
| Gemma 4 12B - r1 | No Role | 5 | 765 | 359 | 217 | 0.6806 | 0.7790 | 0.7265 | 207 | 0.0000 | 0.0000 |
| Gemma 4 12B - r2 | No Role | 5 | 765 | 359 | 217 | 0.6806 | 0.7790 | 0.7265 | 207 | 0.0000 | 0.0000 |
| Gemma 4 12B - r3 | No Role | 5 | 765 | 359 | 217 | 0.6806 | 0.7790 | 0.7265 | 207 | 0.0000 | 0.0000 |
| Gemma 4 12B - r4 | No Role | 5 | 765 | 359 | 217 | 0.6806 | 0.7790 | 0.7265 | 207 | 0.0000 | 0.0000 |
| Gemma 4 12B - r5 | No Role | 5 | 765 | 359 | 217 | 0.6806 | 0.7790 | 0.7265 | 207 | 0.0000 | 0.0000 |

## Runs - With Role

| label | mode | products | TP | FP | FN | precision | recall | F1 | calls | cost_usd | cost_usd_uncached |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GPT Sol Standard - first revision | With Role | 5 | 975 | 92 | 67 | 0.9138 | 0.9357 | 0.9246 | 207 | 10.06 | 12.30 |
| GPT Sol Standard - r1 | With Role | 5 | 942 | 120 | 100 | 0.8870 | 0.9040 | 0.8954 | 207 | 9.7150 | 12.27 |
| GPT Sol Standard - r2 | With Role | 5 | 948 | 115 | 94 | 0.8918 | 0.9098 | 0.9007 | 207 | 6.8140 | 12.18 |
| GPT Sol Standard - r3 | With Role | 5 | 939 | 123 | 103 | 0.8842 | 0.9012 | 0.8926 | 207 | 10.35 | 12.18 |
| GPT Sol Standard - r4 | With Role | 5 | 952 | 115 | 90 | 0.8922 | 0.9136 | 0.9028 | 207 | 7.4765 | 12.29 |
| GPT Sol Standard - r5 | With Role | 5 | 959 | 95 | 83 | 0.9099 | 0.9204 | 0.9151 | 207 | 7.2586 | 12.22 |
| Gemma 4 12B - r1 | With Role | 5 | 536 | 682 | 506 | 0.4401 | 0.5144 | 0.4743 | 207 | 0.0000 | 0.0000 |
| Gemma 4 12B - r2 | With Role | 5 | 536 | 682 | 506 | 0.4401 | 0.5144 | 0.4743 | 207 | 0.0000 | 0.0000 |
| Gemma 4 12B - r3 | With Role | 5 | 536 | 682 | 506 | 0.4401 | 0.5144 | 0.4743 | 207 | 0.0000 | 0.0000 |
| Gemma 4 12B - r4 | With Role | 5 | 536 | 682 | 506 | 0.4401 | 0.5144 | 0.4743 | 207 | 0.0000 | 0.0000 |
| Gemma 4 12B - r5 | With Role | 5 | 536 | 682 | 506 | 0.4401 | 0.5144 | 0.4743 | 207 | 0.0000 | 0.0000 |

## Dispersion over the 5 repetitions

| model | mode | repetitions | precision_mean | precision_sd | recall_mean | recall_sd | F1_mean | F1_sd | F1_min | F1_max | F1_range | pairwise_jaccard_mean | pairwise_jaccard_min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GPT Sol Standard | No Role | 5 | 0.9584 | 0.0038 | 0.9811 | 0.0015 | 0.9696 | 0.0027 | 0.9673 | 0.9743 | 0.0070 | 0.9532 | 0.9449 |
| GPT Sol Standard | With Role | 5 | 0.8930 | 0.0100 | 0.9098 | 0.0076 | 0.9013 | 0.0087 | 0.8926 | 0.9151 | 0.0225 | 0.8930 | 0.8675 |
| Gemma 4 12B | No Role | 5 | 0.6806 | 0.0000 | 0.7790 | 0.0000 | 0.7265 | 0.0000 | 0.7265 | 0.7265 | 0.0000 | 1.0000 | 1.0000 |
| Gemma 4 12B | With Role | 5 | 0.4401 | 0.0000 | 0.5144 | 0.0000 | 0.4743 | 0.0000 | 0.4743 | 0.4743 | 0.0000 | 1.0000 | 1.0000 |

Full statistics (TP/FP/FN included) in results/C_dispersion.csv.
