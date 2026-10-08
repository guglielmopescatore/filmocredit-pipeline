# Second revision - data and analyses

Datasets and scripts for the runs requested by the second review (`RIEPILOGO_run_v2.md`, blocks A-D).
All new runs used commit `f9c9bb6`, GPT-5.6 Sol Standard (reasoning standard/medium), no previous-frame
image, July stopword list, PaddleOCR `lang=en`, fuzzy threshold 88. Each run lives in its own git worktree
`runs/<ID>/` with `run_info.json`, `run.log`, its DB and `run_output/` (export, per-call costs, raw responses).

Run everything from the repository root:

```
python analysis_2nd_revision/build_datasets.py        # data/*.csv + data/datasets.json
python analysis_2nd_revision/section_A_naive.py       # results/A_naive.*
python analysis_2nd_revision/section_B_cap.py         # results/B_cap.*
python analysis_2nd_revision/section_C_repetitions.py # results/C_repetitions.*, results/C_dispersion.csv
python analysis_2nd_revision/section_D_full_films.py  # D_full_films_stage1_stage2.md, results/D_full_films.csv
python analysis_2nd_revision/section_D_frame_overlap.py # results/D_frame_overlap.*
```

## data/

| Dataset | Products | Source |
|---|---:|---|
| `ORIG_SOL_20products` | 20 | first-revision run `db/FUZZY88_GPT_SOL_STANDARD_25products_tvcredits_v3.db`, main corpus |
| `ORIG_SOL_holdout5` | 5 | same run, the 5 hold-out products |
| `A_NAIVE_0.8s` | 20 | first-revision naive run (`NAIVE_FUZZY88_..._20products`) |
| `A_NAIVE_2.4s`, `A_NAIVE_4.0s`, `A_NAIVE_4.8s` | 20 | `runs/NAIVE_2.4s`, `runs/NAIVE_4.0s`, `runs/NAIVE_4.8s` (k = 3, 5 and 6 subsamples of the 0.8 s frames) |
| `B_CAP150` | 20 | `runs/PIPE_cap150` (14 roll titles) + `runs/PIPE_noscroll_cap150` (6 titles) |
| `B_CAP100`, `B_CAP75` | 20 | `runs/PIPE_cap<N>` (14 roll titles) + `runs/PIPE_noscroll_cap<N>` (Yes, Prime Minister, whose roll is found since 5fec349) + the other 5 titles from `runs/PIPE_noscroll_cap150` |
| `C_SOL_r1..r5` | 5 | `runs/REP_SOL_r<n>`, the 207 July hold-out frames |
| `C_GEMMA_r1..r5` | 5 | `runs/REP_GEMMA_r<n>`, same frames, Gemma 4 12B local, temperature 0 |
| `D_FULL` | 20 | `runs/FULL_pipeline`: full-length films, Stage I ("whole episode") + II + III + IV; episode ids `<product>_FULL` mapped back to the product names |

`datasets.json` records, per dataset, the source CSV and DB of every part, the products taken from each,
and the VLM calls with tokens and cost. `cost_usd` is the actual cost (July runs read most input from the
Azure cache; September runs almost never, the default retention became `24h`); `cost_usd_uncached` prices
every input token at full rate so that runs from the two periods are comparable.

Notes:
- **B_CAP150 is not the first-revision run**: all 20 products were redone with f9c9bb6. Its Stage II differs
  from the first revision's (2025-12-04, commit 00f0cbc5) only by 5fec349 (scroll flow percentile chosen by
  scroll direction), which reclassifies some scenes as rolls or static (e.g. Yes, Prime Minister); the
  reference was also made with `lang=en` (`revision_checks/stage2_ocr_language_check_output.txt`).
- There is **no first-revision Gemma run on the hold-out**: the July Gemma run covers only the 20 main
  products, so block C has five new Gemma repetitions and no original to compare them with.

## Analyses

The exact-match scoring is the one of `compare_llm_human_metrics.py` (persons only, No Role / With Role,
deduplicated per product, each dataset evaluated on the products it covers): `metrics_lib.py` imports its
functions instead of copying them, so the numbers cannot drift from the paper's method.

| Script | Question (RIEPILOGO) | Output |
|---|---|---|
| `section_A_naive.py` | A: does a sparser naive sampling cost less than the pipeline and perform as well? | P/R/F1, calls, cost, cost per TP; recall per product |
| `section_B_cap.py` | B: which roll cap works best? | P/R/F1 and cost at cap 150/100/75 vs the first revision, on 20 products and on the 15 cap-dependent ones; recall per product |
| `section_C_repetitions.py` | C: how much does a repeated run change? | P/R/F1 per repetition; mean, SD, min, max, range and pairwise Jaccard of the predictions |
| `section_C_holdout_check.py` | C: why is the published hold-out run (With Role 0.925) above all five repetitions? | F1 per run, published run recomputed, dates/prompt/platform of the published run, credit-by-credit role differences and TP-gap decomposition (`results/C_holdout_check.md`) |
| `section_D_full_films.py` | D: Stage I/II on full films vs hand-cut clips | `D_full_films_stage1_stage2.md` (shots, candidate scenes, frames, machine time per title) |
| `section_D_frame_overlap.py` | D: are the clip frames also selected on the full films? | per title: visual match (pHash after border crop), OCR-word coverage, likely missing frames; per-frame CSV |

Block D ran end to end (4,643 Stage III calls on the films, 214.27 USD); `section_D_full_films.py` also compares
the credits extracted from the films with the gold set, next to the clips (`results/D_credits_vs_gold.csv`).
