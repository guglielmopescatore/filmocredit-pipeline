# Section C - the published hold-out run vs the five repetitions

## 1-2. F1 of every run (same scoring code for all)

Exact match against the hold-out gold set with `compare_llm_human_metrics.py`'s functions, the same code and options for the published run and the repetitions. The published run recomputed this way still gives **0.9246** With Role: the difference is not in the calculation.

| run | F1 No Role | F1 With Role | TP With Role | FP With Role | FN With Role |
|---|---|---|---|---|---|
| published (July 2026) | 0.9647 | 0.9246 | 975 | 92 | 67 |
| repetition r1 | 0.9673 | 0.8954 | 942 | 120 | 100 |
| repetition r2 | 0.9688 | 0.9007 | 948 | 115 | 94 |
| repetition r3 | 0.9688 | 0.8926 | 939 | 123 | 103 |
| repetition r4 | 0.9688 | 0.9028 | 952 | 115 | 90 |
| repetition r5 | 0.9743 | 0.9151 | 959 | 95 | 83 |

## 3. When and with which code the published run was made

Stage III of the published hold-out run: 18-23 July 2026; Stage IV on 24 July 11:10. Commit `5fec349` ("Unify pipeline steps and upgrade IMDB matching", ~1,200 changed lines in Stage III/IV files) is of 2026-07-24 22:41, the previous commit `3c9dcf8` of 2026-07-13 18:06: the published run used the uncommitted working tree of those days. The published 20-product run is of the same days (Stage III 17-22 July 2026).

**Prompt.** The VLM prompt of `f9c9bb6` equals the one of `5fec349`, and differs from the 13 July commit by ~100 tokens (name transcription, guild acronyms, special thanks). On the first frame of every clip (empty previous-credits context, so the input is prompt + image only) the input tokens of the published run and of the repetitions are equal or differ by 5-8 tokens out of 10-12 thousand: the published run already used (almost exactly) today's prompt.

| clip | published Stage III (UTC) | first-frame input tokens, published | same frame, r1 | difference |
|---|---|---|---|---|
| 3_Percent_S01E06_End | 2026-07-18 16:45 - 16:53 | 10017 | 10025 | 8 |
| 3_Percent_S01E06_Opening | 2026-07-18 16:56 - 16:57 | 10017 | 10025 | 8 |
| Blue_Eye_Samurai_S01E01_End | 2026-07-23 06:38 - 06:45 | 12070 | 12070 | 0 |
| Honeyland_End | 2026-07-23 06:09 - 06:20 | 10726 | 10726 | 0 |
| Persepolis_End | 2026-07-22 19:11 - 19:48 | 11993 | 11998 | 5 |
| Persepolis_Opening | 2026-07-22 19:48 - 19:53 | 11993 | 11998 | 5 |
| Wild_Strawberries_End | 2026-07-18 18:22 - 18:23 | 10442 | 10450 | 8 |
| Wild_Strawberries_Opening | 2026-07-18 18:27 - 18:30 | 10442 | 10450 | 8 |

**Model and platform.** Same deployment name, reasoning (standard/medium) and sampling parameters (model=gpt-5.6-sol, service_tier=default, temperature=1.0, top_p=0.98, truncation=disabled) in July and September. The Azure responses, however, changed: September responses carry fields absent in July (tool_usage) and the default prompt cache retention went from `in_memory` to `24h`. The service behind `gpt-5.6-sol` was updated between the two periods; whether the model weights changed cannot be told from the responses (no snapshot id is returned).

## 4. Credit-by-credit role comparison

Of 1,042 gold credits (name + role): **16 are right in the published run and wrong in all five repetitions**, **3 the opposite**. In every one of these the name is found on both sides: only the role differs.

- published right / repetitions wrong, by role: [('art department', 8), ('production managers', 4), ('animation department', 3), ('production finance and accounting', 1)]; by product: [('persepolis', 10), ('blue eye samurai s01e01', 5), ('3 percent s01e06', 1)]
- repetitions right / published wrong, by role: [('editorial department', 2), ('additional crew', 1)]; by product: [('persepolis', 3)]

They are concentrated on a few credit cards with ambiguous roles, where all five repetitions agree on the same alternative: *Persepolis* storyboard (`scenarimage`) credits classified as Animation instead of Art Department, *Blue Eye Samurai* heads of studio as Additional Crew instead of Production Managers, trainees (`stagiaire`, `estagiario`) as Additional Crew. The model reads the same `role_detail` text in both periods and assigns a different `role_group`; the normalization code does not change it.

| product | name | gold_role | published_roles | repetition_roles |
|---|---|---|---|---|
| 3 percent s01e06 | rafael ramos | production finance and accounting | production finance and accounting | additional crew |
| blue eye samurai s01e01 | aoi yamaguchi | animation department | animation department | art department; visual effects |
| blue eye samurai s01e01 | camille gerard | production managers | production managers | additional crew |
| blue eye samurai s01e01 | didier henry | production managers | production managers | additional crew |
| blue eye samurai s01e01 | eleonore moreau | production managers | production managers | additional crew |
| blue eye samurai s01e01 | emilie gaurier | production managers | production managers | additional crew |
| persepolis | alexandre hesse | art department | animation department; art department | animation department |
| persepolis | alexis venet | art department | animation department; art department | animation department; visual effects |
| persepolis | alice lia | art department | animation department; art department | animation department |
| persepolis | benoit bayart | art department | animation department; art department | animation department |
| persepolis | caroline piochon | art department | art department | animation department |
| persepolis | david etien | art department | animation department; art department | animation department |
| persepolis | jean charles finck | art department | art department | animation department |
| persepolis | sebastien lonjon | animation department | animation department | additional crew |
| persepolis | stephane beau | art department | art department | animation department |
| persepolis | valentin capiau | animation department | animation department | additional crew |
| persepolis | angelique muguet | editorial department | production managers | editorial department |
| persepolis | celine merrien | additional crew | animation department; cast | additional crew; cast |
| persepolis | cyril vonck | editorial department | production managers | editorial department |

**Decomposition of the With-Role TP gap** (published minus mean of the repetitions = +27.0 TP):

| product | TP gap contribution |
|---|---|
| persepolis | 12.60 |
| blue eye samurai s01e01 | 9.2000 |
| 3 percent s01e06 | 5.8000 |
| wild strawberries | 0.4000 |
| honeyland | -1.0000 |

| gold role | TP gap contribution |
|---|---|
| art department | 11.80 |
| animation department | 8.4000 |
| production managers | 4.4000 |
| visual effects | 4.2000 |
| production finance and accounting | 2.2000 |
| production department | 1.2000 |
| additional crew | 1.2000 |
| thanks | -1.2000 |
| cast | -1.8000 |
| editorial department | -2.6000 |

## Reading

- Not the calculation (check 2) and not the prompt (check 3: same input tokens on the first frame).
- The published run is not uniformly better: on names (No Role) it is *below* all five repetitions (0.9647 vs 0.9673-0.9743); only on roles is it above (TP 975 vs repetitions 948 +/- 8.0, 3.4 SD). A lucky draw from today's distribution would not be that far off, nor better on roles and worse on names at the same time.
- The difference is a consistent change of the role assigned to a few blocks of credits, the same in all five repetitions, with the service behind the same deployment name updated in between. This looks like a change in model behaviour between July and September rather than a lucky single run, but the two cannot be separated with these data: the July service can no longer be queried.
- On the 20 main products the comparison published vs today (0.904 vs 0.911 With Role) is not on the same input: today's run also changes Stage II (5fec349, OCR) and therefore the frames. The hold-out repetitions are the only same-input comparison.
- Safe statement for the paper: the hold-out figure is the mean of the five repetitions (see results/C_dispersion.csv), with the published run reported as an earlier execution on the July service.
