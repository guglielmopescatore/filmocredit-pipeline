# Section B - scroll cap

Exact match against the human gold set. Calls and cost refer to the whole dataset (20 products); `cost_usd_uncached` prices every input token at full rate. The rerun is f9c9bb6 with PaddleOCR lang=en; it differs from the first revision in Stage II only by 5fec349 (scroll flow percentile).

## 20 products - No Role

| label | mode | products | TP | FP | FN | precision | recall | F1 | calls | cost_usd | cost_usd_uncached |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Pipeline first revision (cap 150) | No Role | 20 | 4563 | 198 | 302 | 0.9584 | 0.9379 | 0.9481 | 1254 | 29.23 | 65.09 |
| Rerun cap 150 | No Role | 20 | 4624 | 184 | 241 | 0.9617 | 0.9505 | 0.9561 | 1262 | 65.30 | 65.30 |
| Rerun cap 100 | No Role | 20 | 4649 | 206 | 216 | 0.9576 | 0.9556 | 0.9566 | 1371 | 71.81 | 71.81 |
| Rerun cap 75 | No Role | 20 | 4653 | 242 | 212 | 0.9506 | 0.9564 | 0.9535 | 1491 | 78.64 | 79.15 |

## 20 products - With Role

| label | mode | products | TP | FP | FN | precision | recall | F1 | calls | cost_usd | cost_usd_uncached |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Pipeline first revision (cap 150) | With Role | 20 | 4437 | 423 | 519 | 0.9130 | 0.8953 | 0.9040 | 1254 | 29.23 | 65.09 |
| Rerun cap 150 | With Role | 20 | 4497 | 420 | 459 | 0.9146 | 0.9074 | 0.9110 | 1262 | 65.30 | 65.30 |
| Rerun cap 100 | With Role | 20 | 4556 | 425 | 400 | 0.9147 | 0.9193 | 0.9170 | 1371 | 71.81 | 71.81 |
| Rerun cap 75 | With Role | 20 | 4541 | 502 | 415 | 0.9005 | 0.9163 | 0.9083 | 1491 | 78.64 | 79.15 |

## 15 cap-dependent products - No Role

| label | mode | products | TP | FP | FN | precision | recall | F1 | calls | cost_usd | cost_usd_uncached |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Pipeline first revision (cap 150) | No Role | 15 | 3555 | 159 | 274 | 0.9572 | 0.9284 | 0.9426 | 1254 | 29.23 | 65.09 |
| Rerun cap 150 | No Role | 15 | 3615 | 149 | 214 | 0.9604 | 0.9441 | 0.9522 | 1262 | 65.30 | 65.30 |
| Rerun cap 100 | No Role | 15 | 3640 | 171 | 189 | 0.9551 | 0.9506 | 0.9529 | 1371 | 71.81 | 71.81 |
| Rerun cap 75 | No Role | 15 | 3644 | 207 | 185 | 0.9463 | 0.9517 | 0.9490 | 1491 | 78.64 | 79.15 |

## 15 cap-dependent products - With Role

| label | mode | products | TP | FP | FN | precision | recall | F1 | calls | cost_usd | cost_usd_uncached |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Pipeline first revision (cap 150) | With Role | 15 | 3428 | 349 | 458 | 0.9076 | 0.8821 | 0.8947 | 1254 | 29.23 | 65.09 |
| Rerun cap 150 | With Role | 15 | 3488 | 347 | 398 | 0.9095 | 0.8976 | 0.9035 | 1262 | 65.30 | 65.30 |
| Rerun cap 100 | With Role | 15 | 3547 | 352 | 339 | 0.9097 | 0.9128 | 0.9112 | 1371 | 71.81 | 71.81 |
| Rerun cap 75 | With Role | 15 | 3532 | 429 | 354 | 0.8917 | 0.9089 | 0.9002 | 1491 | 78.64 | 79.15 |

## Recall per product (No Role)

| product | Pipeline first revision (cap 150) recall | Rerun cap 150 recall | Rerun cap 100 recall | Rerun cap 75 recall |
|---|---|---|---|---|
| 8 e mezzo | 0.9859 | 0.9718 | 0.9718 | 0.9718 |
| amelie | 0.9139 | 0.9306 | 0.9250 | 0.9333 |
| apocalypse now | 0.9265 | 0.9361 | 0.9361 | 0.9361 |
| chernobyl s01e01 | 0.9982 | 0.9964 | 0.9964 | 0.9964 |
| dark s01e09 | 0.8899 | 0.8917 | 0.9188 | 0.9188 |
| el desorden que dejas s01e03 | 0.9510 | 0.9551 | 0.9612 | 0.9633 |
| eternal sunshine of the spotless mind | 0.9504 | 0.9650 | 0.9563 | 0.9738 |
| fight club | 0.9291 | 0.9449 | 0.9501 | 0.9344 |
| hill street blues s01e13 | 0.9859 | 0.9859 | 0.9859 | 0.9859 |
| la grande bellezza | 0.8737 | 0.9293 | 0.9315 | 0.9336 |
| la piovra s01e02 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| maigret s03e01 | 0.9574 | 0.9574 | 1.0000 | 0.9894 |
| planet earth s01e10 | 0.9310 | 0.9828 | 0.9828 | 0.9655 |
| prime suspect s01e01 | 0.9231 | 0.9615 | 1.0000 | 0.9872 |
| psycho | 0.9375 | 0.9688 | 1.0000 | 1.0000 |
| romanzo criminale s01e01 | 0.9525 | 0.9436 | 0.9407 | 0.9436 |
| se7en | 0.9745 | 0.9784 | 0.9804 | 0.9823 |
| the world at war s01e03 | 0.9231 | 0.9231 | 0.9231 | 0.9231 |
| twin peaks s01e03 | 0.9773 | 0.9773 | 0.9773 | 0.9773 |
| yes prime minister s01e08 | 0.6538 | 0.9615 | 1.0000 | 1.0000 |
