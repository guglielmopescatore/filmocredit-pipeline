# Section A - naive interval curve

Exact match against the human gold set, 20 main-corpus products. `cost_usd` is what each run cost with the Azure cache behaviour of its time (July: in-memory cache reads; September: almost none); `cost_usd_uncached` prices every input token at full rate, so the runs are comparable.

## No Role

| label | mode | products | TP | FP | FN | precision | recall | F1 | calls | cost_usd | cost_usd_uncached | cost_per_TP_uncached |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Pipeline (first revision) | No Role | 20 | 4563 | 198 | 302 | 0.9584 | 0.9379 | 0.9481 | 1254 | 29.23 | 65.09 | 0.0143 |
| Naive 0.8 s | No Role | 20 | 4808 | 585 | 57 | 0.8915 | 0.9883 | 0.9374 | 6582 | 152.07 | 327.30 | 0.0681 |
| Naive 2.4 s | No Role | 20 | 4325 | 269 | 540 | 0.9415 | 0.8890 | 0.9145 | 2201 | 110.09 | 111.50 | 0.0258 |
| Naive 4.0 s | No Role | 20 | 3844 | 170 | 1021 | 0.9577 | 0.7901 | 0.8659 | 1328 | 66.64 | 66.95 | 0.0174 |
| Naive 4.8 s | No Role | 20 | 3650 | 145 | 1215 | 0.9618 | 0.7503 | 0.8430 | 1106 | 56.12 | 56.21 | 0.0154 |

## With Role

| label | mode | products | TP | FP | FN | precision | recall | F1 | calls | cost_usd | cost_usd_uncached | cost_per_TP_uncached |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Pipeline (first revision) | With Role | 20 | 4437 | 423 | 519 | 0.9130 | 0.8953 | 0.9040 | 1254 | 29.23 | 65.09 | 0.0147 |
| Naive 0.8 s | With Role | 20 | 4729 | 1030 | 227 | 0.8212 | 0.9542 | 0.8827 | 6582 | 152.07 | 327.30 | 0.0692 |
| Naive 2.4 s | With Role | 20 | 4158 | 634 | 798 | 0.8677 | 0.8390 | 0.8531 | 2201 | 110.09 | 111.50 | 0.0268 |
| Naive 4.0 s | With Role | 20 | 3697 | 426 | 1259 | 0.8967 | 0.7460 | 0.8144 | 1328 | 66.64 | 66.95 | 0.0181 |
| Naive 4.8 s | With Role | 20 | 3458 | 455 | 1498 | 0.8837 | 0.6977 | 0.7798 | 1106 | 56.12 | 56.21 | 0.0163 |

## Recall per product (No Role)

| product | Pipeline (first revision) recall | Naive 0.8 s recall | Naive 2.4 s recall | Naive 4.0 s recall | Naive 4.8 s recall |
|---|---|---|---|---|---|
| 8 e mezzo | 0.9859 | 1.0000 | 0.9859 | 0.9859 | 0.9577 |
| amelie | 0.9139 | 0.9583 | 0.9472 | 0.8889 | 0.8861 |
| apocalypse now | 0.9265 | 0.9904 | 0.9617 | 0.9425 | 0.9297 |
| chernobyl s01e01 | 0.9982 | 0.9982 | 0.4809 | 0.2995 | 0.2577 |
| dark s01e09 | 0.8899 | 0.9964 | 0.9513 | 0.7996 | 0.7960 |
| el desorden que dejas s01e03 | 0.9510 | 0.9939 | 0.8082 | 0.5633 | 0.4571 |
| eternal sunshine of the spotless mind | 0.9504 | 0.9854 | 0.9825 | 0.9446 | 0.9650 |
| fight club | 0.9291 | 0.9790 | 0.9370 | 0.8793 | 0.7165 |
| hill street blues s01e13 | 0.9859 | 0.9859 | 0.7606 | 0.5352 | 0.3944 |
| la grande bellezza | 0.8737 | 0.9700 | 0.9572 | 0.9015 | 0.8865 |
| la piovra s01e02 | 1.0000 | 1.0000 | 1.0000 | 0.9655 | 0.6207 |
| maigret s03e01 | 0.9574 | 1.0000 | 1.0000 | 0.9787 | 0.9681 |
| planet earth s01e10 | 0.9310 | 0.9828 | 0.9828 | 0.8276 | 0.6552 |
| prime suspect s01e01 | 0.9231 | 1.0000 | 1.0000 | 0.9744 | 0.9872 |
| psycho | 0.9375 | 1.0000 | 1.0000 | 0.9062 | 0.9062 |
| romanzo criminale s01e01 | 0.9525 | 0.9970 | 0.9644 | 0.9050 | 0.9021 |
| se7en | 0.9745 | 0.9941 | 0.9862 | 0.9666 | 0.9509 |
| the world at war s01e03 | 0.9231 | 1.0000 | 1.0000 | 0.8462 | 0.9231 |
| twin peaks s01e03 | 0.9773 | 1.0000 | 0.8409 | 0.5682 | 0.4773 |
| yes prime minister s01e08 | 0.6538 | 1.0000 | 1.0000 | 1.0000 | 0.9231 |
