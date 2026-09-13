## Table 1 reproduction: average normalized score

| Setting | Runs | Ours | 95% CI (stratified bootstrap) | Rank | Paper | Paper rank |
| :--- | ---: | ---: | :---: | ---: | ---: | ---: |
| No clipping | 21 | 0.15 | [0.07, 0.23] | 4 | -0.39 | 4 |
| Clip ε=0.1 | 21 | 0.82 | [0.80, 0.84] | 1 | 0.76 | 2 |
| Clip ε=0.2 | 21 | 0.79 | [0.72, 0.87] | 2 | 0.82 | 1 |
| Clip ε=0.3 | 21 | 0.79 | [0.72, 0.86] | 3 | 0.70 | 3 |

## Normalized score per environment (mean over seeds)

| Env | No clipping | Clip ε=0.1 | Clip ε=0.2 | Clip ε=0.3 |
| :--- | ---: | ---: | ---: | ---: |
| HalfCheetah-v5 | 0.10 | 0.44 | 0.63 | 0.64 |
| Hopper-v5 | 0.15 | 0.90 | 0.75 | 0.82 |
| InvertedDoublePendulum-v5 | 0.30 | 0.86 | 0.92 | 0.92 |
| InvertedPendulum-v5 | 0.02 | 0.97 | 0.92 | 0.92 |
| Reacher-v5 | 0.35 | 0.99 | 0.93 | 0.95 |
| Swimmer-v5 | 0.03 | 0.97 | 0.87 | 0.64 |
| Walker2d-v5 | 0.08 | 0.65 | 0.53 | 0.62 |

## Raw return, last 100 training episodes (mean ± std over seeds)

| Env | Random | No clipping | Clip ε=0.1 | Clip ε=0.2 | Clip ε=0.3 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| HalfCheetah-v5 | -287.2 | 113.1 ± 252.7 | 1407.0 ± 66.2 | 2133.6 ± 1177.6 | 2171.7 ± 1215.8 |
| Hopper-v5 | 18.4 | 425.1 ± 522.1 | 2451.3 ± 182.4 | 2063.9 ± 670.7 | 2252.4 ± 740.0 |
| InvertedDoublePendulum-v5 | 48.6 | 2004.8 ± 3075.4 | 5722.7 ± 207.7 | 6103.5 ± 336.4 | 6115.4 ± 471.7 |
| InvertedPendulum-v5 | 5.3 | 27.2 ± 33.8 | 945.7 ± 32.2 | 897.7 ± 55.3 | 896.7 ± 26.2 |
| Reacher-v5 | -42.9 | -29.7 ± 13.4 | -5.3 ± 0.5 | -7.5 ± 1.1 | -6.6 ± 2.1 |
| Swimmer-v5 | 0.5 | 3.1 ± 13.4 | 98.1 ± 4.4 | 88.1 ± 13.8 | 64.9 ± 29.5 |
| Walker2d-v5 | 1.8 | 400.1 ± 68.1 | 3085.0 ± 604.9 | 2534.2 ± 1906.5 | 2937.0 ± 470.9 |
