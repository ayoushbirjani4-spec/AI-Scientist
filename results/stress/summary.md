# Stress test: physics changes inside an experiment

Active strategy, Stage A predictor, budget 10. In change runs, g and μ are re-drawn 0.15–0.65 s into experiment 5; control runs have no change. 6 seeds per cell. Delay counts experiments from the contaminated one (1 = flagged on it). Recovery counts experiments after the contaminated one until the estimate stays within |g error| ≤ 1 % and |μ error| ≤ 0.01. Within-experiment CUSUM settings (drift 3, h 8.3) were chosen on separate calibration seeds (calibrate_cusum.py).

| noise | detector | detected | mean delay | flagged on the contaminated experiment | change-time error, median / mean (s) | false-alarm runs | recovered | mean recovery |
|---|---|---|---|---|---|---|---|---|
| low | between | 6/6 | 2.17 | 0/6 | – | 0/12 | 6/6 | 2.67 |
| low | between+within | 6/6 | 1.00 | 6/6 | 0.07 / 0.09 | 4/12 | 6/6 | 2.00 |
| medium | between | 6/6 | 2.17 | 0/6 | – | 0/12 | 6/6 | 2.83 |
| medium | between+within | 6/6 | 1.17 | 5/6 | 0.16 / 0.17 | 0/12 | 6/6 | 2.00 |
| high | between | 6/6 | 2.83 | 0/6 | – | 0/12 | 6/6 | 2.67 |
| high | between+within | 6/6 | 1.83 | 1/6 | 0.15 / 0.15 | 0/12 | 6/6 | 2.83 |
