# CUSUM calibration (seeds 100+, not used for reporting)

h = 1.25 × the largest CUSUM value seen in any clean experiment (clean runs plus the pre-change part of change runs), so no clean calibration experiment would flag. A setting is only eligible if h exceeds the most one capped frame can add (cap − 1 − drift), so a single bad frame can never trigger it (the spec: flag persistent deviation, not a single error).

| drift | h | needs ≥ 2 frames | changes caught on the contaminated experiment | mean change-time error (s) |
|---|---|---|---|---|
| 0.5 | 68.3 | yes | 9/24 | 0.037 |
| 1.0 | 51.9 | yes | 10/24 | 0.058 |
| 2.0 | 22.9 | yes | 11/24 | 0.095 |
| 3.0 | 8.3 | yes | 12/24 | 0.179 |
| 4.0 | 3.3 | no | 13/24 | 0.180 |
| 5.0 | 1.6 | no | 13/24 | 0.187 |

Chosen: drift = 3.0, h = 8.3 (most changes caught among eligible settings; ties go to the smaller drift).
