# Results

Written by `python -m sm.figures` from `results/summary.json`. Brackets are 95% hierarchical bootstrap intervals over seeds and scenes.

## Seeds

| size | seeds |
|---|---|
| 1M | 3 |
| 5M | 1 |

## Thresholds: magnitude at 75% correct in 2AFC

| family | unit | ideal | dead reckoning | no memory | 1M | 5M |
|---|---|---|---|---|---|---|
| teleport | px | 0.034 | 0.061 | 0.061 | 0.12 [0.095, 0.14] | 0.11 [0.086, 0.13] |
| deflect | deg | 0.37 | 6.8 | 6.8 | 5.7 [4.8, 6.7] | 3.7 [3.2, 4.2] |
| phantom | px | 0.017 | 2.8 | 2.8 | 0.068 [0.035, 0.11] | 0.054 [0.039, 0.073] |
| collision | restitution error | 0.008 | not reached | not reached | 1.5 [1.3, 1.5] | 0.72 [0.64, 0.97] |
| hidden | px | 0.058 | 1.1 | not reached | 2.5 [2.3, not reached] | 1.1 [0.93, 1.3] |

## Efficiency against the ideal observer: (ideal threshold / model threshold) squared

| family | 1M | 5M |
|---|---|---|
| teleport | 0.084 [0.0598, 0.128] | 0.103 [0.0737, 0.157] |
| deflect | 0.00431 [0.00312, 0.00614] | 0.0103 [0.00784, 0.0139] |
| phantom | 0.0668 [0.0267, 0.256] | 0.105 [0.0573, 0.201] |
| collision | 2.96e-05 [2.72e-05, 3.76e-05] | 0.000124 [6.9e-05, 0.00016] |
| hidden | 0.00051 [0, 0.000637] | 0.0027 [0.00195, 0.00385] |

## Yes/no detection: magnitude detected half the time above false alarms, at one lawful frame in a thousand

| family | unit | ideal | 1M | 5M |
|---|---|---|---|---|
| teleport | px | 0.5 | 0.83 | 0.74 |
| deflect | deg | 5.1 | 39 | 32 |
| phantom | px | 0.25 | 0.48 | 0.41 |
| collision | restitution error | 0.1 | not reached | not reached |
| hidden | px | 0.63 | not reached | not reached |

## Hidden time, in frames

| family | measure | 1M | 5M |
|---|---|---|---|
| vanish | 2AFC falls through chance | 2.6 [2.2, 3.0] | 10.1 [8.9, 11.1] |
| vanish | trace reaches half its first value | 7.3 [6.2, 8.3] | 11.5 [10.3, 13.2] |
| rebound | 2AFC falls through chance | 0.0 [0.0, 0.0] | 14.0 [12.8, 15.4] |
| rebound | trace reaches half its first value | 19.1 [15.8, 22.6] | 22.4 [19.5, 25.3] |

## Reference observers on hidden events: 2AFC at the shortest and longest durations

| family | observer | shortest | longest |
|---|---|---|---|
| vanish | ideal | 1.000 | 1.000 |
| vanish | dead reckoning | 0.990 | 1.000 |
| vanish | no memory | 0.005 | 0.198 |
| rebound | ideal | 1.000 | 1.000 |
| rebound | dead reckoning | 0.042 | 0.115 |
| rebound | no memory | 0.042 | 0.115 |

## Decoded position while hidden (px, median over scenes, mean over seeds)

### vanish, read 100% of the way through the hidden interval

| D | 1M | 5M | last seen | straight line |
|---|---|---|---|---|
| 4 | 1.68 | 0.75 | 3.24 | 0.00 |
| 8 | 2.12 | 0.95 | 5.08 | 0.00 |
| 12 | 3.67 | 1.05 | 6.27 | 0.00 |
| 16 | 6.00 | 1.19 | 6.91 | 0.00 |
| 20 | 7.45 | 1.19 | 6.16 | 0.00 |
| 24 | 7.80 | 1.46 | 7.01 | 0.00 |
| 28 | 7.75 | 1.47 | 7.13 | 0.00 |
| 32 | 8.41 | 2.07 | 7.54 | 0.00 |
| 40 | 9.00 | 2.51 | 7.77 | 0.00 |
| 48 | 8.78 | 4.55 | 8.13 | 0.00 |

### rebound, read 75% of the way through the hidden interval

| D | 1M | 5M | last seen | straight line |
|---|---|---|---|---|
| 4 | 1.47 | 0.63 | 0.53 | 2.32 |
| 8 | 1.54 | 0.62 | 1.09 | 3.42 |
| 12 | 2.11 | 0.87 | 1.67 | 4.33 |
| 16 | 8.28 | 0.91 | 2.30 | 5.31 |
| 20 | 11.85 | 0.93 | 2.89 | 6.51 |
| 24 | 11.38 | 0.95 | 3.43 | 7.20 |
| 28 | 11.61 | 1.21 | 3.88 | 7.84 |
| 32 | 11.67 | 1.81 | 3.98 | 7.96 |
| 40 | 11.90 | 3.48 | 4.06 | 7.73 |
| 48 | 12.17 | 4.32 | 3.32 | 6.61 |

## Calibration on held-out lawful frames: measured rate where one in a thousand was promised

| size | all | quiet | edge | contact | reveal |
|---|---|---|---|---|---|
| 1M | 0.00087 | 0.00046 | 0.00160 | 0.00090 | 0.00133 |
| 5M | 0.00088 | 0.00062 | 0.00132 | 0.00095 | 0.00147 |
