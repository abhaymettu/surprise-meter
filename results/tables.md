# Results

Written by `python -m sm.figures` from `results/summary.json`. Brackets are 95% hierarchical bootstrap intervals over seeds and scenes.

## Seeds

| size | seeds |
|---|---|
| 1M | 1m-s0, 1m-s1, 1m-s2, 1m-s3 |
| 5M | 5m-s1, 5m-s2 |
| 1M, 8-frame window | 1m-window8-s0 |

## Thresholds: magnitude at 75% correct in 2AFC

| family | unit | told the state | causal ideal | dead reckoning with walls | wall-blind | no memory | 1M | 5M |
|---|---|---|---|---|---|---|---|---|
| teleport | px | 0.04 | 0.065 | 0.068 | 0.068 | 0.068 | 0.16 [0.13, 0.18] | 0.14 [0.12, 0.16] |
| deflect | deg | 0.44 | 0.63 | 7.1 | 7.1 | 7.1 | 5.6 [4.7, 6.7] | 3.4 [2.9, 3.9] |
| phantom | px | 0.022 | 0.032 | 0.27 | 4.5 | 4.5 | 0.095 [0.072, 0.12] | 0.093 [0.064, 0.1] |
| collision | restitution error | 0.0092 | 0.011 | not reached | not reached | not reached | 1.2 [1.1, 1.3] | 0.74 [0.67, 0.82] |
| hidden | px | below 0.05 | below 0.05 | 1.5 | 1.9 | not reached | 2.3 [1.9, 2.9] | 1.1 [0.91, 1.4] |

## Efficiency against the causal ideal: (its threshold / the model's) squared

| family | 1M | 5M |
|---|---|---|
| teleport | 0.169 [0.135, 0.229] | 0.216 [0.168, 0.31] |
| deflect | 0.0127 [0.00891, 0.0176] | 0.0335 [0.0258, 0.0464] |
| phantom | 0.117 [0.0785, 0.205] | 0.122 [0.104, 0.26] |
| collision | 8.41e-05 [7.36e-05, 9.87e-05] | 0.000237 [0.000193, 0.000285] |
| hidden | not reached | not reached |

## Efficiency against the observer told the exact state

| family | 1M | 5M |
|---|---|---|
| teleport | 0.0664 [0.053, 0.0899] | 0.0847 [0.0658, 0.121] |
| deflect | 0.00627 [0.0044, 0.0087] | 0.0166 [0.0127, 0.0229] |
| phantom | 0.0516 [0.0346, 0.0902] | 0.0536 [0.0459, 0.115] |
| collision | 5.56e-05 [4.86e-05, 6.52e-05] | 0.000157 [0.000127, 0.000189] |
| hidden | not reached | not reached |

## Yes/no detection: magnitude detected half the time above false alarms, at one lawful frame in a thousand

| family | unit | told the state | 1M | 5M |
|---|---|---|---|---|
| teleport | px | 0.5 | 0.85 | 0.75 |
| deflect | deg | 5.1 | 39 | 31 |
| phantom | px | 0.25 | 0.47 | 0.41 |
| collision | restitution error | 0.1 | not reached | not reached |
| hidden | px | 0.53 | not reached | not reached |

## Hidden time where the 2AFC falls through chance, frames

| family | measure | 1M | 5M | 1M, 8-frame window |
|---|---|---|---|---|
| vanish | 3 px of the reveal | 2.6 [2.4, 2.9] | 8.9 [7.4, 10.0] | below 1 |
| vanish | full emergence | below 1 | 3.1 [2.9, 3.4] | below 1 |
| vanish | trace halves, per frame | 3.9 [3.7, 5.5] | 4.0 [3.5, 6.1] | 3.0 [2.4, 3.5] |
| rebound | 3 px of the reveal | below 6 | 9.3 [below 6, 10.8] | below 6 |
| rebound | full emergence | below 6 | below 6 | below 6 |
| rebound | trace halves, per frame | 17.9 [16.0, 19.4] | 25.3 [22.1, 28.7] | 7.7 [7.4, 9.6] |

The same crossing for the observers, which have no learned state:

| family | window | ideal | causal ideal | dead reckoning with walls | wall-blind dead reckoning | frozen at last seen | no memory |
|---|---|---|---|---|---|---|---|
| vanish | 3 px | past 48 | past 48 | past 48 | past 48 | below 1 | below 1 |
| vanish | full | past 48 | past 48 | past 48 | past 48 | below 1 | below 1 |
| rebound | 3 px | past 48 | past 48 | past 48 | below 6 | below 6 | below 6 |
| rebound | full | past 48 | past 48 | past 48 | below 6 | below 6 | below 6 |

## The same state read two ways, scored by the same likelihood: crossing in frames

| family | scored by | 1M | 5M |
|---|---|---|---|
| vanish | head, 3 px | below 4 | 8.9 [7.3, 10.0] |
| vanish | head, full | below 4 | below 4 |
| vanish | head's mean at the world's noise, 3 px | below 4 | 10.0 [8.5, 11.2] |
| vanish | head's mean at the world's noise, full | below 4 | 4.3 [below 4, 6.1] |
| vanish | ridge to the next frame, 3 px | below 4 | 9.9 [8.3, 11.0] |
| vanish | ridge to the next frame, full | below 4 | 4.5 [below 4, 6.2] |
| vanish | ridge to visible pixels, through decoded slabs, 3 px | below 4 | 12.4 [10.7, 14.3] |
| vanish | ridge to visible pixels, through decoded slabs, full | below 4 | 10.2 [8.7, 12.6] |
| vanish | linear read through decoded slabs, 3 px | 4.9 [below 4, 6.3] | 21.1 [15.7, 25.7] |
| vanish | linear read through decoded slabs, full | below 4 | 21.8 [18.4, 24.5] |
| vanish | head's mean minus head, 3 px | +0.0 [+0.0, +0.0] | +1.1 [+0.6, +1.7] |
| vanish | head's mean minus head, full | +0.0 [+0.0, +0.0] | +4.3 [+0.0, +6.1] |
| vanish | ridge to the next frame minus head, 3 px | +0.0 [+0.0, +0.0] | +1.0 [+0.4, +1.6] |
| vanish | ridge to the next frame minus head, full | +0.0 [+0.0, +0.0] | +4.5 [+0.0, +6.2] |
| vanish | ridge to visible pixels minus ridge to the next frame, 3 px | +0.0 [+0.0, +0.0] | +2.5 [+1.5, +3.9] |
| vanish | ridge to visible pixels minus ridge to the next frame, full | +0.0 [+0.0, +0.0] | +5.7 [+4.2, +10.6] |
| vanish | linear read minus ridge to visible pixels, 3 px | +4.9 [+0.0, +6.3] | +8.8 [+4.0, +12.7] |
| vanish | linear read minus ridge to visible pixels, full | +0.0 [+0.0, +0.0] | +11.6 [+9.0, +13.4] |
| vanish | linear read minus head, 3 px | +4.9 [+0.0, +6.3] | +12.2 [+7.6, +16.2] |
| vanish | linear read minus head, full | +0.0 [+0.0, +0.0] | +21.8 [+18.4, +24.5] |
| rebound | head, 3 px | below 4 | 9.3 [below 4, 10.9] |
| rebound | head, full | below 4 | below 4 |
| rebound | head's mean at the world's noise, 3 px | below 4 | 12.5 [9.9, 16.8] |
| rebound | head's mean at the world's noise, full | below 4 | below 4 |
| rebound | ridge to the next frame, 3 px | below 4 | 12.9 [10.1, 16.9] |
| rebound | ridge to the next frame, full | below 4 | below 4 |
| rebound | ridge to visible pixels, through decoded slabs, 3 px | below 4 | 18.9 [16.6, 21.2] |
| rebound | ridge to visible pixels, through decoded slabs, full | below 4 | below 4 [below 4, 10.8] |
| rebound | linear read through decoded slabs, 3 px | below 4 | 26.2 [22.4, 31.1] |
| rebound | linear read through decoded slabs, full | below 4 | 26.5 [23.2, 33.5] |
| rebound | head's mean minus head, 3 px | +0.0 [+0.0, +0.0] | +3.3 [+1.1, +16.2] |
| rebound | head's mean minus head, full | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| rebound | ridge to the next frame minus head, 3 px | +0.0 [+0.0, +0.0] | +3.6 [+1.3, +16.3] |
| rebound | ridge to the next frame minus head, full | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| rebound | ridge to visible pixels minus ridge to the next frame, 3 px | +0.0 [+0.0, +0.0] | +6.0 [+2.8, +8.9] |
| rebound | ridge to visible pixels minus ridge to the next frame, full | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +10.8] |
| rebound | linear read minus ridge to visible pixels, 3 px | +0.0 [+0.0, +0.0] | +7.3 [+3.9, +11.5] |
| rebound | linear read minus ridge to visible pixels, full | +0.0 [+0.0, +0.0] | +26.5 [+18.0, +33.4] |
| rebound | linear read minus head, 3 px | +0.0 [+0.0, +0.0] | +17.0 [+13.4, +30.6] |
| rebound | linear read minus head, full | +0.0 [+0.0, +0.0] | +26.5 [+23.2, +33.5] |

## Taking directions out of the state while hidden: the head's 2AFC

| family | size | D | intact | read's top 4 removed | random 4 removed | read's top 16 removed | random 16 removed |
|---|---|---|---|---|---|---|---|
| vanish | 1M | 8 | 0.111 | 0.009 | 0.056 | 0.016 | 0.018 |
| vanish | 1M | 16 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 |
| vanish | 1M | 24 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| vanish | 5M | 8 | 0.562 | 0.065 | 0.474 | 0.062 | 0.294 |
| vanish | 5M | 16 | 0.112 | 0.000 | 0.055 | 0.000 | 0.021 |
| vanish | 5M | 24 | 0.013 | 0.000 | 0.005 | 0.000 | 0.000 |
| rebound | 1M | 8 | 0.089 | 0.000 | 0.052 | 0.004 | 0.013 |
| rebound | 1M | 16 | 0.031 | 0.000 | 0.004 | 0.000 | 0.000 |
| rebound | 1M | 24 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |
| rebound | 5M | 8 | 0.544 | 0.042 | 0.406 | 0.039 | 0.195 |
| rebound | 5M | 16 | 0.336 | 0.008 | 0.214 | 0.005 | 0.044 |
| rebound | 5M | 24 | 0.068 | 0.000 | 0.029 | 0.000 | 0.008 |

Share of the state's variance on hidden frames that those directions carry:

| size | directions | how many | share |
|---|---|---|---|
| 1M | read | 4 | 0.004 |
| 1M | read | 16 | 0.007 |
| 1M | random | 4 | 0.029 |
| 1M | random | 16 | 0.138 |
| 5M | read | 4 | 0.003 |
| 5M | read | 16 | 0.006 |
| 5M | random | 4 | 0.010 |
| 5M | random | 16 | 0.049 |

## Decoded position while hidden: linear read straight to position (px, median over scenes, mean over seeds)

### vanish, read 100% of the way through

| D | 1M | 5M | last seen | straight line | slab middle |
|---|---|---|---|---|---|
| 4 | 2.19 | 1.54 | 3.16 | 0.00 | 1.62 |
| 6 | 2.33 | 1.55 | 4.24 | 0.00 | 2.26 |
| 8 | 2.58 | 1.56 | 5.55 | 0.00 | 2.76 |
| 12 | 2.45 | 1.53 | 5.90 | 0.00 | 3.01 |
| 16 | 2.61 | 1.75 | 6.59 | 0.00 | 3.36 |
| 20 | 2.77 | 1.84 | 6.76 | 0.00 | 3.34 |
| 24 | 2.81 | 1.90 | 6.92 | 0.00 | 3.46 |
| 28 | 3.39 | 1.91 | 7.01 | 0.00 | 3.59 |
| 32 | 3.80 | 2.16 | 7.52 | 0.00 | 3.82 |
| 40 | 4.58 | 2.64 | 7.82 | 0.00 | 3.99 |
| 48 | 4.93 | 2.97 | 8.27 | 0.00 | 4.28 |

### rebound, read 75% of the way through

| D | 1M | 5M | last seen | straight line | slab middle |
|---|---|---|---|---|---|
| 6 | 1.39 | 1.26 | 1.07 | 2.07 | 0.41 |
| 8 | 1.32 | 1.06 | 1.08 | 3.24 | 0.46 |
| 12 | 1.43 | 1.23 | 1.74 | 4.66 | 0.66 |
| 16 | 1.40 | 1.19 | 2.46 | 5.73 | 0.71 |
| 20 | 1.34 | 1.12 | 2.79 | 6.10 | 0.79 |
| 24 | 1.50 | 1.25 | 3.41 | 7.13 | 0.82 |
| 28 | 1.54 | 1.27 | 3.83 | 7.80 | 0.96 |
| 32 | 1.60 | 1.19 | 4.06 | 8.03 | 0.85 |
| 40 | 1.84 | 1.36 | 4.33 | 8.02 | 0.90 |
| 48 | 2.31 | 1.53 | 3.29 | 6.65 | 0.84 |

## Following the unseen bounce: motion along x, read against truth

| size | D | leg | direction right | slope |
|---|---|---|---|---|
| 1M | 16 | outward | 0.94 | 0.80 |
| 1M | 16 | return | 0.96 | 0.70 |
| 1M | 28 | outward | 0.95 | 0.73 |
| 1M | 28 | return | 0.98 | 0.60 |
| 5M | 16 | outward | 0.97 | 0.91 |
| 5M | 16 | return | 0.99 | 0.96 |
| 5M | 28 | outward | 1.00 | 0.84 |
| 5M | 28 | return | 0.99 | 0.85 |

## 5m-s0, watched while the analysis was built and so left out of its group: crossing in frames

| family | 3 px | full emergence |
|---|---|---|
| vanish | 11.0 | 3.5 |
| rebound | 12.5 | below 6 |

## Speed bands, vanish: crossing in frames

| band, px per frame | 1M | 5M |
|---|---|---|
| 0.15 to 0.4 | below 4 | below 4 |
| 0.4 to 0.6 | below 4 | 9.5 |
| 0.6 to 1.3 | 3.1 | 12.0 |

## Restitution where the model's likelihood peaks (the world's is 1)

| size | restitution | by seed |
|---|---|---|
| 1M | 0.47 | 0.47, 0.44, 0.52, 0.44 |
| 5M | 0.74 | 0.74, 0.74 |

## A ball passing through a wall: pairs favouring the lawful twin

| size | proportion |
|---|---|
| 1M | 1.00 |
| 5M | 1.00 |

## Calibration on held-out lawful frames from the training generator: measured rate where one in a thousand was promised

| size | all | quiet | edge | contact | reveal |
|---|---|---|---|---|---|
| 1M | 0.00087 | 0.00045 | 0.00161 | 0.00086 | 0.00127 |
| 5M | 0.00093 | 0.00063 | 0.00144 | 0.00107 | 0.00147 |

## The same, by time since the start of a clip (training clips are 64 frames)

| size | 4-96 | 96-200 | 200-400 | 400-800 | 800-1200 |
|---|---|---|---|---|---|
| 1M | 0.00111 | 0.00092 | 0.00097 | 0.00090 | 0.00088 |
| 5M | 0.00098 | 0.00067 | 0.00087 | 0.00085 | 0.00079 |

