# Results

Written by `python -m sm.figures` from `results/summary.json`. Brackets are 95% hierarchical bootstrap intervals over seeds and scenes.

## Seeds

| size | seeds |
|---|---|
| 1M | 1m-s0, 1m-s1, 1m-s2, 1m-s3 |
| 5M | 5m-s0 |

## Thresholds: magnitude at 75% correct in 2AFC

| family | unit | told the state | causal ideal | dead reckoning with walls | wall-blind | no memory | 1M | 5M |
|---|---|---|---|---|---|---|---|---|
| teleport | px | 0.04 | 0.065 | 0.068 | 0.068 | 0.068 | 0.16 [0.13, 0.18] | 0.14 [0.12, 0.16] |
| deflect | deg | 0.44 | 0.63 | 7.1 | 7.1 | 7.1 | 5.6 [4.8, 6.6] | 3.7 [3.2, 4.3] |
| phantom | px | 0.022 | 0.032 | 0.27 | 4.5 | 4.5 | 0.095 [0.072, 0.12] | 0.098 [0.094, 0.099] |
| collision | restitution error | 0.0092 | 0.011 | not reached | not reached | not reached | 1.2 [1.1, 1.3] | 0.66 [0.6, 0.72] |
| hidden | px | below 0.05 | below 0.05 | 1.5 | 1.9 | not reached | 2.3 [1.9, not reached] | 1.2 [0.97, 1.4] |

## Efficiency against the causal ideal: (its threshold / the model's) squared

| family | 1M | 5M |
|---|---|---|
| teleport | 0.169 [0.135, 0.229] | 0.198 [0.161, 0.274] |
| deflect | 0.0127 [0.00902, 0.017] | 0.0284 [0.0217, 0.0386] |
| phantom | 0.117 [0.0776, 0.201] | 0.11 [0.106, 0.12] |
| collision | 8.41e-05 [7.34e-05, 9.83e-05] | 0.000296 [0.00025, 0.000353] |
| hidden | not reached | not reached |

## Efficiency against the observer told the exact state

| family | 1M | 5M |
|---|---|---|
| teleport | 0.0664 [0.053, 0.0899] | 0.0777 [0.0632, 0.107] |
| deflect | 0.00627 [0.00445, 0.0084] | 0.014 [0.0107, 0.0191] |
| phantom | 0.0516 [0.0342, 0.0886] | 0.0487 [0.0468, 0.053] |
| collision | 5.56e-05 [4.85e-05, 6.49e-05] | 0.000195 [0.000165, 0.000234] |
| hidden | not reached | not reached |

## Yes/no detection: magnitude detected half the time above false alarms, at one lawful frame in a thousand

| family | unit | told the state | 1M | 5M |
|---|---|---|---|---|
| teleport | px | 0.5 | 0.85 | 0.74 |
| deflect | deg | 5.1 | 39 | 32 |
| phantom | px | 0.25 | 0.47 | 0.4 |
| collision | restitution error | 0.1 | not reached | not reached |
| hidden | px | 0.53 | not reached | not reached |

## Hidden time where the 2AFC falls through chance, frames

| family | measure | 1M | 5M |
|---|---|---|---|
| vanish | 3 px of the reveal | 2.6 [2.3, 2.9] | 11.0 [10.1, 12.2] |
| vanish | full emergence | below 1 | 3.5 [3.1, 4.1] |
| vanish | trace halves, per frame | 3.9 [3.6, 5.2] | 5.8 [3.9, 7.3] |
| rebound | 3 px of the reveal | below 6 | 12.5 [11.3, 14.1] |
| rebound | full emergence | below 6 | below 6 |
| rebound | trace halves, per frame | 17.9 [16.0, 19.5] | 23.9 [19.8, 25.6] |

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
| vanish | head, 3 px | below 4 | 11.0 [10.1, 12.2] |
| vanish | head, full | below 4 | below 4 [below 4, 4.0] |
| vanish | head's mean at the world's noise, 3 px | below 4 | 12.2 [11.2, 13.0] |
| vanish | head's mean at the world's noise, full | below 4 | 5.5 [5.0, 6.4] |
| vanish | ridge to the next frame, 3 px | below 4 | 11.9 [10.9, 12.8] |
| vanish | ridge to the next frame, full | below 4 | 5.1 [4.5, 6.1] |
| vanish | ridge to visible pixels, through decoded slabs, 3 px | below 4 | 13.6 [12.4, 14.6] |
| vanish | ridge to visible pixels, through decoded slabs, full | below 4 | 9.3 [8.1, 10.2] |
| vanish | linear read through decoded slabs, 3 px | 4.9 [below 4, 6.3] | 21.3 [20.2, 22.3] |
| vanish | linear read through decoded slabs, full | below 4 | 20.1 [15.9, 21.2] |
| vanish | head's mean minus head, 3 px | +0.0 [+0.0, +0.0] | +1.2 [+0.5, +1.9] |
| vanish | head's mean minus head, full | +0.0 [+0.0, +0.0] | +5.5 [+1.9, +6.4] |
| vanish | ridge to the next frame minus head, 3 px | +0.0 [+0.0, +0.0] | +0.9 [+0.2, +1.6] |
| vanish | ridge to the next frame minus head, full | +0.0 [+0.0, +0.0] | +5.1 [+1.3, +6.0] |
| vanish | ridge to visible pixels minus ridge to the next frame, 3 px | +0.0 [+0.0, +0.0] | +1.7 [+0.8, +2.8] |
| vanish | ridge to visible pixels minus ridge to the next frame, full | +0.0 [+0.0, +0.0] | +4.2 [+2.7, +5.3] |
| vanish | linear read minus ridge to visible pixels, 3 px | +4.9 [+0.0, +6.3] | +7.7 [+6.2, +9.1] |
| vanish | linear read minus ridge to visible pixels, full | +0.0 [+0.0, +0.0] | +10.8 [+6.9, +12.3] |
| vanish | linear read minus head, 3 px | +4.9 [+0.0, +6.3] | +10.3 [+8.5, +11.7] |
| vanish | linear read minus head, full | +0.0 [+0.0, +0.0] | +20.1 [+15.8, +21.2] |
| rebound | head, 3 px | below 4 | 12.5 [11.3, 14.0] |
| rebound | head, full | below 4 | below 4 |
| rebound | head's mean at the world's noise, 3 px | below 4 | 15.2 [13.7, 16.9] |
| rebound | head's mean at the world's noise, full | below 4 | below 4 [below 4, 6.8] |
| rebound | ridge to the next frame, 3 px | below 4 | 15.0 [13.5, 16.8] |
| rebound | ridge to the next frame, full | below 4 | below 4 [below 4, 6.2] |
| rebound | ridge to visible pixels, through decoded slabs, 3 px | below 4 | 18.2 [13.8, 20.8] |
| rebound | ridge to visible pixels, through decoded slabs, full | below 4 | below 4 [below 4, 6.0] |
| rebound | linear read through decoded slabs, 3 px | below 4 | 25.6 [24.2, 26.9] |
| rebound | linear read through decoded slabs, full | below 4 | 23.5 [22.6, 24.7] |
| rebound | head's mean minus head, 3 px | +0.0 [+0.0, +0.0] | +2.7 [+1.3, +5.2] |
| rebound | head's mean minus head, full | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +6.8] |
| rebound | ridge to the next frame minus head, 3 px | +0.0 [+0.0, +0.0] | +2.5 [+1.2, +5.1] |
| rebound | ridge to the next frame minus head, full | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +6.2] |
| rebound | ridge to visible pixels minus ridge to the next frame, 3 px | +0.0 [+0.0, +0.0] | +3.2 [-0.1, +6.0] |
| rebound | ridge to visible pixels minus ridge to the next frame, full | +0.0 [+0.0, +0.0] | +0.0 [-6.2, +0.0] |
| rebound | linear read minus ridge to visible pixels, 3 px | +0.0 [+0.0, +0.0] | +7.4 [+4.6, +11.8] |
| rebound | linear read minus ridge to visible pixels, full | +0.0 [+0.0, +0.0] | +23.5 [+17.7, +24.7] |
| rebound | linear read minus head, 3 px | +0.0 [+0.0, +0.0] | +13.1 [+11.2, +14.9] |
| rebound | linear read minus head, full | +0.0 [+0.0, +0.0] | +23.5 [+22.6, +24.7] |

## Taking directions out of the state while hidden: the head's 2AFC

| family | size | D | intact | read's top 4 removed | random 4 removed | read's top 16 removed | random 16 removed |
|---|---|---|---|---|---|---|---|
| vanish | 1M | 8 | 0.111 | 0.009 | 0.056 | 0.016 | 0.018 |
| vanish | 1M | 16 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 |
| vanish | 1M | 24 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| vanish | 5M | 8 | 0.661 | 0.188 | 0.526 | 0.146 | 0.177 |
| vanish | 5M | 16 | 0.172 | 0.016 | 0.130 | 0.010 | 0.010 |
| vanish | 5M | 24 | 0.016 | 0.005 | 0.016 | 0.000 | 0.005 |
| rebound | 1M | 8 | 0.089 | 0.000 | 0.052 | 0.004 | 0.013 |
| rebound | 1M | 16 | 0.031 | 0.000 | 0.004 | 0.000 | 0.000 |
| rebound | 1M | 24 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |
| rebound | 5M | 8 | 0.755 | 0.073 | 0.651 | 0.062 | 0.240 |
| rebound | 5M | 16 | 0.391 | 0.026 | 0.271 | 0.016 | 0.016 |
| rebound | 5M | 24 | 0.146 | 0.000 | 0.057 | 0.000 | 0.000 |

Share of the state's variance on hidden frames that those directions carry:

| size | directions | how many | share |
|---|---|---|---|
| 1M | read | 4 | 0.004 |
| 1M | read | 16 | 0.007 |
| 1M | random | 4 | 0.029 |
| 1M | random | 16 | 0.138 |
| 5M | read | 4 | 0.003 |
| 5M | read | 16 | 0.007 |
| 5M | random | 4 | 0.018 |
| 5M | random | 16 | 0.062 |

## Decoded position while hidden: linear read straight to position (px, median over scenes, mean over seeds)

### vanish, read 100% of the way through

| D | 1M | 5M | last seen | straight line | slab middle |
|---|---|---|---|---|---|
| 4 | 2.19 | 1.52 | 3.16 | 0.00 | 1.62 |
| 6 | 2.33 | 1.54 | 4.24 | 0.00 | 2.26 |
| 8 | 2.58 | 1.47 | 5.55 | 0.00 | 2.76 |
| 12 | 2.45 | 1.54 | 5.90 | 0.00 | 3.01 |
| 16 | 2.61 | 1.69 | 6.59 | 0.00 | 3.36 |
| 20 | 2.77 | 1.63 | 6.76 | 0.00 | 3.34 |
| 24 | 2.81 | 1.62 | 6.92 | 0.00 | 3.46 |
| 28 | 3.39 | 1.73 | 7.01 | 0.00 | 3.59 |
| 32 | 3.80 | 1.93 | 7.52 | 0.00 | 3.82 |
| 40 | 4.58 | 2.20 | 7.82 | 0.00 | 3.99 |
| 48 | 4.93 | 2.47 | 8.27 | 0.00 | 4.28 |

### rebound, read 75% of the way through

| D | 1M | 5M | last seen | straight line | slab middle |
|---|---|---|---|---|---|
| 6 | 1.39 | 1.23 | 1.07 | 2.07 | 0.41 |
| 8 | 1.32 | 0.97 | 1.08 | 3.24 | 0.46 |
| 12 | 1.43 | 1.25 | 1.74 | 4.66 | 0.66 |
| 16 | 1.40 | 1.24 | 2.46 | 5.73 | 0.71 |
| 20 | 1.34 | 1.14 | 2.79 | 6.10 | 0.79 |
| 24 | 1.50 | 1.13 | 3.41 | 7.13 | 0.82 |
| 28 | 1.54 | 1.34 | 3.83 | 7.80 | 0.96 |
| 32 | 1.60 | 1.17 | 4.06 | 8.03 | 0.85 |
| 40 | 1.84 | 1.38 | 4.33 | 8.02 | 0.90 |
| 48 | 2.31 | 1.48 | 3.29 | 6.65 | 0.84 |

## Following the unseen bounce: motion along x, read against truth

| size | D | leg | direction right | slope |
|---|---|---|---|---|
| 1M | 16 | outward | 0.94 | 0.80 |
| 1M | 16 | return | 0.96 | 0.70 |
| 1M | 28 | outward | 0.95 | 0.73 |
| 1M | 28 | return | 0.98 | 0.60 |
| 5M | 16 | outward | 0.98 | 0.89 |
| 5M | 16 | return | 1.00 | 0.93 |
| 5M | 28 | outward | 0.98 | 0.84 |
| 5M | 28 | return | 0.99 | 0.82 |

## Speed bands, vanish: crossing in frames

| band, px per frame | 1M | 5M |
|---|---|---|
| 0.15 to 0.4 | below 4 | below 4 |
| 0.4 to 0.6 | below 4 | 12.5 |
| 0.6 to 1.3 | 3.1 | 12.0 |

## Restitution where the model's likelihood peaks (the world's is 1)

| size | restitution | by seed |
|---|---|---|
| 1M | 0.47 | 0.47, 0.44, 0.52, 0.44 |
| 5M | 0.77 | 0.77 |

## A ball passing through a wall: pairs favouring the lawful twin

| size | proportion |
|---|---|
| 1M | 1.00 |
| 5M | 1.00 |

## Calibration on held-out lawful frames from the training generator: measured rate where one in a thousand was promised

| size | all | quiet | edge | contact | reveal |
|---|---|---|---|---|---|
| 1M | 0.00087 | 0.00045 | 0.00161 | 0.00086 | 0.00127 |
| 5M | 0.00088 | 0.00062 | 0.00132 | 0.00095 | 0.00147 |

## The same, by time since the start of a clip (training clips are 64 frames)

| size | 4-96 | 96-200 | 200-400 | 400-800 | 800-1200 |
|---|---|---|---|---|---|
| 1M | 0.00111 | 0.00092 | 0.00097 | 0.00090 | 0.00088 |
| 5M | 0.00081 | 0.00073 | 0.00087 | 0.00086 | 0.00074 |

