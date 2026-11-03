# Results

Written by `python -m sm.figures` from `results/summary.json`. Brackets are 95% hierarchical bootstrap intervals over seeds and scenes.

## Seeds

| size | seeds |
|---|---|
| 1M | 1m-s0, 1m-s1, 1m-s2, 1m-s3 |
| 5M | 5m-s1, 5m-s2 |
| 20M | 20m-s0 |
| 1M, 8-frame window | 1m-window8-s0 |
| 5M, 10000 steps | 5m-short-s0 |

## Thresholds: magnitude at 75% correct in 2AFC

| family | unit | told the state | causal ideal | dead reckoning with walls | wall-blind | no memory | 1M | 5M | 20M |
|---|---|---|---|---|---|---|---|---|---|
| teleport | px | 0.04 | 0.065 | 0.068 | 0.068 | 0.068 | 0.16 [0.13, 0.18] | 0.14 [0.12, 0.16] | 0.14 [0.12, 0.15] |
| deflect | deg | 0.44 | 0.63 | 7.1 | 7.1 | 7.1 | 5.6 [4.8, 6.7] | 3.4 [2.9, 3.9] | 3.5 [2.9, 4.1] |
| phantom | px | 0.022 | 0.032 | 0.27 | 4.5 | 4.5 | 0.095 [0.071, 0.12] | 0.093 [0.065, 0.1] | 0.099 [0.098, 0.1] |
| collision | restitution error | 0.0092 | 0.011 | not reached | not reached | not reached | 1.2 [1.1, 1.3] | 0.74 [0.66, 0.8] | 0.57 [0.51, 0.61] |
| hidden | px | below 0.05 | below 0.05 | 1.5 | 1.9 | not reached | 2.3 [1.9, not reached] | 1.1 [0.9, 1.4] | 1.3 [1.1, 1.9] |

## Efficiency against the causal ideal: (its threshold / the model's) squared

| family | 1M | 5M | 20M |
|---|---|---|---|
| teleport | 0.169 [0.135, 0.229] | 0.216 [0.168, 0.31] | 0.227 [0.182, 0.311] |
| deflect | 0.0127 [0.00892, 0.0175] | 0.0335 [0.0256, 0.0457] | 0.0319 [0.0233, 0.0465] |
| phantom | 0.117 [0.0769, 0.207] | 0.122 [0.105, 0.251] | 0.106 [0.103, 0.11] |
| collision | 8.41e-05 [7.25e-05, 0.0001] | 0.000237 [0.000198, 0.00029] | 0.000396 [0.00034, 0.000494] |
| hidden | not reached | not reached | not reached |

## Efficiency against the observer told the exact state

| family | 1M | 5M | 20M |
|---|---|---|---|
| teleport | 0.0664 [0.053, 0.0899] | 0.0847 [0.0658, 0.121] | 0.089 [0.0712, 0.122] |
| deflect | 0.00627 [0.0044, 0.00864] | 0.0166 [0.0126, 0.0226] | 0.0157 [0.0115, 0.0229] |
| phantom | 0.0516 [0.0339, 0.0913] | 0.0536 [0.0461, 0.111] | 0.0469 [0.0453, 0.0486] |
| collision | 5.56e-05 [4.8e-05, 6.63e-05] | 0.000157 [0.000131, 0.000192] | 0.000262 [0.000225, 0.000327] |
| hidden | not reached | not reached | not reached |

## Yes/no detection: magnitude detected half the time above false alarms, at one lawful frame in a thousand

| family | unit | told the state | 1M | 5M | 20M |
|---|---|---|---|---|---|
| teleport | px | 0.5 | 0.85 | 0.75 | 0.74 |
| deflect | deg | 5.1 | 39 | 31 | 31 |
| phantom | px | 0.25 | 0.47 | 0.41 | 0.42 |
| collision | restitution error | 0.1 | not reached | not reached | not reached |
| hidden | px | 0.53 | not reached | not reached | not reached |

## Hidden time where the 2AFC falls through chance, frames

| family | measure | 1M | 5M | 20M | 1M, 8-frame window | 5M, 10000 steps |
|---|---|---|---|---|---|---|
| vanish | 3 px of the reveal | 2.6 [2.4, 2.9] | 8.9 [7.3, 10.0] | 9.8 [9.2, 10.3] | below 1 | 2.9 [2.6, 3.1] |
| vanish | full emergence | below 1 | 3.1 [2.8, 3.4] | 3.5 [3.2, 4.0] | below 1 | below 1 |
| vanish | trace halves, per frame | 3.9 [3.6, 5.6] | 4.0 [3.5, 5.7] | 4.2 [3.5, 6.7] | 3.0 [2.4, 3.5] | 3.9 [3.6, 5.0] |
| rebound | 3 px of the reveal | below 6 | 9.3 [below 6, 10.7] | 10.6 [9.4, 12.1] | below 6 | below 6 |
| rebound | full emergence | below 6 | below 6 | below 6 | below 6 | below 6 |
| rebound | trace halves, per frame | 17.9 [16.3, 19.4] | 25.3 [21.6, 28.5] | 20.9 [18.4, 23.6] | 7.7 [7.4, 9.4] | 17.9 [15.9, 18.9] |

The same crossing for the observers, which have no learned state:

| family | window | ideal | causal ideal | dead reckoning with walls | wall-blind dead reckoning | frozen at last seen | no memory |
|---|---|---|---|---|---|---|---|
| vanish | 3 px | past 48 | past 48 | past 48 | past 48 | below 1 | below 1 |
| vanish | full | past 48 | past 48 | past 48 | past 48 | below 1 | below 1 |
| rebound | 3 px | past 48 | past 48 | past 48 | below 6 | below 6 | below 6 |
| rebound | full | past 48 | past 48 | past 48 | below 6 | below 6 | below 6 |

## The same state read two ways, scored by the same likelihood: crossing in frames

| family | scored by | 1M | 5M | 20M |
|---|---|---|---|---|
| vanish | head, 3 px | below 4 | 8.9 [7.4, 10.1] | 9.8 [9.2, 10.3] |
| vanish | head, full | below 4 | below 4 | below 4 |
| vanish | head's mean at the world's noise, 3 px | below 4 | 10.0 [8.7, 11.1] | 11.2 [10.3, 12.2] |
| vanish | head's mean at the world's noise, full | below 4 | 4.3 [below 4, 5.9] | 5.4 [4.4, 6.4] |
| vanish | ridge to the next frame, 3 px | below 4 | 9.9 [8.3, 11.1] | 10.5 [9.8, 11.3] |
| vanish | ridge to the next frame, full | below 4 | 4.5 [below 4, 6.1] | 5.2 [4.4, 6.1] |
| vanish | ridge to visible pixels, through decoded slabs, 3 px | below 4 | 12.4 [10.8, 14.3] | 13.0 [12.0, 13.7] |
| vanish | ridge to visible pixels, through decoded slabs, full | below 4 | 10.2 [8.8, 12.5] | 8.8 [7.9, 9.4] |
| vanish | linear read through decoded slabs, 3 px | 4.9 [below 4, 6.3] | 21.1 [15.6, 25.6] | 19.5 [18.4, 20.9] |
| vanish | linear read through decoded slabs, full | below 4 | 21.8 [18.5, 24.5] | 16.0 [14.7, 17.0] |
| vanish | head's mean minus head, 3 px | +0.0 [+0.0, +0.0] | +1.1 [+0.6, +1.8] | +1.4 [+0.7, +2.3] |
| vanish | head's mean minus head, full | +0.0 [+0.0, +0.0] | +4.3 [+0.0, +5.9] | +5.4 [+4.3, +6.4] |
| vanish | ridge to the next frame minus head, 3 px | +0.0 [+0.0, +0.0] | +1.0 [+0.4, +1.6] | +0.7 [+0.2, +1.4] |
| vanish | ridge to the next frame minus head, full | +0.0 [+0.0, +0.0] | +4.5 [+0.0, +6.1] | +5.2 [+4.4, +6.1] |
| vanish | ridge to visible pixels minus ridge to the next frame, 3 px | +0.0 [+0.0, +0.0] | +2.5 [+1.5, +3.6] | +2.5 [+1.6, +3.2] |
| vanish | ridge to visible pixels minus ridge to the next frame, full | +0.0 [+0.0, +0.0] | +5.7 [+4.0, +10.7] | +3.6 [+2.2, +4.5] |
| vanish | linear read minus ridge to visible pixels, 3 px | +4.9 [+0.0, +6.3] | +8.8 [+4.0, +12.6] | +6.5 [+5.2, +8.1] |
| vanish | linear read minus ridge to visible pixels, full | +0.0 [+0.0, +0.0] | +11.6 [+8.9, +13.5] | +7.2 [+5.9, +8.4] |
| vanish | linear read minus head, 3 px | +4.9 [+0.0, +6.3] | +12.2 [+7.4, +16.2] | +9.7 [+8.4, +11.2] |
| vanish | linear read minus head, full | +0.0 [+0.0, +0.0] | +21.8 [+18.5, +24.5] | +16.0 [+14.6, +17.0] |
| rebound | head, 3 px | below 4 | 9.3 [below 4, 10.7] | 10.6 [9.4, 12.1] |
| rebound | head, full | below 4 | below 4 | below 4 |
| rebound | head's mean at the world's noise, 3 px | below 4 | 12.5 [10.1, 16.9] | 12.8 [11.4, 13.8] |
| rebound | head's mean at the world's noise, full | below 4 | below 4 | below 4 |
| rebound | ridge to the next frame, 3 px | below 4 | 12.9 [10.3, 16.9] | 12.3 [11.0, 13.4] |
| rebound | ridge to the next frame, full | below 4 | below 4 | below 4 |
| rebound | ridge to visible pixels, through decoded slabs, 3 px | below 4 | 18.9 [16.4, 21.1] | 14.2 [11.5, 16.6] |
| rebound | ridge to visible pixels, through decoded slabs, full | below 4 | below 4 [below 4, 13.9] | below 4 |
| rebound | linear read through decoded slabs, 3 px | below 4 | 26.2 [22.3, 30.9] | 22.6 [21.8, 23.5] |
| rebound | linear read through decoded slabs, full | below 4 | 26.5 [23.3, 33.5] | 20.2 [17.8, 21.2] |
| rebound | head's mean minus head, 3 px | +0.0 [+0.0, +0.0] | +3.3 [+1.1, +16.2] | +2.3 [+1.2, +3.3] |
| rebound | head's mean minus head, full | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| rebound | ridge to the next frame minus head, 3 px | +0.0 [+0.0, +0.0] | +3.6 [+1.4, +16.3] | +1.7 [+0.8, +2.7] |
| rebound | ridge to the next frame minus head, full | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| rebound | ridge to visible pixels minus ridge to the next frame, 3 px | +0.0 [+0.0, +0.0] | +6.0 [+3.1, +8.8] | +1.9 [-0.5, +4.9] |
| rebound | ridge to visible pixels minus ridge to the next frame, full | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +13.9] | +0.0 [+0.0, +0.0] |
| rebound | linear read minus ridge to visible pixels, 3 px | +0.0 [+0.0, +0.0] | +7.3 [+4.0, +11.4] | +8.4 [+5.7, +11.3] |
| rebound | linear read minus ridge to visible pixels, full | +0.0 [+0.0, +0.0] | +26.5 [+17.0, +33.5] | +20.2 [+17.7, +21.2] |
| rebound | linear read minus head, 3 px | +0.0 [+0.0, +0.0] | +17.0 [+13.1, +30.3] | +12.0 [+10.5, +13.6] |
| rebound | linear read minus head, full | +0.0 [+0.0, +0.0] | +26.5 [+23.3, +33.5] | +20.2 [+17.8, +21.2] |

## Taking directions out of the state while hidden: the head's 2AFC

| family | size | D | intact | read's top 4 removed | random 4 removed | read's top 16 removed | random 16 removed |
|---|---|---|---|---|---|---|---|
| vanish | 1M | 8 | 0.111 | 0.009 | 0.056 | 0.016 | 0.018 |
| vanish | 1M | 16 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 |
| vanish | 1M | 24 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| vanish | 5M | 8 | 0.562 | 0.065 | 0.474 | 0.062 | 0.294 |
| vanish | 5M | 16 | 0.112 | 0.000 | 0.055 | 0.000 | 0.021 |
| vanish | 5M | 24 | 0.013 | 0.000 | 0.005 | 0.000 | 0.000 |
| vanish | 20M | 8 | 0.661 | 0.307 | 0.505 | 0.240 | 0.266 |
| vanish | 20M | 16 | 0.146 | 0.021 | 0.073 | 0.010 | 0.010 |
| vanish | 20M | 24 | 0.016 | 0.000 | 0.010 | 0.000 | 0.000 |
| rebound | 1M | 8 | 0.089 | 0.000 | 0.052 | 0.004 | 0.013 |
| rebound | 1M | 16 | 0.031 | 0.000 | 0.004 | 0.000 | 0.000 |
| rebound | 1M | 24 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |
| rebound | 5M | 8 | 0.544 | 0.042 | 0.406 | 0.039 | 0.195 |
| rebound | 5M | 16 | 0.336 | 0.008 | 0.214 | 0.005 | 0.044 |
| rebound | 5M | 24 | 0.068 | 0.000 | 0.029 | 0.000 | 0.008 |
| rebound | 20M | 8 | 0.615 | 0.255 | 0.505 | 0.151 | 0.224 |
| rebound | 20M | 16 | 0.260 | 0.031 | 0.130 | 0.016 | 0.016 |
| rebound | 20M | 24 | 0.078 | 0.005 | 0.021 | 0.000 | 0.000 |

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
| 20M | read | 4 | 0.002 |
| 20M | read | 16 | 0.004 |
| 20M | random | 4 | 0.006 |
| 20M | random | 16 | 0.037 |

## Decoded position while hidden: linear read straight to position (px, median over scenes, mean over seeds)

### vanish, read 100% of the way through

| D | 1M | 5M | 20M | last seen | straight line | slab middle |
|---|---|---|---|---|---|---|
| 4 | 2.19 | 1.54 | 1.51 | 3.16 | 0.00 | 1.62 |
| 6 | 2.33 | 1.55 | 1.65 | 4.24 | 0.00 | 2.26 |
| 8 | 2.58 | 1.56 | 1.47 | 5.55 | 0.00 | 2.76 |
| 12 | 2.45 | 1.53 | 1.35 | 5.90 | 0.00 | 3.01 |
| 16 | 2.61 | 1.75 | 1.60 | 6.59 | 0.00 | 3.36 |
| 20 | 2.77 | 1.84 | 1.60 | 6.76 | 0.00 | 3.34 |
| 24 | 2.81 | 1.90 | 1.77 | 6.92 | 0.00 | 3.46 |
| 28 | 3.39 | 1.91 | 1.79 | 7.01 | 0.00 | 3.59 |
| 32 | 3.80 | 2.16 | 1.96 | 7.52 | 0.00 | 3.82 |
| 40 | 4.58 | 2.64 | 2.45 | 7.82 | 0.00 | 3.99 |
| 48 | 4.93 | 2.97 | 3.17 | 8.27 | 0.00 | 4.28 |

### rebound, read 75% of the way through

| D | 1M | 5M | 20M | last seen | straight line | slab middle |
|---|---|---|---|---|---|---|
| 6 | 1.39 | 1.26 | 1.09 | 1.07 | 2.07 | 0.41 |
| 8 | 1.32 | 1.06 | 0.96 | 1.08 | 3.24 | 0.46 |
| 12 | 1.43 | 1.23 | 1.15 | 1.74 | 4.66 | 0.66 |
| 16 | 1.40 | 1.19 | 1.25 | 2.46 | 5.73 | 0.71 |
| 20 | 1.34 | 1.12 | 1.22 | 2.79 | 6.10 | 0.79 |
| 24 | 1.50 | 1.25 | 1.30 | 3.41 | 7.13 | 0.82 |
| 28 | 1.54 | 1.27 | 1.32 | 3.83 | 7.80 | 0.96 |
| 32 | 1.60 | 1.19 | 1.33 | 4.06 | 8.03 | 0.85 |
| 40 | 1.84 | 1.36 | 1.38 | 4.33 | 8.02 | 0.90 |
| 48 | 2.31 | 1.53 | 1.32 | 3.29 | 6.65 | 0.84 |

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
| 20M | 16 | outward | 0.98 | 0.90 |
| 20M | 16 | return | 0.99 | 1.02 |
| 20M | 28 | outward | 0.99 | 0.86 |
| 20M | 28 | return | 0.99 | 0.87 |

## 5m-s0, watched while the analysis was built and so left out of its group: crossing in frames

| family | 3 px | full emergence |
|---|---|---|
| vanish | 11.0 | 3.5 |
| rebound | 12.5 | below 6 |

## Speed bands, vanish: crossing in frames

| band, px per frame | 1M | 5M | 20M |
|---|---|---|---|
| 0.15 to 0.4 | below 4 | below 4 | below 4 |
| 0.4 to 0.6 | below 4 | 9.5 | 10.9 |
| 0.6 to 1.3 | 3.1 | 12.0 | 11.5 |

## Restitution where the model's likelihood peaks (the world's is 1)

| size | restitution | by seed |
|---|---|---|
| 1M | 0.47 | 0.47, 0.44, 0.52, 0.44 |
| 5M | 0.74 | 0.74, 0.74 |
| 20M | 0.84 | 0.84 |

## A ball passing through a wall: pairs favouring the lawful twin

| size | proportion |
|---|---|
| 1M | 1.00 |
| 5M | 1.00 |
| 20M | 1.00 |

## Calibration on held-out lawful frames from the training generator: measured rate where one in a thousand was promised

| size | all | quiet | edge | contact | reveal |
|---|---|---|---|---|---|
| 1M | 0.00087 | 0.00045 | 0.00161 | 0.00086 | 0.00127 |
| 5M | 0.00093 | 0.00063 | 0.00144 | 0.00107 | 0.00147 |
| 20M | 0.00092 | 0.00063 | 0.00141 | 0.00107 | 0.00160 |

## The same, by time since the start of a clip (training clips are 64 frames)

| size | 4-96 | 96-200 | 200-400 | 400-800 | 800-1200 |
|---|---|---|---|---|---|
| 1M | 0.00111 | 0.00092 | 0.00097 | 0.00090 | 0.00088 |
| 5M | 0.00098 | 0.00067 | 0.00087 | 0.00085 | 0.00079 |
| 20M | 0.00093 | 0.00073 | 0.00091 | 0.00085 | 0.00080 |

