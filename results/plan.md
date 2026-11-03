# The analysis plan, evaluated

Written by `python -m sm.plan` from `results/summary.json`, against `docs/analysis-plan.md`. Confirmatory 5M seeds: 5m-s1, 5m-s2, 5m-s3, 5m-s4.

## Q1. Does one state hold more than its head reads out?

**Met as computed. Not used as evidence: see Changes after the freeze in docs/analysis-plan.md.**

- vanish: head 8.8, read 22.1, 2.52 times at 3 px; read minus head +13.4 [9.9, 16.6] at 3 px, +23.4 [20.8, 25.5] at 9 px
- vanish by seed, 3 px: head 8.1, 9.6, 7.8, 9.5, read 16.6, 24.7, 25.3, 21.2
- rebound: head 9.1, read 27.1, 3.00 times at 3 px; read minus head +18.1 [14.9, 26.8] at 3 px, +28.8 [25.3, 33.2] at 9 px
- rebound by seed, 3 px: head 9.0, 9.6, 8.6, 9.6, read 22.9, 30.1, 27.9, 28.0

## Q2. Where does the gap come from?

**Reported, no pass or fail.**

- vanish, the learned variance (mean against head): +1.0 [0.6, 1.6], not the cause
- vanish, how the map was trained (refit against head): +1.0 [0.6, 1.6], not the cause
- vanish, covered pixels and drawing through slabs (visible against refit): +2.7 [1.6, 3.8], counts
- vanish, being told where the hidden ball is (read against visible): +9.7 [6.2, 13.1], counts
- rebound, the learned variance (mean against head): +2.7 [1.5, 11.2], counts
- rebound, how the map was trained (refit against head): +3.0 [1.7, 11.4], counts
- rebound, covered pixels and drawing through slabs (visible against refit): +5.5 [0.6, 8.0], counts
- rebound, being told where the hidden ball is (read against visible): +9.6 [5.9, 16.0], counts

## Q3. Does the head use those directions?

**Met.**

- vanish, 5m-s1, D = 8: head 0.505, read's four out 0.052, random four out 0.411, extra drop 0.359
- vanish, 5m-s2, D = 8: head 0.620, read's four out 0.078, random four out 0.536, extra drop 0.458
- vanish, 5m-s3, D = 8: head 0.490, read's four out 0.083, random four out 0.396, extra drop 0.312
- vanish, 5m-s4, D = 8: head 0.589, read's four out 0.125, random four out 0.536, extra drop 0.411
- rebound, 5m-s1, D = 8: head 0.542, read's four out 0.052, random four out 0.401, extra drop 0.349
- rebound, 5m-s2, D = 8: head 0.547, read's four out 0.031, random four out 0.411, extra drop 0.380
- rebound, 5m-s3, D = 8: head 0.542, read's four out 0.031, random four out 0.474, extra drop 0.443
- rebound, 5m-s4, D = 8: head 0.542, read's four out 0.062, random four out 0.453, extra drop 0.391
- the read's four carry 0.0032, 0.0028, 0.0028, 0.0034 of the state's variance

## Q4. Does the read follow an unseen bounce?

**Met.**

- 5m-s1, D = 16: return direction 0.99 slope 1.00, outward direction 0.95
- 5m-s1, D = 28: return direction 0.99 slope 0.84, outward direction 0.99
- 5m-s2, D = 16: return direction 0.99 slope 0.92, outward direction 0.98
- 5m-s2, D = 28: return direction 1.00 slope 0.85, outward direction 1.00
- 5m-s3, D = 16: return direction 0.99 slope 0.99, outward direction 0.98
- 5m-s3, D = 28: return direction 0.98 slope 0.86, outward direction 0.99
- 5m-s4, D = 16: return direction 0.99 slope 1.01, outward direction 0.98
- 5m-s4, D = 28: return direction 0.99 slope 0.87, outward direction 0.99

## Q5. Is the trace a trace of the ball?

**Met.**

- window control, vanish trace at D = 1: 0.441 nats; largest past D = 12: 0.000
- 5M vanish trace by D 1: 6.297 2: 5.871 4: 3.215 6: 2.609 8: 2.493 12: 1.373 16: 0.833 20: 0.550 24: 0.346 28: 0.278 32: 0.149 40: 0.101 48: 0.055
- 5M exit trace by D 1: 0.086 4: 0.039 8: 0.009 16: 0.000 24: 0.000 32: -0.000 48: 0.000

## Q6. What does size do at equal training? (met means bigger decides better, on both families)

**Not met.**

- vanish, head at 3 px: 1M 2.6 [2.3, 2.9], 5M 8.8 [7.6, 9.8], 5M short 2.9 [2.7, 3.1], 1M long 9.4 [8.4, 10.2], 20M 9.8 [9.3, 10.4]
- vanish, trace halves: 1M 3.9 [3.7, 5.5], 5M 4.2 [3.6, 6.1]
- vanish: 1M long reaches 5M's interval: True
- rebound, head at 3 px: 1M below the first level [below the first level, below the first level], 5M 9.1 [6.8, 10.3], 5M short below the first level [below the first level, below the first level], 1M long 11.4 [10.6, 12.7], 20M 10.6 [9.4, 12.1]
- rebound, trace halves: 1M 17.9 [15.6, 19.5], 5M 22.6 [19.5, 26.1]
- rebound: 1M long reaches 5M's interval: True

## Q7. Is there a speed-independent fall?

**Reported, no pass or fail.**

- 0.15 to 0.4 px per frame: below the first level
- 0.4 to 0.6 px per frame: 9.6
- 0.6 to 1.3 px per frame: 11.4

## Q8. Efficiency

**Reported, no pass or fail.**

- teleport: against the causal ideal 0.221 [0.168, 0.313], against the state told 0.087
- phantom: against the causal ideal 0.117 [0.107, 0.179], against the state told 0.052
- deflect: against the causal ideal 0.030 [0.025, 0.042], against the state told 0.015

## Q9. Collision bias: quoted if under 1 in every seed

**Met.**

- peak restitution by seed: 0.74, 0.74, 0.79, 0.72

## Q10. Calibration, demo only

**Reported, no pass or fail.**

- frames 4-96: measured at a promised 1 in 1000: 0.00102, 0.00093, 0.00108, 0.00104, within a factor of 2
- frames 96-200: measured at a promised 1 in 1000: 0.00073, 0.00060, 0.00064, 0.00066, within a factor of 2
- frames 200-400: measured at a promised 1 in 1000: 0.00092, 0.00083, 0.00089, 0.00090, within a factor of 2
- frames 400-800: measured at a promised 1 in 1000: 0.00089, 0.00082, 0.00088, 0.00089, within a factor of 2
- frames 800-1200: measured at a promised 1 in 1000: 0.00080, 0.00078, 0.00079, 0.00074, within a factor of 2

## Checks added after the confirmatory results, so exploratory

**Reported, no pass or fail.**

- vanish, first differing frame: head 25.4 [15.0, 30.2], read 48.0 [28.0, 48.0], read minus head +22.6 [10.5, 32.9]
- vanish, same frame at D = 4: head 0.57, read 0.78, the observer told the lawful frame 0.66; levels where the read beats that observer: 11 of 11
- vanish, read on the sliver by level: brightness against the truth 1.37 to 1.82, share of its predicted change along the sliver 0.09 to 0.36
- vanish, visible map on the sliver by level: brightness against the truth 1.33 to 3.39, share of its predicted change along the sliver 0.12 to 0.23
- vanish, head's mean on the sliver by level: brightness against the truth 0.18 to 1.06, share of its predicted change along the sliver 0.05 to 0.57
- vanish, 5M head's share of that observer's sensitivity by D, where the observer is at 0.55 or better: 4: 0.410 6: 0.577 8: 0.529 12: 0.643 16: 0.473 20: 0.161 32: -0.398
- vanish, first differing frame, head crossing by group: 1m below the first level (seeds 14.9, 0.0, 0.0, 0.0), 5m 25.4 (seeds 17.3, 26.1, 26.0, 25.9), 20m 26.4 (seeds 26.4), 1m-window8 below the first level (seeds 0.0), 1m-long 25.3 (seeds 25.3), 5m-short below the first level (seeds 0.0)
- rebound, first differing frame: head 21.7 [16.3, 36.3], read 48.0 [46.3, 48.0], read minus head +26.3 [11.4, 31.7]
- rebound, same frame at D = 6: head 0.62, read 0.81, the observer told the lawful frame 0.72; levels where the read beats that observer: 7 of 10
- rebound, read on the sliver by level: brightness against the truth 0.88 to 1.54, share of its predicted change along the sliver 0.12 to 0.43
- rebound, visible map on the sliver by level: brightness against the truth 0.98 to 2.31, share of its predicted change along the sliver 0.18 to 0.32
- rebound, head's mean on the sliver by level: brightness against the truth 0.18 to 0.87, share of its predicted change along the sliver 0.10 to 0.64
- rebound, 5M head's share of that observer's sensitivity by D, where the observer is at 0.55 or better: 6: 0.538 8: 0.612 12: 0.489 16: 0.329 20: 0.089 24: -0.123 28: 0.036 32: -0.035 40: -0.659 48: -0.840
- rebound, first differing frame, head crossing by group: 1m 14.2 (seeds 13.5, 15.1, 16.0, 12.6), 5m 21.7 (seeds 19.6, 35.5, 16.8, 19.8), 20m 38.9 (seeds 38.9), 1m-window8 below the first level (seeds 0.0), 1m-long 38.0 (seeds 38.0), 5m-short 10.1 (seeds 10.1)
- vanish, head with the read's four out while hidden, by D: 8: 0.085 16: 0.001 24: 0.000
- vanish, head with four from the slab cover map out while hidden, by D: 8: 0.016 16: 0.000 24: 0.000
- vanish, head with four random, matched to the read's variance out while hidden, by D: 8: 0.404 16: 0.053 24: 0.010
- vanish, head with four random out while hidden, by D: 8: 0.470 16: 0.074 24: 0.010
- rebound, head with the read's four out while hidden, by D: 8: 0.044 16: 0.007 24: 0.000
- rebound, head with four from the slab cover map out while hidden, by D: 8: 0.056 16: 0.000 24: 0.000
- rebound, head with four random, matched to the read's variance out while hidden, by D: 8: 0.404 16: 0.138 24: 0.020
- rebound, head with four random out while hidden, by D: 8: 0.435 16: 0.225 24: 0.040
- share of the state's variance, the read's four: 0.0032, 0.0028, 0.0028, 0.0034
- share of the state's variance, the cover map's four: 0.0839, 0.0688, 0.0814, 0.0909
- share of the state's variance, the matched random four: 0.0039, 0.0034, 0.0032, 0.0044
