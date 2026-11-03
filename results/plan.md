# The analysis plan, evaluated

Written by `python -m sm.plan` from `results/summary.json`, against `docs/analysis-plan.md`. Confirmatory 5M seeds: 5m-s1, 5m-s2.

## Q1. Does one state hold more than its head reads out?

**Met.**

- vanish: head 8.9, read 21.1, 2.37 times at 3 px; read minus head +12.2 [7.6, 15.9] at 3 px, +21.8 [18.4, 24.4] at 9 px
- vanish by seed, 3 px: head 8.1, 9.6, read 16.6, 24.7
- rebound: head 9.3, read 26.2, 2.83 times at 3 px; read minus head +17.0 [12.9, 30.4] at 3 px, +26.5 [23.3, 33.6] at 9 px
- rebound by seed, 3 px: head 9.0, 9.6, read 22.9, 30.1

## Q2. Where does the gap come from?

**Reported, no pass or fail.**

- vanish, the learned variance (mean against head): +1.1 [0.6, 1.8], not the cause
- vanish, how the map was trained (refit against head): +1.0 [0.5, 1.6], not the cause
- vanish, covered pixels and drawing through slabs (visible against refit): +2.5 [1.7, 3.9], counts
- vanish, being told where the hidden ball is (read against visible): +8.8 [3.9, 12.3], counts
- rebound, the learned variance (mean against head): +3.3 [1.0, 16.1], counts
- rebound, how the map was trained (refit against head): +3.6 [1.2, 16.3], counts
- rebound, covered pixels and drawing through slabs (visible against refit): +6.0 [2.6, 8.9], counts
- rebound, being told where the hidden ball is (read against visible): +7.3 [4.1, 12.2], counts

## Q3. Does the head use those directions?

**Met.**

- vanish, 5m-s1, D = 8: head 0.505, read's four out 0.052, random four out 0.411, extra drop 0.359
- vanish, 5m-s2, D = 8: head 0.620, read's four out 0.078, random four out 0.536, extra drop 0.458
- rebound, 5m-s1, D = 8: head 0.542, read's four out 0.052, random four out 0.401, extra drop 0.349
- rebound, 5m-s2, D = 8: head 0.547, read's four out 0.031, random four out 0.411, extra drop 0.380
- the read's four carry 0.0032, 0.0028 of the state's variance

## Q4. Does the read follow an unseen bounce?

**Met.**

- 5m-s1, D = 16: return direction 0.99 slope 1.00, outward direction 0.95
- 5m-s1, D = 28: return direction 0.99 slope 0.84, outward direction 0.99
- 5m-s2, D = 16: return direction 0.99 slope 0.92, outward direction 0.98
- 5m-s2, D = 28: return direction 1.00 slope 0.85, outward direction 1.00

## Q5. Is the trace a trace of the ball?

**Met.**

- window control, vanish trace at D = 1: 0.441 nats; largest past D = 12: 0.000
- 5M vanish trace by D 1: 6.281 2: 6.010 4: 3.130 6: 2.624 8: 2.516 12: 1.380 16: 0.831 20: 0.524 24: 0.326 28: 0.279 32: 0.135 40: 0.086 48: 0.042
- 5M exit trace by D 1: 0.108 4: 0.060 8: 0.013 16: 0.000 24: 0.000 32: 0.000 48: 0.000

## Q6. What does size do at equal training? (met means bigger decides better, on both families)

**Not met.**

- vanish, head at 3 px: 1M 2.6 [2.3, 2.9], 5M 8.9 [7.3, 10.1], 5M short 2.9 [2.7, 3.1], 1M long not measured yet, 20M not measured yet
- vanish, trace halves: 1M 3.9 [3.6, 5.4], 5M 4.0 [3.6, 5.9]
- rebound, head at 3 px: 1M below the first level [below the first level, below the first level], 5M 9.3 [below the first level, 10.7], 5M short below the first level [below the first level, below the first level], 1M long not measured yet, 20M not measured yet
- rebound, trace halves: 1M 17.9 [15.7, 19.4], 5M 25.3 [21.9, 28.4]

## Q7. Is there a speed-independent fall?

**Reported, no pass or fail.**

- 0.15 to 0.4 px per frame: below the first level
- 0.4 to 0.6 px per frame: 9.5
- 0.6 to 1.3 px per frame: 12.0

## Q8. Efficiency

**Reported, no pass or fail.**

- teleport: against the causal ideal 0.216 [0.168, 0.310], against the state told 0.085
- phantom: against the causal ideal 0.122 [0.104, 0.236], against the state told 0.054
- deflect: against the causal ideal 0.034 [0.025, 0.044], against the state told 0.017

## Q9. Collision bias: quoted if under 1 in every seed

**Met.**

- peak restitution by seed: 0.74, 0.74

## Q10. Calibration, demo only

**Reported, no pass or fail.**

- frames 4-96: measured at a promised 1 in 1000: 0.00102, 0.00093, within a factor of 2
- frames 96-200: measured at a promised 1 in 1000: 0.00073, 0.00060, within a factor of 2
- frames 200-400: measured at a promised 1 in 1000: 0.00092, 0.00083, within a factor of 2
- frames 400-800: measured at a promised 1 in 1000: 0.00089, 0.00082, within a factor of 2
- frames 800-1200: measured at a promised 1 in 1000: 0.00080, 0.00078, within a factor of 2
