# Analysis plan

Written before any confirmatory seed was trained. The commit that last touched this file before `runs/5m-s1` started is the freeze. A later change that moves a reported number goes in the last section with its reason, and every run is measured again.

## What was seen before the freeze

Noise 0.15, 64-frame clips and crossing scenes were picked while watching vanish curves on single 1M runs, with seven pilot evaluations on disk. Fresh scenes (seed 20261102) remove scene selection. They do not remove design selection, and the seeds below cannot either, because they share the design.

A reviewer's scripts produced rough versions of most measures below on `5m-s0`, and the analysis was debugged on the same run. So the predictions below are informed by that seed.

## Runs

| Group | Runs | Role |
|---|---|---|
| Exploratory | `1m-s0` to `s3`, `5m-s0` | Seen while the analysis was built |
| Controls | `1m-window8-s0`, `5m-short-s0`, `1m-long-s0` | Known memory length; 5M at 10000 steps; 1M at 60000 steps |
| Confirmatory | `5m-s1` to `s4`, `20m-s0` to `s2` | Not trained or looked at before the freeze |

Every 5M number comes from confirmatory seeds only. `5m-s0` is set aside and printed once, so shrinkage is visible. 1M gets no new seeds, because its crossings already agree within about 0.2 frames, so the 1M side of each size comparison stays exploratory.

If time runs out, runs are dropped in this order: `20m-s2`, `20m-s1`, `5m-s4`, `5m-s3`. The bare minimum for a size claim is two confirmatory 5M seeds and one 20M seed. A group with one seed has intervals over scenes only, and the text will say so.

## How every number is made

- All models and observers see the same scenes and noise, seed 20261102. The differences between them are paired.
- 2AFC: proportion of matched pairs where the lawful twin costs fewer nats than the impossible one, over the frames where they differ. Two windows for occluded events: until the lawful ball is 3 px out, and until it is fully out at 9 px.
- Crossing: first fall through 0.5 by linear interpolation. If the curve starts under 0.5 the report says "below" the first level.
- Threshold: binomial maximum likelihood logistic in log magnitude at 75% correct, with a fitted floor, because a directional bias under matched noise holds proportion correct at a constant other than 0.5 as magnitude falls.
- Intervals: 95%, hierarchical bootstrap, 500 resamples of seeds then scenes.
- Trace: median surprisal per frame at identical empty frames, ball seen earlier minus ball never there.
- Linear read: ridge from the frozen state, fitted on fresh lawful clips, one fit for measurement and the demo.

## Questions and what each answer allows

These are the only claims that can reach the headline. Each is reported whatever it shows. Nothing here is a significance test, so there is no multiple-comparison correction. The criterion is what decides the sentence.

**Q1. Does one state hold more than its head uses?** The linear read goes through the same pixel likelihood as the head, on the same twins.

Criterion: on vanish and on rebound, in the 3 px and 9 px windows, read crossing minus head crossing has a 95% interval above 0 on confirmatory 5M, and the read crossing is at least 1.5 times the head crossing. Reported per seed as well.

- All met: "A linear map keeps the lawful twin ahead about N times as long as the model's head, on the same state and the same likelihood."
- Only under the 3 px window: the gap is the instrument, so the claim is dropped and the 3 px curve is described as such.

**Q2. Why does the head fall behind?** The head's mean is scored under the world's fixed noise, with the learned variance removed.

- Mean crossing near the read's, meaning its interval on (mean minus head) excludes 0 and it is within 25% of the read: the head kept the ball and its own uncertainty hides it.
- Mean crossing near the head's: the head's output has lost what the read still finds.
- Between: both, and the numbers are given without a causal sentence.

**Q3. Does the head use those directions?** While hidden, the read's directions are projected out of the state and the head is measured again. Control: a random subspace of the same size.

Criterion: at D = 8, the drop in head proportion correct from removing the read's directions exceeds the drop from removing random ones by at least 0.10, in every confirmatory 5M seed. Met: "the head depends on those directions." Not met: "removing them changes the head no more than a random subspace", which is itself the finding. Removing random directions also damages the state, so a large drop in the control means both are reported and no causal sentence is written.

**Q4. Does the read follow an unseen bounce?** Signed motion along x, read against truth, on the outward and return strokes.

Criterion: at D = 16 and D = 28, in every confirmatory 5M seed, return-stroke direction right at least 0.90 of the time with slope in 0.7 to 1.3, and outward direction right at least 0.80. Met: "a linear read of a causal pixel model trained only on next-frame likelihood follows the ball to the wall and back." Quoted read error is not evidence, because three quarters of a round trip is the middle of the slab. The sentence never says it beats dead reckoning: dead reckoning with walls is perfect here.

**Q5. Is the trace a trace of the ball?** Two nulls.

- `1m-window8-s0`: past D = 12 its trace falls under 10% of its value at D = 1. If not, the trace is contaminated and is not quoted as memory.
- Exit: the ball visibly leaves through the wall, then the same empty frames against never there. The vanish trace is reported beside it, and only the part above the exit trace is called holding on to a hidden ball.

**Q6. What does size do at equal training?** Crossing in the 9 px window, trace amplitude at D = 1 and trace half-life, 1M against 5M against 20M at 20000 steps.

Criterion for "bigger decides better here": 5M's interval above 1M's on vanish and on rebound, and `5m-short-s0` still above 1M. If `1m-long-s0` reaches 5M's interval, the sentence is about training length and not size. The same for 20M against 5M. A size difference is never written as a memory-length difference unless trace half-life also separates.

**Q7. Is there a speed-independent fall?** Crossing within the bands 0.15 to 0.4, 0.4 to 0.6, 0.6 to 1.3 px per frame. Long hidden times come with slow balls, so any pooled vanish crossing is quoted only with its bands next to it.

**Q8. Efficiency.** Against the causal ideal, marginal over initial position and velocity, as the denominator, and against the observer told the state as a bound. Teleport, phantom and deflect are quoted. Collision and hidden are shown and not called efficiencies, since bias and stimulus shift dominate them. The causal ideal knows radius and brightness, so the real limit is a little weaker and these efficiencies are a little low.

**Q9. Collision bias.** Peak of a parabola through median log Bayes factors along signed restitution. Quoted if it is under 1 in every confirmatory 5M seed. A unimodal head averaging over contact timing predicts the same, so no sentence says the model expects softer collisions.

**Q10. Calibration, demo only.** Promised against measured false alarms on scenes from the training generator, by context and by time since clip start to 1200 frames on 512 clips. "Holds to N frames" is written only through bins within a factor of 2 of the promise at 1 in 1000. Frames within a clip are dependent, so this is a descriptive check and not a coverage guarantee. It voids once a demo interaction leaves the generator.

## Words not used

"Memory lasts 10 frames", "the model believes", "dead reckoning" without "wall-blind" or "with walls", "size buys permanence", "contradicts 2610.07355", "bias-free", any comparison of durations with V-JEPA 2.

## Changes after the freeze

None yet.
