# Surprise Meter

A small world model, trained from scratch on a laptop on nothing but next-frame pixels, then measured the way a psychophysicist measures a person. It runs in your browser, guesses every frame before it sees it, and a meter shows how strange each frame was. Break the physics and watch the meter jump.

**[Open the demo](https://abhaymettu.github.io/surprise-meter/)**. No install, and nothing leaves your machine.

![A ball goes behind a slab. The outline is read straight out of the model's state.](results/figures/vanish.png)

## What it found

Delete a ball while it is hidden behind a slab. Nothing comes out. Does the model notice?

Two versions of each clip go in, identical until the ball should reappear: one lawful, one with the ball gone. The model is right when the lawful one costs fewer nats. Scored until the lawful ball is 3 px out, its head stops getting this right after about 9 frames hidden, pooled over speeds. Hidden time and speed are tied here, because a slab can only be so wide: below 0.4 px per frame the head is wrong even at 4 frames, from 0.4 to 0.6 it crosses at 9.5, and faster balls still win at 12, the longest a fast ball stays hidden.

That number depends on how the twins are scored. Over a reveal window each twin is scored under its own state, so the impossible twin's state has already seen empty frames. On the first frame where the twins differ, both are predicted from one state, and there the head stays above chance to about 25 frames on a crossing, with a wide interval, 11.3 to 28.5. That frame carries little signal: an observer told the exact lawful frame gets 0.66 at 4 frames hidden and 0.50 by 28, and the head runs 0.03 to 0.09 below it throughout.

The state keeps where the ball is. A second linear read, fitted to the ball's position on hidden frames of other scenes from these families, follows the ball to the wall and back: on the return stroke the direction is right at least 99% of the time, with slope 0.84 to 1.00 against truth, at 16 and 28 frames hidden. Dead reckoning with walls is perfect in this world, so this says what the state holds and does not rank it against anything better.

## A comparison that did not hold

I compared the head with five readers that turn the same frozen state into a picture of the next frame, each one change from the last, scored on the same twins under the world's pixel noise. Over the 3 px window, a ridge map told where hidden balls were when it was fitted kept the lawful twin ahead to 21.1 frames on a crossing and 26.2 on a hidden bounce, against the head's 8.9 and 9.3, and the analysis plan's criterion for that gap was met.

| Hidden frames where the lawful twin stops winning, 3 px window | Straight crossing | Hidden bounce |
|---|---|---|
| The model's head, with its own variance | 8.9 [7.4, 10.1] | 9.3 [below 4, 10.7] |
| The head's mean, at the world's noise | 10.0 [8.7, 11.1] | 12.5 [10.1, 16.9] |
| Ridge to the next frame | 9.9 [8.3, 11.1] | 12.9 [10.3, 16.9] |
| Ridge to visible pixels, drawn through slabs | 12.4 [10.8, 14.3] | 18.9 [16.4, 21.1] |
| Ridge to balls without slabs, told where they are when fitted | 21.1 [15.6, 25.6] | 26.2 [22.3, 30.9] |

It is not evidence of what the state holds. On the first frame where the twins differ, that map wins more often than the observer told the exact lawful frame, at every hidden time on a crossing: 0.78 against 0.66 at 4 frames, and still above chance at 48. No reader does that by knowing more. Maps drawn through slabs decoded from the state paint the sliver of ball that comes out brighter than it is, 1.2 to 1.9 times on a crossing, and squared error at a fixed noise rewards a bright guess in the right place. The visible map's overshoot grows the longer the ball was hidden, to 3.4 times at 48 frames, which is a glow at the slab's edge and not memory.

The ablation did not hold either. Projecting the map's four strongest directions out of the state while the ball is hidden takes the head at 8 frames from about 0.55 to under 0.1, and four random directions leave 0.40 to 0.54. But four directions chosen the same way from a map to slab cover, a probe for something else, take it to 0.03 or lower. Directions picked by the size of a ridge map's weights are low-variance directions the head needs whatever the probe was for.

Both checks were added after the confirmatory results, following an outside review, and are recorded in [the plan](docs/analysis-plan.md#changes-after-the-freeze).

## How it is measured

**The world.** Up to two balls and up to two slabs on 32 by 32 pixels, with pixel noise of 0.15. Every event is built as a twin: same scene, same noise, one law broken.

**The model.** A recurrent token state updated once per frame by a small transformer. There is no clock and no access to old pixels, so whatever it remembers is in the state. It trains on next-frame likelihood and nothing else.

**The observers.** Six reference observers see the same scenes and noise, from the observer told the lawful continuation down to one with no memory. The denominator for efficiency is the causal ideal, which knows the laws, radii and brightness but not where the balls started or how fast. An 8-frame attention window is the control with a known memory length.

Every 5M number comes from seeds trained after [the analysis plan](docs/analysis-plan.md) was frozen. [`results/plan.md`](results/plan.md) is that plan's criteria evaluated by code. Brackets are 95% hierarchical bootstrap intervals over seeds and scenes.

## Thresholds, against an ideal

At 75% correct, as magnitudes: teleport at 0.14 px, phantom bounce at 0.093 px, deflection at 3.4 degrees. The causal ideal reaches 0.065 px, 0.032 px and 0.63 degrees. Efficiency here is the squared ratio of 75% thresholds, against a causal ideal linearized around the true trajectory: 0.216 [0.168, 0.310], 0.122 [0.105, 0.251] and 0.034 [0.026, 0.046]. The causal ideal knows radius and brightness, so the real limit is a little weaker and these efficiencies are a little low.

![Psychometric curves for each violation.](results/figures/psychometric.png)

The model's likelihood peaks at a restitution of 0.74, where the world's is 1. A unimodal head averaging over contact timing predicts the same, so this does not say it expects softer collisions.

## The meter in the demo

The meter is a table of held-out lawful frames: "1 in 9,300" means that share of frames from the training generator look at least this strange. At a promised 1 in 1000 it measures 0.0006 to 0.0010 through 1200 frames on 512 clips, far past the 64-frame clips the model trained on. Frames within a clip are dependent, so that is a descriptive check and not a coverage guarantee, and it stops applying the moment you drag a ball, which is the point.

## What else to keep in mind

- The design was not blind. Noise, clip length and crossing scenes were picked while watching vanish curves, and the seeds cannot remove that because they share the design. `5m-s0` was watched while the analysis was built, so it is set aside. It read 11.0 and 12.5 where the confirmatory seeds read 8.9 and 9.3.
- A longer reveal window lowers the head instead of raising it, because a twin decision needs the predicted ball above about half its contrast and frames with a ball cost more than empty ones.
- Size is not separated from training length. At 20000 steps 5M crosses at 8.9 frames on a vanish and 1M at 2.6, but 1M trained to 60000 steps reaches 9.4 [8.5, 10.3], and 5M stopped at 10000 stays at 2.9 [2.7, 3.1]. 20M at 20000 steps reads 9.8 [9.2, 10.3]. Those three are one seed each, so their intervals are over scenes only.
- The 8-frame window control is weak inside its window too. Its trace at one hidden frame is about a tenth of the recurrent 1M's, so it shows the trace ends with memory and not much more.
- Every training scene holds at least one ball, and a ball leaving the frame has no precedent in training.

## Rerun everything

```
python -m venv .venv && .venv/bin/pip install -e .
make results   # every table, figure and verdict, from the measurements in this repo. About two minutes
make test      # the instrument, checked without any trained model
make measure   # the measurements again from the published weights. Hours on an M-series Mac
make train     # everything from nothing. Days on one Mac, safe to stop and rerun
make demo      # the demo at http://127.0.0.1:8791
```

Trained on an M4 Pro with 24 GB. No job goes above 8 GB. A 5M seed takes about 1.5 hours and a 20M seed about 7.

## Where this sits

Scoring matched valid and invalid videos by likelihood is established: LikePhys (Yuan, Pizzati, Pinto, Kunze, Laptev, Newman, Torr and De Martini, [arXiv:2510.11512](https://arxiv.org/abs/2510.11512)) does it for video diffusion models, and [N8python/physics-trainer](https://github.com/N8python/physics-trainer) does it on a small procedural world model at fixed magnitudes. Probes that find more than a model's output shows are a live question: Xie and Alanwar ([arXiv:2610.07355](https://arxiv.org/abs/2610.07355)) compare a probe on V-JEPA 2's encoder with its predictor for hidden objects, and Esmati, Nath and colleagues ([arXiv:2606.05328](https://arxiv.org/abs/2606.05328)) probe video diffusion models for physical plausibility. Their scenes, scores and scale differ from these, so nothing here says anything about their results. What this adds is thresholds against an ideal observer for a small model trained from scratch in a world where every law is known, confirmatory seeds behind a frozen plan, and a case where that observer is what showed a probe's apparent advantage to be an artifact of scoring.

MIT licensed. [`AGENTS.md`](AGENTS.md) maps the code.
