# Surprise Meter

## The question

Infants look longer at an impossible event than at a lawful one, and that difference in looking is one way developmental psychology asks what a baby expects. A world model that predicts pixels can be asked the same way. Show it a clip, then an impossible twin of that clip, and see which one costs it more.

I trained small recurrent world models from scratch on a laptop, on nothing but next-frame pixels from a procedural world of balls and slabs. Then I measured them the way psychophysics measures a person: matched pairs of clips, thresholds, and observers that know the laws as the yardstick. Between frames the model carries only a small set of state vectors, so whatever it keeps about a ball it can no longer see has to be in there.

The sharpest test is a deletion. A ball goes behind a slab and never comes out. The lawful twin is the same clip with the ball coming out, and the model is right when the lawful twin costs it fewer nats.

## How long it holds on depends on how you ask

Scored until the lawful ball is 3 px out of the slab, the model stops favoring the lawful twin after about 9 frames hidden, on four seeds trained after the analysis plan was frozen. That number mixes memory with speed, since only slow balls stay hidden long. Balls under 0.4 px per frame fail even at 4 frames, and balls over 0.6 cross at 11.4.

It also depends on the scoring. Over a window of frames each twin is judged by its own state, and the impossible twin's state has already seen empty frames. On the first frame where the twins differ, both are judged from one state, and there the model stays above chance to about 25 frames. A model that only sees the last 8 frames is below chance on that frame at every hidden time, so this is memory. Mostly it is memory that a ball went in. How well the model's predicted change lines up with the ball coming out, as a cosine, falls from 0.57 at 4 frames hidden to 0.19 at 24.

A linear read of the state, fitted to hidden positions in other scenes, follows a hidden ball to the wall and back. On the return stroke it gets the direction right at least 98% of the time, with a slope of 0.84 to 1.01 against the truth. The model uses those directions. Removing the four the read leans on most, 0.3% of the state's variance, takes the model from 0.55 to 0.085 at 8 frames hidden, and four random directions with as much variance leave it at 0.40.

## The probe that beat an observer told the answer

I wanted to show that the state holds more than the model's own predictions use. A ridge map from the frozen state, fitted while told where hidden balls were, kept the lawful twin ahead to 22 frames against the model's 9, and the frozen analysis plan's criterion for that gap was met.

An outside review asked me to check it, and it did not hold. On the first frame where the twins differ, the map wins more often than an observer told the exact lawful frame, 0.78 against 0.66 at 4 frames hidden. At a fixed noise that decision goes to whoever draws more than half the sliver's brightness onto the sliver, so a reader that draws the ball at its true brightness cannot beat that observer. The map draws the sliver 1.4 to 1.8 times brighter than it is. It beats the model the same way: the model's prediction lines up with the sliver better, a cosine of 0.57 against 0.36, but carries less of its brightness.

The observer told the answer is what caught it. Without that reference, a brighter guess looks like a better-informed one.

A second review caught a control of mine that was not fair. Four directions from a map to slab cover broke the model even more than the read's four, which made the ablation look meaningless, but they carry 7 to 9% of the state's variance, and the model needs to know where slabs are anyway. Random directions matched in variance are the fair control, and the read's four still stand out against them.

## Against an observer that knows the laws

On visible events the model is far from efficient. Its thresholds at 75% correct are 0.14 px for a teleport, 0.095 px for a bounce short of the wall, and 3.6 degrees for a deflection. An observer that knows the laws but not where the balls started reaches 75% at 0.065 px, 0.032 px and 0.63 degrees. The squared ratio is 0.22, 0.12 and 0.030.

## The demo

The model runs in your browser on WebGPU and guesses every frame before it sees it. A meter shows how strange each frame was, as a rate on held-out lawful frames: "1 in 9,300" means one ordinary frame in 9,300 looks at least this strange. Drag a ball to break the physics and the meter jumps. The outline of a hidden ball is a linear read of the model's state.

## What did not hold

I wanted to say that a bigger model holds a hidden ball longer, and I can't. At the same number of steps the 5M model crosses at about 9 frames and the 1M model under 3. But the 1M model trained three times as long reaches 9.4, and the 5M model trained half as long stays at 2.9, so these runs cannot separate size from training length. Each of those is one seed.

The design was not blind. The noise level, the clip length and the share of scenes with a ball heading for a slab were picked while watching these curves on single runs, and the confirmatory seeds share that design. The seed I watched while building the analysis reads 11.0 frames, above all four confirmatory seeds, so the intervals understate how far a new seed can move.

## Rerun it

`make results` rebuilds every table, figure and verdict from the measurements in the repository in about two minutes, and checks that every number in this write-up is in them. `make measure` reruns the measurements from the published weights, and `make train` trains everything from nothing on one Mac.
