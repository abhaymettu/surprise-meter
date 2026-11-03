# Surprise Meter

## The question

Infants look longer at an impossible event than at a lawful one, and that difference in looking is how developmental psychology asks what a baby expects. A world model that predicts pixels can be asked the same way. Show it a clip, then an impossible twin of that clip, and see which one costs it more.

I trained small recurrent world models from scratch on a laptop, on nothing but next-frame pixels from a procedural world of balls and slabs. Then I measured them the way psychophysics measures a person: matched pairs of clips, thresholds, and an ideal observer as the yardstick. Between frames the model carries only a small set of state vectors, so whatever it keeps about a ball it can no longer see has to be in there.

The sharpest test is a deletion. A ball goes behind a slab and never comes out. The lawful twin is the same clip with the ball coming out, and the model is right when the lawful twin costs it fewer nats.

## How long it holds on depends on how you ask

Scored until the lawful ball is 3 px out of the slab, the model stops favouring the lawful twin after about 9 frames hidden. That number mixes memory with speed, since only slow balls stay hidden long. Balls under 0.4 px per frame fail even at 4 frames, and faster ones still win at 12.

It also depends on the scoring. Over a window of frames each twin is judged by its own state, and the impossible twin's state has already seen empty frames. On the first frame where the twins differ, both are judged from one state, and there the model stays above chance to about 25 frames, with a wide interval. That single frame carries little signal. An observer told the exact lawful frame gets 0.66 at 4 frames hidden and 0.50 by 28, and the model runs a little below it the whole way.

A linear read of the state follows a hidden ball to the wall and back. On the return stroke it gets the direction right at least 99% of the time, with a slope of 0.84 to 1.00 against the truth.

## The probe that beat the ideal observer

I wanted to show that the state holds more than the model's own predictions use. A ridge map from the frozen state, fitted while told where hidden balls were, kept the lawful twin ahead to 21 frames against the head's 9, and the frozen analysis plan's criterion for that gap was met. Removing the map's four strongest directions from the state broke the head, and four random directions did not.

An outside review asked me to check both, and neither held. On the first frame where the twins differ, the map wins more often than the observer told the exact lawful frame, 0.78 against 0.66 at 4 frames hidden. Nothing does that by knowing more. The map, drawn through slabs decoded from the state, paints the emerging sliver of ball 1.2 to 1.9 times brighter than it is, and squared error rewards a bright guess in the right place. Then four directions chosen the same way from a map to slab cover, a probe for something else, broke the head just as badly.

The ideal observer is what caught it. Without a reference that knows everything, an over-bright probe looks like a probe that knows more.

## Against an ideal observer

On visible events the model is far from efficient. Its thresholds at 75% correct are 0.14 px for a teleport, 0.093 px for a bounce short of the wall, and 3.4 degrees for a deflection. An observer that knows the laws but not where the balls started reaches 75% at 0.065 px, 0.032 px and 0.63 degrees. The squared ratio is 0.22, 0.12 and 0.034.

## The demo

The model runs in your browser on WebGPU and guesses every frame before it sees it. A meter shows how strange each frame was, as a rate on held-out lawful frames: "1 in 9,300" means that share of ordinary frames look at least this strange. Drag a ball to break the physics and the meter jumps. The outline of a hidden ball is read straight out of the model's state.

## What did not hold

I wanted to say that a bigger model holds a hidden ball longer, and I can't. At the same number of steps the 5M model crosses at about 9 frames and the 1M model under 3. But the 1M model trained three times as long reaches 9.4, and the 5M model trained half as long stays at 2.9, so training length explains it as well as size does.

The design was not blind. The noise level, the clip length and the share of scenes with a ball heading for a slab were picked while watching these curves on single runs, and the confirmatory seeds share that design.

## Rerun it

`make results` rebuilds every table, figure and verdict from the measurements in the repository in about two minutes. `make measure` reruns the measurements from the published weights, and `make train` trains everything from nothing on one Mac.
