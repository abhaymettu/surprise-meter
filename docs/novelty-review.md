# surprise-meter: second novelty review

2026-11-02. State read: 1M three seeds, 5M one seed, 20M untrained.

Numbers marked (R) are reviewer checks: CPU scripts on frozen weights and stored `eval.npz`, 5M seed 0 unless stated, 20 to 96 scenes. They are not in `results/`. Rebuild them in the pipeline before quoting any.

The first review still bears. Its apples-to-oranges argument still blocks any contest with 2610.07355. Its quote "'Lost' describes the decision, not the information" now collides with C2. It asked for a known-posterior observer and a memory-length control; SKE is not a posterior and `1m-window8` is unrun.

## 1. Verdicts

| Claim | Verdict | Closest prior work | Wording that survives |
|---|---|---|---|
| C1 thresholds, SKE efficiency | NEW as a combination | MultiEcho (sweeps, no thresholds); physics-trainer (twins, fixed magnitudes); SKE ideal-observer approximation (2608.24768) | "Per-violation thresholds with an ideal-observer denominator." Quote teleport and phantom: 0.10 of SKE, about 0.34 of a causal ideal (R). Not collision or hidden: bias and a shifting stimulus population contaminate them. |
| C2 readouts come apart | PARTLY DONE; the sentence is ALREADY DONE | 2610.07355 calls this gap its "central measurement", encoder versus predictor in V-JEPA 2 | "Within one recurrent state, a linear map scored by the same likelihood keeps the lawful twin ahead about twice as long as the model's head, near 20 versus 10 hidden frames" (R, needs seeds). Drop 11.5. |
| C3 size and memory | PARTLY DONE | 2606.28455 (state inputs, small models, three seeds); Garrido 2025 (first review); 2610.07355 is not comparable | "At equal steps 5M's reveal decision is far better than 1M's; trace amplitude is 1.4 to 2 times larger; trace half-life is close." |
| C4 unseen bounce | PARTLY DONE | Nayebi 2023 (linear read of occluded ball state, Mental Pong); Rajalingham 2022; KVAE 2017 (imputes an unseen bouncing ball, with a smoother); Deep Tracking 2016; 2610.18637 lists the bounce as open | "A linear read of a causal pixel model trained only on next-frame likelihood follows the ball to the wall and back" (stroke test, section 3). Not "beats dead reckoning". |
| C5 conformal meter | ALREADY DONE as method; demo only | Bates et al. 2104.08279; 2608.15090; 2609.27179; 2610.34999 | "Frame false alarms near the promised rate on generator scenes; by context, up to 1.6 times high and 2.2 times low." The 1200-frame claim is not quotable yet. |
| C6 soft collisions | NEW as a datum; the method is a standard point of subjective equality | None found; 2610.07355's bias floor is nearest | "5M's likelihood peaks near restitution 0.8, not 1" (R, rough). Not "expects softer collisions". |

## 2. What changes now

1. **[before more seeds] Score a longer reveal.** In `events.py` keep `REVEAL_PX = 3` and add a window through full emergence, about 9 px (`_finish` returns both; the `T` formulas grow). A Bayes observer certain of the ball, with exact velocity and position spread s per axis, gets PC 1.00, 0.65, 0.00 at s = 0.5, 1, 1.5 under 3 px, because "late" explains an empty frame. At 9 px it holds 1.00 at s = 1 (R, 20 scenes). 5M's read error near D=10 is about 1 px, so the 10-frame crossing may be partly instrument. Eval only, minutes per run; remeasure all.
2. **[before more seeds] Put the read through the decision.** Add to `xray.py`: decoded image as prediction, slab composited, noise sigma, same twins. Head versus decoder PC: vanish D=12 0.40 versus 0.875; D=20 0.07 versus 0.54; rebound D=20 0.27 versus 0.69 (R, 48 scenes). This is the strongest result found. Then run the 2606.28455 test: project the decoder's directions out of the state while hidden and remeasure the head. If nothing moves, the head does not use that information. About 2 minutes per model.
3. **[before more seeds] Add observers to `observers.py`.** Wall-aware dead reckoning, slab midpoint, frozen at last seen, and a causal ideal: residual energy after projecting on the render Jacobian of initial position and velocity, 12 s on CPU for all families. Make it the headline denominator and keep SKE as the bound. Causal thresholds (R, 64 scenes): teleport 0.064, deflect 0.77, phantom 0.031, hidden 0.10, collision 0.013. Efficiencies: 0.34, 0.043, 0.34, 0.008, 0.0003. Radius and brightness are given, so the true causal limit is somewhat weaker.
4. **[before more seeds] Sign the axis.** Add negative restitution levels to `FAMILIES["collision"]`, store teleport sign, and replace the fixed 0.5 floor in `analyze.fit_threshold` with a fit giving PSE and JND. Under matched noise the bias and noise terms both scale with magnitude, so PC tends to a constant as magnitude falls: phantom 0.60 at 0.01 and 0.03, teleport 0.63 at 0.04 and 0.08, collision 0.31 to 0.38. Correct `surprise.py`: 2AFC is criterion-free, not "bias-free". Fix `hidden`: `vlo` depends on d, so median speed runs 0.47 to 0.68 and the window doubles across levels. Minutes.
5. **[before more seeds] Give the trace a null.** Move `1m-window8` to the front of `ladder.PLAN`: with 8 frames of memory, trace and read must vanish past D=8. Add an exit control (the `passthrough` edit, then empty frames against never-there), as 2610.07355 does. "Still positive at 48" (0.06 nats per frame, R) means nothing without them. About an hour per window seed; minutes for exit.
6. **[before more seeds] Normalize the trace.** `analyze.trace` uses window sums, and the window grows from 4 to 19 frames as balls slow. Per frame, vanish half-life falls from 11.5 to 6.6 at 5M and 7.3 to 5.2 at 1M (R, stored evals, seed 0). Rebound barely moves (22 and 19). No rerun.
7. **[before more seeds] Test step length.** 1M at 20000 steps has validation excess 4.6 nats per frame; 5M at 10000 has 4.67. Train one 1M to 60000 steps (2 to 4 hours at logged rates) and one 5M to 10000. 1M vanish crossings are 2.7, 2.7, 2.5, so skip `1m-s4`; 5M brackets cover scenes only and need at least 3 seeds.
8. **[before more seeds] Fix the read.** `xray.centroid` clips at brightness 0.25 and falls back to frame centre. 1M's 8 to 12 px errors are largely that fallback (seed 0, median mass 0, quartiles 1.0 and 13.2 at rebound D=16). Report hit rate and error given a hit, add a ridge-to-position read, report all four fractions.
9. **[before more seeds] Freeze and split.** Commit the analysis before further runs. Call 1M s0 to s3 and 5M s0 exploratory, 5M s1 onward and 20M confirmatory. Store per-context null tails in `evaluate.py` (only the pooled tail is kept). Move `scratch/long_roll.py` into the pipeline with at least 512 clips; today it has 64, about 26 expected alarms in the last bin, so "holds to 1200" is not established.
10. **[wording] Disclose** that noise 0.15, 64-frame clips and crossing scenes were picked while watching vanish curves, with seven pilot eval logs on disk. Fresh scenes remove scene selection, not design selection.
11. **[wording] Never write:** "memory lasts 10 frames", "the model believes", "dead reckoning" without "wall-blind", "size buys permanence", "contradicts 2610.07355", "bias-free". Replace 1M's rebound crossing "0.0 [0.0, 0.0]" with "below 6": `falls_through` returns 0 when the curve starts under chance, and 1M starts at 0.16 to 0.20.
12. **[wording] Conformal:** say "on scenes from the training generator", and state the exchangeability condition beside "valid for any observer". The 465K frames come from about 5000 clips, so they are dependent; Bates' high-probability bounds assume independent points. A demo interaction leaving the generator voids "1 in N". Not a paper contribution.

## 3. Method attacks

**Decoder power versus Gaussian head: lands partly.** Function class is not the difference: `self.head` and the xray map are both linear on the same state token. The target differs (slab-free frames, so the read may use directions the head must keep off the slab), and the centroid discards amplitude (vanish read mass at D=28 is about 0.2; a real ball is about 8). Head overlap with the true ball falls six times from D=4 to D=20 (0.25 to 0.04) while predicted ball mass only halves (0.48 to 0.23; R, 32 scenes, empty path): blur and timing more than lost presence. The unimodal head is not the sole cause, since a Bayes mixture crosses at a similar spread under 3 px. Separators: items 1 and 2, plus `mix > 1` on the final design (`e1-mix3` is an old pilot, not comparable).

**Rebound baseline: lands twice.** Decision: wall-aware dead reckoning scores 1.00 at every D tested, 0.97 to 1.00 with an assumed 0.1 px position noise (R, 64 scenes), against 5M at 0.82 falling to 0.04. The table's "dead reckoning" is wall-blind and equals no memory. Static observers are fairly beaten: frozen at last seen scores 0.06 to 0.20. Read: three quarters of a symmetric round trip is the middle of the slab. Slab midpoint at the last seen row errs 0.50, 0.68, 0.98 at D=8, 16, 28 against 5M's 0.62, 0.91, 1.21 (R, 96 scenes). C4's quoted numbers therefore show nothing.

C4 still holds on better evidence. Signed motion along x, read versus truth: return stroke sign agreement 1.00, slope 0.96 to 1.06; outward stroke 0.90 to 0.92, slope 0.60 to 0.67 (D=16, 28; R, 48 scenes). A static read scores zero and a straight line gets the return sign wrong.

**Speed confound: lands on vanish, not rebound.** Vanish D=12 PC is 0.29, 0.40, 0.72 across speed bands 0.25 to 0.4, 0.4 to 0.6, 0.6 to 0.9; the slowest band is already 0.50 at D=4. The pooled 10.1 is a mixture. Within one band the fall with D survives (0.90 to 0.19), so the effect is real but the number is not. Rebound bands agree on a crossing near D=15 to 16. Report by band.

**Design chosen on vanish: lands.** See item 10.

**Fixed 20000 steps: lands on C3.** See item 7. Learning rates differ too (1e-3, 7e-4). Sharpness is partly readout: 1M scores 0.66 at D=1, with almost no memory demand, and 0.00 at every D below speed 0.4 (seed 0).

**SKE on nonlinear signals: lands only on hidden and collision.** Efficiency by signal energy versus by magnitude: teleport 0.097, 0.096; deflect 0.011, 0.010; phantom 0.102, 0.099; hidden 0.0017, 0.0028; collision 0.0003, 0.00012 (R). The larger problem is that SKE is told the state, which overstates waste 3 to 4 times.

**C6: direction confirmed, reading not.** PC at m = -0.25, -0.1, -0.03: 0.82, 0.66, 0.65; at +0.03, +0.1, +0.25: 0.45, 0.43, 0.375 (R, 96 scenes). A parabola through median log Bayes factors peaks near restitution 0.8. A unimodal head averaging over contact timing predicts the same, and wall-blind dead reckoning, which expects no contact, scores near 0. To separate them, read the predicted post-contact speed. Collision's 1e-4 is not an efficiency.

## 4. Sources

About 7 searches and 9 paper opens, local orx. Absence means not found, not proven.

- **read-sections** (full text fetched, relevant passages read): 2610.07355 Xie and Alanwar; 2606.28455 event-conditioned diagnostics; 2610.18637 McMillen and Yilmaz, Grid or Nodes; 2305.11772 Nayebi et al.; 1602.00991 Ondruska and Posner, Deep Tracking (no bounce found); 1710.05741 Fraccaro et al., KVAE; 2104.08279 Bates et al.
- **abstract-only**: 10.1038/s41467-022-33581-6 Rajalingham et al.; 2608.15090; 2609.27179; 2610.34999; 2608.24768.
- **second-hand**, first review only: MultiEcho 2609.33347; Garrido 2025 2502.11831; N8python/physics-trainer.

Not checked: human thresholds on these stimuli; any 20M behaviour; demo latency; Itti and Baldi on video surprise; non-arXiv venues.
