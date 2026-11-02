# surprise-meter: adversarial novelty review

Written 2026-11-01 20:05 CST, before any training. Reviewer stance: try to break the novelty claim.
Budget used: about 20 lookups (10 `orx paper`, 6 `orx discover`, 4 `gh api` / `gh search` calls). Local orx only; no paid or remote backend.
Every source below carries a label: **read-full** (whole text read), **read-sections** (full text fetched, relevant sections read via grep and excerpts), **abstract-only**, **code-read** (repository files read), **secondhand** (only what another read paper says about it), **title-only**.
Saved raw outputs: `local research notes`.

## 0. Bottom line

| Claim | Verdict | Who already did what |
|---|---|---|
| (a) Psychometric thresholds per violation type for a learned world model | **PARTLY DONE.** The exact measurement (logistic psychometric fits over violation magnitude, 75% thresholds, seed CIs, per type) was not found. Every ingredient was. | N8python/physics-trainer: matched-twin impossible events (teleport, reversal, stop, color change, disappearance, appearance, collisions off) scored by frame-wise NLL on a 50.6M procedural-physics world model, fixed magnitudes, bootstrap CIs. Garrido 2025 / IntPhys / IntPhys 2: pairwise VoE per property, binary. MultiEcho (2609.33347): magnitude and visibility sweeps on frozen world models (response laws, not detection thresholds). |
| (b) Permanence-vs-occlusion-duration curve for a small pixel model with seeds and CIs, set against 2610.07355 | **PARTLY DONE, and the headline as worded is apples to oranges.** | 2610.07355 itself already plots belief vs seconds hidden for V-JEPA 2 at 3 scales, VideoMAE (pixel), Cosmos (autoregressive), and for predictors trained on synthetic rigid-motion scenes, which reach 1.3-1.6 s carry-over. Left: model seeds (theirs are scene seeds), intervals on carry-over (they give none), a rendered occluder, from-scratch pixel models. |
| (c) Tiny-model scale ladder | **PARTLY DONE.** | Garrido 2025 ablated encoder size (115M still above 85% on IntPhys). 2610.07355: "Scale does not buy permanence" across ViT-L/H/g. 2606.28455: GRU / Transformer-lite / RSSM-lite with occlusion events, 3 seeds (object-state inputs, not pixels). Nothing at 1M to 20M on pixels with psychophysical thresholds was found. |
| (d) Live browser surprise meter | **PARTLY DONE (demo novelty only).** | physics-trainer runs a 50.62M world model plus FSQ codec client side in hand-written WebGPU, and has a script that renders offline videos with animated frame-perplexity plots for impossible events. Its browser app is a sampler with no surprise readout. No interactive, mouse-driven, live surprise meter was found. |

The owner's headline ("a small model trained on pixels holds object permanence through occlusion longer than the published result for a far larger model suggests") does not survive. Section 4 says why and Section 6 gives the framing that does.

## 1. What 2610.07355 actually did (read-full)

Source: Peng Xie and Amr Alanwar (TUM), "Tracking Is Not Permanence: What Video World Models Keep of a Hidden Object", arXiv:2610.07355v1, 5 Oct 2026, "Preprint. Under review." Command: `orx paper 2610.07355 --full`, saved as `sources/2610.07355.full.txt`. It is a v1 preprint under review, not a "published result".

**Models.** "The model under test is the released V-JEPA 2 pretraining predictor at three scales, frozen, on rendered scenes" (ViT-L, ViT-H, ViT-g; checkpoints `facebook/vjepa2-vitl-fpc64-256`, `-vith-`, `-vitg-`). Also VideoMAE v1 (base, large) and Cosmos-Predict1-4B. V-JEPA 2-AC and V-JEPA 2.1 are explicitly not probed.

**Stimuli.** "We render 256×256 scenes at 7.5 fps: a textured background, K ∈ {0, 2, 4} moving distractor rectangles, and a ball of radius R ∈ {4, 8, 16, 32} px moving along a row at constant speed." Speed: "3.2 px per frame in 64-frame clips, 24 px/s at 7.5 fps". A checker texture on the ball. "Five seeds per cell give 60 decisions per point." Window: "We encode a 32-frame window that starts four frames before the event", so the ball is visible for 0.53 s before hiding. The visible container is "an opaque 80×88 px rectangle".

**There is no rendered occluder in the main probe.** Hiding is done by deleting tokens: "drop the tokens of the trajectory band from the event onwards, and let P infer them from the remaining tokens". The two worlds are rendered: "At the event frame the ball either continues (present) or vanishes (absent)". The only rendered occluder is the visible container (Table 6) and, for Cosmos, a box closed over the ball.

**Readout.** A two-alternative choice in latent space: "The decision P (present) is the fraction of the 60 scenes whose prediction is nearer z present in mean per-token L1, a proportion over scenes rather than a probability held by the model; chance is 0.5". Carry-over: "the last time point at which P (present) ≥ 0.5 while its control is below 0.5 ... quoted as the start time of that tubelet." Graded readouts too: a signed distance e and a projection α onto the present-absent axis. Controls: never rendered / never visible (bias floor), an exit control (ball leaves the frame when hiding starts), copy-last-view (encoder tokens of the last visible slot).

**What "0.3 s" is.** One tubelet: "One tubelet spans two frames, 0.267 s; each decision is timed by the start of its tubelet". The main result: "P (present) falling below chance between the first and second tubelet (a two-frame slot, 0.27 s) after occlusion at all three scales (0.92 to 0.00 for ViT-H)". So "lost within 0.3 s" means the decision is present in the tubelet starting at 0 s and absent in the tubelet starting at 0.27 s; the carry-over by their own definition is 0 s. In physical terms the ball travels 6.4 px in that tubelet, 0.1 to 0.8 ball diameters depending on R. Resolution and error bars: "Carry-over durations are threshold crossings of a 60-scene proportion whose binomial half-width near 0.5 is 0.13, so they are resolved to about one tubelet, and we give no interval for them."

**The 0.3 s is query-dependent.** "within 0.5 s under V-JEPA's own tube geometry (at pretraining's 90% masking ratio ViT-L and ViT-g still lose it within 0.3 s; ViT-H keeps it to 1.1 s)". Frame rate: at 15 and 30 fps ViT-H's decision is "last on the present side ... at 0.27 s", ViT-L leaves "within one tubelet at every rate".

**The decision is lost, the information is not.** "'Lost' describes the decision, not the information." Projection α for a moving ball, ViT-H: "0.43, 0.38, 0.42 and 0.33 at 0.53, 1.07, 1.87 and 2.93 s ... against 0.14, 0.12, 0.17 and 0.13 for the never-rendered control", a trace "at 14–60% of the copy baseline". The encoder reads presence "at 1.00, down to 4 px".

**The copy baseline already beats the released predictor.** "The copy baseline keeps a stationary object at 0.80–1.00 throughout and a moving object's last position for one to two seconds (0.58 at 1.07 s, 0.62 at 1.87 s), so the released predictor scores below a model that merely froze its last view."

**Scale.** Figure 2 caption: "Scale does not buy permanence: on a moving object under the main query all three released models cross to absent after one tubelet".

**What "training installs permanence" means concretely.** No small model was trained from scratch. "All start from the released encoder and, except (v), the released predictor". Five interventions (Table 2): (i) predictor-only, 20k steps, temporally-partial masks on 59k egocentric/robot clips; (ii) predictor-only, 3,000 steps, synthetic container curriculum; (iii) predictor-only, 3,000 steps, one mask kind each, synthetic; (iv) encoder + predictor, 16k steps, tube masks with EMA target on natural or internet-style video; (v) encoder + predictor, 8k steps, real hands. The abstract's line: "training installs it cheaply, as a prior: three thousand predictor-only steps on synthetic containers take this belief from 0.05 to 1.00 against two matched controls." The authors' own caveats: "The curriculum trains on the probe's own renderer, so 1.00 there is expected; over natural-video backgrounds, the one held-out test, it reads 0.85–0.97; what is installed is a prior that a reveal of the empty band does not revise", and "'Cheaply' refers to the step count, not to a measured trade-off."

**The result that pre-empts the owner's headline (Table 4).** Predictors fine-tuned 3,000 steps on the synthetic rigid-motion renderer with a single mask kind keep a moving ball: tube 1.00, 0.98, 0.97, 0.77, 0.77, 0.58, 0.53 at 0 to 1.60 s; "a predictor acquires the same 1.3–1.6 s of carry-over under each single mask kind we tried, keeps a stationary object for the whole window and keeps no ball inside a visible box". So "a model trained on this kind of synthetic physics keeps the object far longer than released V-JEPA 2" is already in the paper.

**Pixel and autoregressive comparators (the ones closest to surprise-meter's model class).** VideoMAE: "VideoMAE-L reconstructs a hidden 32 px ball for 0.3 s and the base models for none". Cosmos-Predict1-4B, teacher-forced log-likelihood of the reveal chunk (one chunk is 1.07 s), memory against the exit control: stationary box +20.7 and +18.7 nats at 1.07 and 2.13 s hidden (positive in 97% and 87% of 30 scenes), moving box −2.7 and −10.3 nats (53% and 23%).

**Small-model baseline.** None. The authors list the missing piece themselves: "a positive control with built-in memory would calibrate the probe's ceiling and is the first addition we would make", and "A calibration on a predictor trained with a known posterior, where P (present), e and α can be compared with the truth, would settle how each readout maps onto belief and is not in this paper." Also: "the curves of the released models are deterministic given the scenes, so their seeds are the scene seeds, not model seeds."

**Other facts worth knowing.** IntPhys-2019 accuracy (their fixed two-frame protocol): ViT-H 84.2%, raised to 93.3% by the container curriculum, but "without the container masks the curriculum ... gives 93.9–94.7%", so the benchmark gain is not containment. Their minimum-rule IntPhys implementation "does not reproduce their published numbers" (70.0% against 87-98% reported). Magic-trick clips: prediction error "is indistinguishable from motion energy at the human-rated moments of surprise". Compute: "roughly 150 GPU-hours" on H100/H200 plus "about 60 GPU-hours" on RTX 5090. Citation hygiene: their PLATO reference gives DOI 10.1038/s41562-022-01432-5, which resolves to the Publisher Correction (checked with `orx paper`); the article is 10.1038/s41562-022-01394-8 (OpenAlex). Cite the article DOI if you cite PLATO.

### Where the scout summary is wrong

1. "V-JEPA 2's predictor loses a hidden moving object within 0.3 s": true only for the thresholded 2AFC decision, under the dense-context query (band deleted from the event on, rest of frame visible), at 7.5 fps with 2-frame tubelets, with no interval. Under V-JEPA's own tube geometry it is 0.5 s; ViT-H at the 90% masking ratio keeps it to 1.1 s; a graded trace above control persists to 2.93 s.
2. "Training installs permanence": overstated. The paper says it installs a containment prior, in-distribution, by fine-tuning a pretrained predictor for 3,000 steps, and that this prior is not revised by contrary evidence. Nobody trained a small model from scratch.
3. "Published result": it is an arXiv v1 under review, with an AI-use statement covering hypotheses, experiments, code and drafting. Treat its numbers as unreplicated.

## 2. Closest prior work, what each already did

**N8python/physics-trainer** (code-read; `gh api repos/N8python/physics-trainer`, created 2026-07-19, 0 stars; files in `sources/n8/`). The closest artifact to surprise-meter by a wide margin.
- Model: "The default 11-layer / 512-dim configuration has 50,621,696 parameters", an autoregressive transformer over 8x8x6 FSQ tokens (64 tokens per frame, vocabulary 15,625), trained online on "64-frame, 8 FPS videos of one to six fixed-radius colored balls", 256x256 RGB, rigid-body sim at 120 Hz.
- `evaluate_impossible_events.py` ("Measure frame-wise perplexity spikes caused by impossible physics events"): variants `control`, `teleport` (to the farthest corner), `reverse` ("Instant reversal"), `stop` ("Instant stop"), `color_change`, `disappear` (ball set invisible in the open, no occluder), `appearance`; 30 matched pairs by default, same scene up to the event frame; frame-wise mean NLL; "Anomaly / matched-control PPL ratio by frame"; bootstrap with 10,000 samples.
- `evaluate_collision_ppl.py`: "Compare physics-transformer perplexity with ball collisions on versus off" (the pass-through case). `evaluate_teleport_ppl.py`: smooth vs teleport, `--teleport-dy` default 96 px (one magnitude). `evaluate_velocity_generalization.py`: in- and out-of-distribution velocities. `evaluate_anomaly_emergence.py`: "Track peak matched-scene anomaly PPL ratios across training checkpoints". `make_three_anomaly_surprisal_video.py`: "Make a 1x3 video of impossible events with animated frame-PPL plots."
- `webgpu-sim/`: "The 50.62M-parameter transformer and the complete 5.98M-parameter FSQ encoder and decoder run locally through hand-written WebGPU compute shaders." The UI ("Set the scene. Let it dream.", "Sample the next 4 seconds") is a sampler; `app.js` has zero matches for surprisal, NLL or log-prob.
- Not there: magnitude sweeps, psychometric fits, occluders, model seeds or a size ladder, a calibrated null, an ideal observer, a live surprise display. No committed results file for the impossible-event runs (repo tree lists only `TRAINING_RESULTS.md`, `TRAINING_RESULTS_8X8.md`, `VIDEO_EVAL_RESULTS.md`).
- Consequence: surprise-meter's matched-twin per-frame NLL design, four of its five violation types (teleport, speed change, pass-through, vanish), its tokenized-or-pixel small world model on procedural physics, and its WebGPU runtime all exist here in some form. Only the occlusion types and the psychophysics layer are not.

**Garrido et al. 2025, arXiv:2502.11831** (read-sections). V-JEPA VoE with prediction-error surprise; "average accuracies of 98% (95% CI [95%,99%]), 66% (95% CI [64%,68%]), 62% (95% CI [60%,63%]) respectively on IntPhys, GRASP, and InfLevel-lab"; VideoMAEv2 and MLLMs "only marginally above that of randomly-initialized models"; a scale ablation: "a 115 million parameters model still achieves an accuracy of over 85%"; "even models trained on one week of unique video achieve above chance performance"; "lower performance in both V-JEPA and humans for videos where the physics-breaking event happens behind an occluder". Max surprise per video beats average for single-video scoring. Their appendix raises the hidden-object question ("if it was shown at the beginning of the context and then hidden, the model may struggle to predict that the red ball is visible again") that 2610.07355 answered. Binary pairs, no magnitude axis.

**IntPhys 2, Bordes et al. 2025, arXiv:2506.09849** (read-sections). Four principles: Permanence, Immutability, Spatio-Temporal Continuity, Solidity. Table 2 (best of about a dozen runs per model): V-JEPA 2-h 57.51% overall, 56.40% held out, 87.22% on original IntPhys; Cosmos-4B 49.41% / 85.42%; humans 96.44% overall. Table 3, Permanence fixed / moving camera: V-JEPA 2-h 59.62 / 57.35, humans 100.0 / 99.26. Abstract: "most models performing at chance levels (50%)". Binary, no magnitude axis.

**V-JEPA 2 paper, arXiv:2506.09985** (read-sections). The orx text extraction contains no "IntPhys" string; the V-JEPA 2 IntPhys 2 numbers above come from the IntPhys 2 paper.

**PLATO, Piloto et al. 2022, Nature Human Behaviour, DOI 10.1038/s41562-022-01394-8** (abstract-only, via OpenAlex). Learns VoE responses on the Physical Concepts dataset; "our model can learn a diverse set of physical concepts, which depends critically on object-level representations". A learned model with seeds and VoE probes; object-centric inputs, binary probes. 2610.07355 (secondhand) describes it as learning "violation-of-expectation responses from object-centric inputs".

**ADEPT, Smith et al. 2019, NeurIPS, OpenAlex W2970184844** (abstract-only). "Inference integrates deep recognition networks, extended probabilistic physical simulation, and particle filtering for forming predictions and expectations across occlusion"; a developmental-psychology test set; compared to "standard network architectures" and human expectations. This is the object-tracking ideal-observer baseline that surprise-meter plans, already built and compared to humans in 2019.

**IntPhys 2019 (Riochet et al., TPAMI) and Physion (Bear et al. 2021)** (secondhand, from 2610.07355 Related Work only; not read this session). IntPhys pairs possible and impossible videos scored by prediction error; Physion scores contact prediction. Neither, as described there, has a magnitude axis.

**MultiEcho, arXiv:2609.33347, 27 Sep 2026** (read-sections). Treats frozen world models as experimental systems with "response laws"; seven model configurations (VideoSAUR, C-JEPA, LeWM, V-JEPA 2-AC, Cosmos3-Nano and others); "Magnitude sweeps expose small image errors alongside large relative effect errors"; a visibility study "lowers the ball's Weber contrast from 0.3 to 0". This is the nearest thing found to graded-magnitude psychophysics on world models, but it measures whether the model's response tracks a counterfactual intervention, not detection thresholds for physics violations.

**"Do Generative Priors Align with Human Naturalness Perception?", arXiv:2610.09928** (abstract-only). Native prediction-loss differences on "content-preserving, paired relational interventions" across 25 image and video generators "reproduce human-like selective sensitivities", including the Thatcher effect. Method overlap with matched-twin surprise psychophysics; domain is faces and illumination, not dynamics.

**Shin et al., "Can Video World Models Track Unobserved World States?", arXiv:2608.30692v2** (read-sections). Video shell game; backbones "adjusting the depth to about 200M parameters", "All models train from scratch on 30,000 five-swap episodes". Backbones "predict the correct ball location up to 5 swaps" but "fail to learn the rule and generalize to longer swap chains". From-scratch training on synthetic hidden-state tracking already exists, at 200M, with a generalization-length axis.

**"Event-Conditioned Diagnostics of Kinematic, Contact, and Object-Permanence Structure in Passive Object-State World Models", arXiv:2606.28455** (abstract-only). GRU, Transformer-lite and RSSM-lite on a controlled generator with "free-motion, collision, and occlusion events", "Across three independent seeds". Small models, seeds, occlusion; object-state inputs, not pixels; readout probes, not surprise thresholds.

**PhysTrace, arXiv:2610.31429** (read-sections). MLLM-generated, history-conditioned expectations aligned with video windows to localize violations; introduces IntPhys2-Timing; uses a V-JEPA2 VoE surprise baseline for localization. Not a psychophysics of a learned model; no overlap with (a) to (d) beyond "time-resolved surprise".

**Foresight at the Event Boundary, arXiv:2609.36531** (read-sections). 62 real free-fall recordings, 124 clips; scores whether six video generators (Cosmos-3, Cosmos-Predict-2.5, MAGI-1, PhyWorld, Runway, Veo) produce release and impact events and when. Generation-based; no surprise readout. No overlap.

**MemoBench, arXiv:2606.27537** (abstract-only). Disappear-and-reappear in changing environments for video generators. Adjacent to (b), generation-based.

**iSEE, arXiv:2610.01201** (abstract-only). Self-supervised slot model built to keep objects through occlusion. Built-in permanence, a candidate positive control.

**GitHub sweep for a live surprise meter** (`gh search repos`, 3 queries: "surprise world model physics", "violation of expectation", "surprise meter"). Nothing relevant beyond `jord534/surprise-readout` (title-only: "re-examining the zero-shot violation-of-expectation protocol on IntPhys 2."). Three queries are not proof of absence.

## 3. Verdicts in detail

### (a) Psychometric thresholds per violation type: PARTLY DONE

Already done: matched-twin per-frame NLL across violation types on a small procedural world model (physics-trainer, fixed magnitudes, 30 pairs, bootstrap); pairwise VoE accuracy per property on large models (Garrido, IntPhys 2); graded magnitude and contrast sweeps on frozen world models (MultiEcho); paired-intervention loss differences that reproduce human perceptual signatures (2610.09928). Not found: logistic psychometric functions over violation magnitude with 75% thresholds and seed CIs, per violation type, for any learned world model.

What a reviewer will still say: a threshold from a model with no human comparison is a summary statistic of a dose-response curve. "Psychophysics" earns the name only through an ideal observer on the same pixels (efficiency) or human thresholds on the same stimuli. The ideal observer is in the plan; use it as the denominator, not as a side baseline.

### (b) Permanence vs occlusion duration, set against 2610.07355: PARTLY DONE; the headline is apples to oranges

2610.07355 already has belief-vs-hidden-time curves for three V-JEPA 2 scales, a pixel model, an autoregressive model, and in-distribution-trained predictors that keep the ball 1.3-1.6 s. What is left: model seeds, intervals on carry-over, a rendered occluder with reveal-time surprise, and from-scratch pixel models. The comparison as worded fails on five counts (Section 4).

### (c) Tiny-model scale ladder: PARTLY DONE

Garrido ran a size ablation down to 115M. 2610.07355 found no scale effect across ViT-L/H/g. 2606.28455 has tiny models with seeds and occlusion, on object states. 1M/5M/20M on pixels with five seeds each is a new range, but the default expectation (bigger in-domain model, lower thresholds, saturating at the ideal) is unsurprising. It becomes a result only if something dissociates, for example kinematic thresholds improve with size while permanence does not.

### (d) Live browser surprise meter: PARTLY DONE (demo, not research)

physics-trainer already ships a client-side WebGPU runtime for a larger world model and offline animated surprisal plots. The interactive break-physics-with-the-mouse meter appears new as a demo. It carries no research novelty and should not be listed as a contribution in a paper; it is the public-facing piece.

## 4. A fair comparison to 2610.07355

### Why "our small model holds permanence longer than V-JEPA 2's 0.3 s" is apples to oranges

1. **Different hiding operation.** Their 0.3 s comes from deleting tokens of the ball's band with the rest of the frame visible, "which V-JEPA 2 never trained on". surprise-meter hides with a rendered occluder. Behind an opaque occluder the present and absent worlds are pixel-identical, so a causal pixel model's belief is readable only at the reveal. Their comparable numbers are the reveal tests: the visible container (released ViT-H, moving box: 0.05, 0.02, 0.00, 0.00 at 0.53, 1.07, 2.00, 2.93 s hidden; stationary box: 0.05, 0.03, 0.17, 0.08) and Cosmos's reveal chunk (stationary box +20.7 / +18.7 nats, moving box −2.7 / −10.3 nats against the exit control). The 0.3 s figure is not.
2. **Different readout.** Theirs is a thresholded nearest-reference decision over 60 scenes in latent space with no interval; under graded readouts the same model keeps a trace above control to 2.93 s. A graded NLL surprise from the small model compared against their thresholded decision mixes a decision with a likelihood.
3. **In-domain vs zero-shot.** The small model trains and tests on its own renderer. Released V-JEPA 2 is zero-shot on theirs. Their own in-domain fine-tunes already keep the ball 1.3-1.6 s (Table 4), and they say of in-distribution curricula that reaching ceiling "is expected". Beating a zero-shot model in-domain shows nothing about scale.
4. **Different units of time.** Their seconds are 7.5 fps tubelets of 2 frames; 0.27 s is 2 frames and 6.4 px of travel, 0.1 to 0.8 ball diameters. A procedural world's "seconds" are arbitrary. At 32x32, a direct downscale of their stimulus (factor 8) gives ball radii of 0.5 to 4 px and 0.4 px per frame, so only their R = 32 px condition is even renderable.
5. **The bar is trivially low and not about size.** Their copy-last-view baseline already beats the released predictor (0.58 at 1.07 s on a moving ball); scale did not help across ViT-L/H/g. Clearing "longer than one tubelet" says nothing about small versus large.

### What a fair comparison requires

- **Same task and controls.** Matched present/absent twins that differ only after hiding; the never-rendered (bias floor), exit, and copy-last-view controls; their three-world readout (kept moving / frozen where last seen / gone), which is what separated their training regimes.
- **Same hiding operation, run both ways.** Either (i) run released V-JEPA 2 (ViT-L is feasible on an M4 Pro) through surprise-meter's rendered-occluder reveal test with Garrido's prediction-error surprise, on the same physics rendered at 256x256, or (ii) compare only against their visible-container and Cosmos reveal numbers. The token-deletion probe cannot be applied to a causal pixel model without inpainting it was never trained for.
- **Same units.** Report hidden time in frames, in object diameters traveled while hidden, and occluder width in diameters, for both sides. Convert theirs: 1 tubelet = 2 frames = 6.4 px.
- **Same data regime, or held-out.** Either compare against an in-domain fine-tuned V-JEPA 2 (their Table 4 recipe: 3,000 predictor-only steps on the shared renderer), or test the small models out of distribution: occlusion durations, speeds and occluder widths outside the training range.
- **Comparable scale accounting.** Compare predictor parameters and training compute, not encoder size; their predictors are not 300M to 1B.
- **Intervals.** Model-seed bootstrap CIs on the carry-over, which they explicitly lack.

## 5. Design attacks that bite whatever the framing

1. **Context window confound.** A block-causal transformer whose context covers the whole clip can attend straight back to the pre-occlusion frames. Permanence vs occlusion duration will then be flat until the hidden interval exceeds the window, then collapse. That curve measures the attention window, not memory. Vary context length, use a sliding window shorter than the longest occlusion, or report duration as a fraction of the window.
2. **Pixel-count scaling.** Per-pixel Gaussian NLL scales with object area and contrast. At 32x32 a 2-pixel object's teleport is a few pixels of error, so raw thresholds partly measure rendering. Normalize by an ideal observer on the same pixels and noise model (statistical efficiency), which absorbs this.
3. **Learned variance.** A learned per-pixel variance moves surprise up or down independently of the mean prediction. Fix the noise model or give the ideal observer the same one.
4. **Null calibration leaks.** The lawful-world tail is dominated by lawful-but-rare frames: wall bounces, collisions, occluder entry and exit. A "1-in-10,000 frame" claim needs at least about 10^5 to 10^6 null frames per model and a null conditioned on event context (near a wall, at the occluder edge), or false-alarm rates differ by context.
5. **What the psychometric noise is.** With deterministic models and matched twins, the 2AFC proportion varies only over scenes. Say whether the curve's randomness is scenes or seeds, and bootstrap over seeds for model uncertainty (2610.07355's "seeds are the scene seeds" problem).
6. **Event visibility.** Pass-through and wrong-time reappearance at 32x32 with slow objects can be sub-pixel. Set the magnitude range from the ideal observer's own threshold so the curve has room on both sides.
7. **In-distribution permanence.** If every training occlusion ends in a lawful reappearance, permanence is interpolation. Hold out longer occlusions and other speeds, or the claim is "fits its renderer", which 2610.07355 already shows for V-JEPA 2.
8. **Budget.** 15 models (3 sizes x 5 seeds) in 8 Mac-hours leaves about 30 minutes per model on average, less for 20M. Check that the 20M models converge before reading anything into the ladder.

## 6. The framing that survives best

**"Ideal-observer psychophysics of tiny pixel world models: per-violation thresholds and efficiency against a known posterior, and what thresholded readouts get wrong about permanence."**

Why it survives:
- It turns the ideal observer from a side baseline into the denominator: report thresholds and statistical efficiency, (d'_model / d'_ideal)^2, per violation type, with model-seed CIs. That is what makes it psychophysics rather than curve fitting, and it absorbs the pixel-count and noise-model confounds.
- It answers gaps 2610.07355 names in its own limitations section. They list as missing a positive control with built-in memory ("the first addition we would make") and "a calibration on a predictor trained with a known posterior, where P (present), e and α can be compared with the truth". On a procedural world the Bayes posterior is computable, so you can run their decision readout and a graded likelihood readout side by side and show how each maps onto true belief. That is a contribution to the 2610.07355 line of work, not a contest with it.
- It replaces "small beats big" with a falsifiable dissociation: does the tracking-versus-permanence split appear in tiny pixel models trained from scratch, that is, do teleport and speed thresholds approach the ideal with size while permanence (reveal surprise vs hidden frames, with the three-world readout) does not? Either outcome is reportable. A pre-registered prediction makes it stronger.
- Comparisons to V-JEPA 2 stay in a clearly labeled zero-shot section, on the same rendered-occluder reveal test and in frame and diameter units.

Drop from the headline: the 0.3 s comparison, "far larger model", and "longer than". Keep the browser meter as the public demo, not a paper claim.

## 7. Sources and provenance

All lookups were made 2026-11-01. Commands ran with `ORX_NO_UPDATE_CHECK=1`; orx 0.2.19, telemetry off.

| Id | Title (short) | Link | Command | Saved file | Label |
|---|---|---|---|---|---|
| 2610.07355 | Tracking Is Not Permanence | https://www.alphaxiv.org/abs/2610.07355 | `orx paper 2610.07355 --full` | `sources/2610.07355.full.txt` | read-full |
| 2502.11831 | Intuitive physics understanding emerges from self-supervised pretraining on natural videos | https://www.alphaxiv.org/abs/2502.11831 | `orx paper 2502.11831 --full` | `sources/2502.11831.full.txt` | read-sections |
| 2506.09849 | IntPhys 2 | https://www.alphaxiv.org/abs/2506.09849 | `orx paper 2506.09849 --full` | `sources/2506.09849.full.txt` | read-sections |
| 2506.09985 | V-JEPA 2 | https://www.alphaxiv.org/abs/2506.09985 | `orx paper 2506.09985 --full` | `sources/2506.09985.full.txt` | read-sections (grep) |
| 2609.33347 | MultiEcho | https://www.alphaxiv.org/abs/2609.33347 | `orx paper 2609.33347 --full` | `sources/2609.33347.full.txt` | read-sections |
| 2608.30692 | Can Video World Models Track Unobserved World States? | https://www.alphaxiv.org/abs/2608.30692 | `orx paper 2608.30692 --full` | `sources/2608.30692.full.txt` | read-sections |
| 2610.31429 | PhysTrace | https://www.alphaxiv.org/abs/2610.31429 | `orx paper 2610.31429` | `sources/2610.31429.report.txt` | read-sections |
| 2609.36531 | Foresight at the Event Boundary | https://www.alphaxiv.org/abs/2609.36531 | `orx paper 2609.36531` | `sources/2609.36531.report.txt` | read-sections |
| 10.1038/s41562-022-01394-8 | PLATO (article) | https://doi.org/10.1038/s41562-022-01394-8 | `orx discover openalex "<ADEPT title>"` (returned PLATO too) | `sources/s5-adept-openalex.json` | abstract-only |
| 10.1038/s41562-022-01432-5 | PLATO Publisher Correction | https://doi.org/10.1038/s41562-022-01432-5 | `orx paper 10.1038/s41562-022-01432-5` | `sources/plato.txt` | read-full (correction notice) |
| W2970184844 | ADEPT | https://openalex.org/W2970184844 | `orx discover openalex` | `sources/s5-adept-openalex.json` | abstract-only |
| 2606.28455 | Event-Conditioned Diagnostics ... Object-Permanence ... | https://www.alphaxiv.org/abs/2606.28455 | `orx discover embedding` | `sources/s6-emb-graded.json` | abstract-only |
| 2610.09928 | Do Generative Priors Align with Human Naturalness Perception? | https://www.alphaxiv.org/abs/2610.09928 | `orx discover embedding` | `sources/s2-emb-psychophysics.json` | abstract-only |
| 2606.27537 | MemoBench | https://www.alphaxiv.org/abs/2606.27537 | `orx discover embedding` | `sources/s3-emb-permanence-duration.json` | abstract-only |
| 2610.01201 | iSEE | https://www.alphaxiv.org/abs/2610.01201 | `orx discover embedding` | `sources/s3-emb-permanence-duration.json` | abstract-only |
| N8python/physics-trainer | tinyWorldModel / physics-trainer | https://github.com/N8python/physics-trainer | `gh api repos/...` (README, tree, scripts, app.js) | `sources/n8python-physics-trainer.txt`, `sources/n8/` | code-read |
| GitHub search | 3 repo queries | n/a | `gh search repos ...` | `sources/s7-gh-repos.json` | title-only |
| Riochet 2022 IntPhys; Bear 2021 Physion | as cited in 2610.07355 | n/a | none | n/a | secondhand |

Discovery runs whose hits were leads only: `sources/s1-keyword-psychometric.json` (alphaXiv full-text "psychometric function video prediction model violation": no relevant hit), `s2`, `s3`, `s4`, `s6` (embedding searches).

## 8. Not checked (say so before claiming novelty in public)

- Human psychophysics thresholds for the same violation types (motion discontinuity, speed change, occlusion reappearance timing). Not searched; needed if the write-up compares model thresholds to human ones.
- Itti and Baldi's Bayesian-surprise work on video (mid-2000s) as prior art for a "surprise meter" on video: reviewer's memory only, not looked up this session. Verify before citing.
- GRASP, InfLevel, Physion and IntPhys 2019 originals: not read; any claim about their stimuli or magnitude axes is unverified.
- The impossible-event results of physics-trainer: scripts read, no results file found, so no numbers exist to compare against.
- Searches beyond about 20 lookups, and non-arXiv venues (OpenReview submissions, workshop papers, blogs) beyond what OpenAlex and alphaXiv indexed.
