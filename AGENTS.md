# Working in this repo

A tiny recurrent world model, measured like a psychophysics subject. Python with MLX for training and measurement, plain
JavaScript and WebGPU for the browser demo.

## Layout

- `sm/world.py` the procedural physics and renderer. `NOISE` is the one constant behind the whole likelihood, and it is mirrored in `demo/world.js` and `demo/engine.js`.
- `sm/events.py` the matched lawful and impossible twins for every family and magnitude.
- `sm/model.py` the recurrent model and the windowed control. `sm/train.py` trains and resumes.
- `sm/surprise.py` defines surprisal, excess, evidence and surprise. Read its docstring before changing anything downstream.
- `sm/observers.py` dead reckoning with and without memory. The ideal observer is in `sm/evaluate.py`.
- `sm/evaluate.py` writes `runs/<name>/eval.npz`. `sm/xray.py` fits the linear decoder and its curves. `sm/analyze.py` turns those into `results/summary.json`. `sm/figures.py` turns that into `results/figures/` and `results/tables.md`.
- `sm/ladder.py` is the experiment plan. `sm/export.py` packages a run for `demo/`.

## Commands

- `make results` rebuilds all tables and figures from the measurements in the repo.
- `make test` checks the instrument with no trained model.
- `node tools/parity.mjs` checks the browser CPU engine against MLX after `sm.export`. It needs an export first.
- `make demo` serves the demo on port 8791.
- `make measure` remeasures from the published weights. `make train` trains from nothing and takes days on one Mac.

## Rules for changes

- Every number in the README and the paper comes from `results/tables.md` or `results/summary.json`. Change the pipeline, rerun it, then change the prose.
- Changing `SEED` in `sm/evaluate.py`, a family's levels or `NOISE` invalidates every `eval.npz`. Remeasure all of them and the three reference observers together.
- Weight layout is frozen by the head shape in `sm/model.py`. Trained weights will not load after a change there.
- The browser engines must match MLX. After any model change run the parity checks before trusting the demo.
- Colors in figures and the demo differ in lightness as well as hue, with the meaning also in text or shape.
- Everything the demo says about the model has to have been measured. Check captions against `sm/evaluate.py` output or a browser trial first.
