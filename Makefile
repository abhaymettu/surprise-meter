PY ?= python

# Every table and figure, from the measurements in this repo. About two minutes.
results:
	$(PY) -m sm.analyze
	$(PY) -m sm.figures

# Run the measurements again from the published weights, then redo the results. About an hour on an M-series Mac.
measure: weights
	for r in runs/*-s[0-9]; do $(PY) -m sm.evaluate $$r && case $$r in *window*) ;; *) $(PY) -m sm.xray $$r ;; esac; done
	for k in ideal cv cv0; do $(PY) -m sm.evaluate $$k; done
	$(MAKE) results

weights:
	gh release download weights --pattern 'runs-weights.tar' --clobber
	tar xf runs-weights.tar && rm runs-weights.tar

# Everything from nothing: train every model, measure it, draw it. Days on one Mac. Safe to stop and rerun.
train:
	$(PY) -m sm.ladder
	$(MAKE) results

test:
	$(PY) -m pytest tests -q

demo:
	$(PY) -m http.server 8791 --bind 127.0.0.1 --directory demo

.PHONY: results measure weights train test demo
