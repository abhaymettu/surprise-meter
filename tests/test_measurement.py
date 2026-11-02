"""The instrument, checked without any trained model: twins, the ideal observer, the null and the thresholds."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from sm import analyze as A
from sm import events as E
from sm import evaluate as V
from sm import model as M
from sm import surprise as S
from sm import world as W


@pytest.mark.parametrize("fam,level", [("teleport", 1.0), ("deflect", 32), ("phantom", 2.0), ("passthrough", 1.0), ("collision", 1.0), ("vanish", 12), ("rebound", 16), ("hidden", 2.0)])
def test_twins_share_noise_and_differ_only_from_the_fork(fam, level):
    p = E.FAMILIES[fam]["fn"](np.random.default_rng(1), 8, level)
    for b in range(8):
        start = p.window[b, 0]
        assert np.abs(p.lawful[b, :start] - p.impossible[b, :start]).max() < 0.02  # far below the pixel noise of 0.15
        assert not np.array_equal(p.lawful[b, start:], p.impossible[b, start:])


@pytest.mark.parametrize("fam", ["vanish", "rebound"])
def test_occlusion_twins_differ_only_after_the_hidden_interval(fam):
    D = 16
    p = E.FAMILIES[fam]["fn"](np.random.default_rng(2), 8, D)
    for b in range(8):
        hidden = p.law_vis[b, p.window[b, 0] - D : p.window[b, 0], 0]
        assert (hidden < 1e-3).sum() >= D - 1  # the fork is the first changed pixel, which can lead visibility by a frame


def test_control_clip_is_the_impossible_clip_without_a_ball_once_hidden():
    p = E.vanish(np.random.default_rng(3), 8, 12)
    for b in range(8):
        t = p.window[b, 0]
        assert np.allclose(p.control[b, t:], p.impossible[b, t:])
        assert not np.allclose(p.control[b, :t], p.impossible[b, :t])


def test_ideal_observer_separates_every_large_event():
    obs = V.ReferenceObserver("ideal")
    for fam, level in [("teleport", 2.0), ("vanish", 24), ("rebound", 24), ("collision", 2.0)]:
        p = E.FAMILIES[fam]["fn"](np.random.default_rng(4), 16, level)
        nl, _, ni, _ = obs.twins(p)
        assert (V.window_sum(ni, p.window) > V.window_sum(nl, p.window)).all()


def test_memoryless_tracker_prefers_a_world_where_the_ball_stays_gone():
    p = E.vanish(np.random.default_rng(5), 32, 8)
    nl, _, ni, _ = V.ReferenceObserver("cv0").twins(p)
    assert np.mean(V.window_sum(ni, p.window) > V.window_sum(nl, p.window)) < 0.3


def test_dead_reckoning_gets_vanish_right_and_rebound_wrong():
    cv = V.ReferenceObserver("cv")
    v = E.vanish(np.random.default_rng(6), 32, 16)
    nl, _, ni, _ = cv.twins(v)
    assert np.mean(V.window_sum(ni, v.window) > V.window_sum(nl, v.window)) > 0.9
    r = E.rebound(np.random.default_rng(7), 32, 16)
    nl, _, ni, _ = cv.twins(r)
    assert np.mean(V.window_sum(ni, r.window) > V.window_sum(nl, r.window)) < 0.3


def test_conformal_p_values_hold_their_promise_on_new_draws():
    rng = np.random.default_rng(8)
    null = S.Null(rng.gumbel(size=20000))
    p = null.p(rng.gumbel(size=20000))
    for alpha in (0.1, 0.01, 0.001):
        assert abs(np.mean(p <= alpha) - alpha) < 3 * np.sqrt(alpha / 20000) + 1e-3


def test_null_tail_gives_the_same_cut_as_the_full_table():
    null = S.Null(np.random.default_rng(9).normal(size=100000))
    ranks, tail = null.tail()
    ev = {"null_n": len(null.sorted), "null_ranks": ranks, "null_tail": tail}
    assert A.cut(ev, 1e-3) == pytest.approx(null.cut(1e-3), abs=0.02)


def test_threshold_fit_recovers_a_known_curve():
    x = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0])
    true = lambda v: 0.5 + 0.49 / (1 + np.exp(-(np.log(v) - np.log(0.2)) / 0.5))
    t, _ = A.fit_threshold(x, true(x), 10000, 0.5, 0.75)
    assert t == pytest.approx(0.2, rel=0.1)


def test_threshold_reports_unbracketed_curves_as_such():
    assert A.fit_threshold([1, 2, 4], [0.5, 0.55, 0.6], 100, None, 0.75)[0] == float("inf")
    assert A.fit_threshold([1, 2, 4], [0.8, 0.9, 1.0], 100, 0.5, 0.75)[0] == 0.0


def test_falls_through():
    assert A.falls_through(np.array([1.0, 0.75, 0.25]), [1, 2, 4], 0.5) == pytest.approx(3.0)
    assert A.falls_through(np.array([0.4, 0.3]), [1, 2], 0.5) == 0.0
    assert A.falls_through(np.array([0.9, 0.8]), [1, 2], 0.5) == 2.0


def test_noise_is_the_constant_the_browser_uses():
    import re
    from pathlib import Path
    for f in ("demo/world.js", "demo/engine.js"):
        assert float(re.search(r"NOISE = ([0-9.]+)", Path(f).read_text()).group(1)) == W.NOISE


def test_reader_gap_finds_the_longer_lasting_reader():
    levels = [4, 8, 12, 16, 20, 24]
    rng = np.random.default_rng(0)
    def margin(cross):
        return {D: np.where(rng.random(200) < (0.95 if D < cross else 0.05), 1.0, -1.0) for D in levels}
    seeds = []
    for _ in range(2):
        head, read, mean, refit = margin(10), margin(18), margin(10), margin(10)
        seeds.append({f"vanish/{D}/{k}{w}": curve[D] for D in levels for k, curve in (("head", head), ("mean", mean), ("refit", refit), ("visible", refit), ("read", read)) for w in ("", "_full")})
    out = A.reader_gaps(seeds, "vanish", levels, np.random.default_rng(1))
    assert 8 <= out["head_crossing"] <= 12
    assert 16 <= out["read_crossing"] <= 20
    assert out["read_minus_head_ci"][0] > 0
    assert abs(out["mean_minus_head"]) < 3



def test_rebuilding_frames_for_the_backward_pass_leaves_the_gradient_alone():
    frames = mx.array(np.random.default_rng(0).normal(0.25, 0.15, (2, 6, 32, 32)).astype(np.float32))
    blank = mx.zeros((2, 5, M.TOKENS))
    grads = []
    for recompute in (0, 4):
        mx.random.seed(0)
        model = M.RecurrentModel(M.Config(d=32, layers=2, heads=2, recompute=recompute))
        grads.append(tree_flatten(nn.value_and_grad(model, lambda m: M.frame_nll(m, frames, blank).mean())(model)[1]))
    assert len(grads[0]) > 0
    for (name, a), (_, b) in zip(*grads):
        assert np.allclose(np.array(a), np.array(b), atol=1e-4), name


def test_window_scores_do_not_depend_on_how_clips_are_chunked():
    mx.random.seed(0)
    observer = V.ModelObserver(M.WindowModel(M.Config(d=32, layers=1, heads=2, arch="window", window=8)))
    frames = np.random.default_rng(0).normal(0.25, 0.15, (8, 24, 32, 32)).astype(np.float32)
    observer.chunk = 8
    whole = observer.score(frames)
    observer.chunk = 3
    parts = observer.score(frames)
    for a, b in zip(whole, parts):
        assert np.allclose(a[:, 1:], b[:, 1:], atol=1e-3)

@pytest.mark.parametrize("fam,level", [("vanish", 8), ("exit", 8), ("teleport", 1.0)])
def test_shared_prefix_scores_each_twin_as_a_full_rollout_does(fam, level):
    mx.random.seed(0)
    model = M.RecurrentModel(M.Config(d=32, layers=1, heads=2))
    observer = V.ModelObserver(model)
    p = E.FAMILIES[fam]["fn"](np.random.default_rng(0), 4, level)
    _, _, imp, _ = observer.twins(p)
    whole = observer.score(p.impossible)[0]
    for b in range(4):
        start = p.fork[b]
        assert np.allclose(imp[b, start:], whole[b, start:], atol=1e-2)
