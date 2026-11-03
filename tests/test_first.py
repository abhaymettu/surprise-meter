import numpy as np

from sm.first import K, matched, share


def test_matched_directions_are_orthonormal_and_carry_the_target_share():
    x = np.random.default_rng(0).normal(size=(4000, 64)) * np.geomspace(0.01, 1, 64)
    x -= x.mean(0)
    q = matched(x, 0.004)
    assert q.shape == (64, K)
    assert np.allclose(q.T @ q, np.eye(K), atol=1e-5)
    assert 0.002 < share(x, q) < 0.008
