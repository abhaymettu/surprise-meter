"""What "surprise" means, in one place.

  surprisal   h_t = -log p(x_t | x_<t), nats, summed over pixels. Comparing h across matched twins is a log
              Bayes factor between the two worlds: positive means the observer found the impossible twin less
              likely. This is the bias-free 2AFC readout.
  excess      (z^2 - 1) / 2 per pixel, with z the pixel's error in the observer's own predicted sigmas. Mean
              zero when the observer is calibrated and a pixel is lawful, large only where it was confident
              and wrong.
  evidence    m_t, the largest excess summed over any 12x12 window of a frame. A violation is local, so this
              keeps the noise of the other 900 pixels out of it.
  surprise    the conformal tail of m_t among held-out lawful frames: p_t = (1 + #{null >= m_t}) / (n + 1).
              "This frame is a 1 in 1/p_t event in lawful physics." It needs no noise model and is valid for
              any observer, which is what makes a tiny network and an ideal observer comparable.
"""

import mlx.core as mx
import numpy as np

WINDOW, STRIDE = 12, 4
BURN_IN = 4  # velocity is unknowable in the first frames, so they are not scored


def frame_stats(x, mean, sigma):
    """x, mean, sigma (B, H, W) numpy or mlx -> surprisal (B,) and evidence (B,), same library as the input."""
    z = (x - mean) / sigma
    excess = 0.5 * (z * z - 1)
    nll = (0.5 * z * z + _log(sigma) + 0.9189385332046727).sum((-1, -2))
    B, H = excess.shape[:2]
    best = None
    for y in range(0, H - WINDOW + 1, STRIDE):
        rows = excess[:, y : y + WINDOW].sum(1)
        for u in range(0, H - WINDOW + 1, STRIDE):
            s = rows[:, u : u + WINDOW].sum(-1)
            best = s if best is None else _max(best, s)
    return nll, best


def _log(a):
    return np.log(a) if isinstance(a, np.ndarray) else mx.log(a)


def _max(a, b):
    return np.maximum(a, b) if isinstance(a, np.ndarray) else mx.maximum(a, b)


class Null:
    """The sorted evidence of lawful frames, and the p-value map it defines."""

    def __init__(self, evidence):
        self.sorted = np.sort(np.asarray(evidence, np.float64))

    def p(self, m):
        n = len(self.sorted)
        return (1 + n - np.searchsorted(self.sorted, m, side="left")) / (n + 1)

    def cut(self, alpha):
        """The evidence level that a fraction alpha of lawful frames reach."""
        return float(self.sorted[min(int(np.ceil(len(self.sorted) * (1 - alpha))), len(self.sorted) - 1)])
