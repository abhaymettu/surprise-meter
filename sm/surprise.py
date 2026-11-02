"""What "surprise" means, in one place.

  surprisal   h_t = -log p(x_t | x_<t), nats, summed over pixels. Comparing h across matched twins is a log
              Bayes factor between the two worlds: positive means the observer found the impossible twin less
              likely. This is the bias-free 2AFC readout.
  excess      a patch's surprisal minus the entropy the observer itself predicted for that patch. Near zero when
              the observer is calibrated and the patch is lawful, large only where it was confident and wrong.
  evidence    m_t, the largest excess over any 2x2 block of patches (16x16 pixels). A violation is local, so
              this keeps the noise of the rest of the frame out of it.
  surprise    the conformal tail of m_t among held-out lawful frames: p_t = (1 + #{null >= m_t}) / (n + 1).
              "This frame is a 1 in 1/p_t event in lawful physics." It needs no noise model and is valid for
              any observer, which is what makes a tiny network and an ideal observer comparable.
"""

import numpy as np

BURN_IN = 4  # velocity is unknowable in the first frames, so they are not scored


def evidence(nll, entropy):
    """Per-patch nll and entropy (B, 16) -> surprisal (B,) and evidence (B,), the largest excess over any 2x2 patches."""
    e = (nll - entropy).reshape(nll.shape[0], 4, 4)
    pairs = e[:, :-1] + e[:, 1:]
    return nll.sum(-1), (pairs[:, :, :-1] + pairs[:, :, 1:]).reshape(nll.shape[0], -1).max(-1)


def gaussian_patches(x, mean, sigma):
    """A per-pixel Gaussian observer, numpy (B, H, W) -> per-patch nll and entropy (B, 16)."""
    c = 0.5 * np.log(2 * np.pi)
    z = (x - mean) / sigma
    nll = 0.5 * z * z + np.log(sigma) + c
    ent = np.log(sigma) + 0.5 + c
    B = len(x)
    pool = lambda a: a.reshape(B, 4, 8, 4, 8).sum((2, 4)).reshape(B, 16)
    return pool(nll), pool(ent)


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
