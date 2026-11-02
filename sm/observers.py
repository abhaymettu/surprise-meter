"""Reference observers, all with the true pixel noise as their likelihood, N(prediction, NOISE^2).

  ideal    knows the lawful continuation of every clip exactly. This is the signal-known-exactly observer of
           detection theory, and no observer can separate the twins better
  cv       sees every visible ball perfectly, assumes constant velocity and no walls, remembers hidden balls
  cv0      the same, but a ball out of sight is forgotten
"""

import numpy as np

from . import world as W

KINDS = {"cv": True, "cv0": False}  # memory for hidden balls
SEEN = 0.02  # a ball is perceived when at least this fraction of it is visible


def render_state(scene, pos, alive):
    occ_cov, occ_lum = W.occluder_cover(scene)
    cov = W.ball_cover(pos, scene.rad, alive)
    balls = np.max(cov * scene.lum[:, :, None, None], axis=1)
    return balls * (1 - occ_cov) + occ_lum * occ_cov


def predict(traj, vis, kind):
    """Predicted noiseless frames (B, T, H, W); index t is the prediction of frame t made after seeing frame t-1."""
    memory = KINDS[kind]
    sc = traj.scene
    B, T, N = traj.alive.shape
    pos = np.zeros((B, N, 2), np.float32)
    vel = np.zeros((B, N, 2), np.float32)
    known = np.zeros((B, N), bool)
    prev_obs = np.zeros((B, N), bool)
    prev_pos = np.zeros((B, N, 2), np.float32)
    out = np.zeros((B, T, W.SIZE, W.SIZE), np.float32)
    for t in range(T - 1):
        obs = vis[:, t] > SEEN
        p = traj.pos[:, t]
        fd = np.where((prev_obs & obs)[..., None], p - prev_pos, 0)
        pos = np.where(obs[..., None], p, pos)
        vel = np.where(obs[..., None], fd, vel)
        known = (known | obs) if memory else obs
        prev_pos = np.where(obs[..., None], p, prev_pos)
        prev_obs = obs
        pos = pos + vel
        out[:, t + 1] = render_state(sc, pos, known)
    return out


def frame_nll(frames, pred):
    """(B, T) Gaussian NLL in nats per frame under the true pixel noise; frame 0 has no prediction and gets 0."""
    z = (frames - pred) / W.NOISE
    nll = (0.5 * z * z).sum((-1, -2)) + W.SIZE * W.SIZE * (np.log(W.NOISE) + 0.5 * np.log(2 * np.pi))
    nll[:, 0] = 0
    return nll
