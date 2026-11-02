"""Reference observers with oracle perception of every visible ball.

They differ only in what they know:
  ideal    true physics (walls, ball contacts) and memory for hidden balls: the physics limit
  cv       constant velocity, no walls, and memory for hidden balls (dead reckoning)
  cv0      constant velocity, no walls, no memory: a ball out of sight is forgotten

Each predicts the next noiseless frame; its likelihood is the true pixel noise, N(prediction, NOISE^2).
"""

import numpy as np

from . import world as W

KINDS = {"ideal": (True, True), "cv": (False, True), "cv0": (False, False)}  # (physics, memory)
SEEN = 0.02  # a ball is perceived when at least this fraction of it is visible


def render_state(scene, pos, alive):
    occ_cov, occ_lum = W.occluder_cover(scene)
    cov = W.ball_cover(pos, scene.rad, alive)
    balls = np.max(cov * scene.lum[:, :, None, None], axis=1)
    return balls * (1 - occ_cov) + occ_lum * occ_cov


def predict(traj, vis, kind):
    """Predicted noiseless frames (B, T, H, W); index t is the prediction of frame t made after seeing frame t-1."""
    physics, memory = KINDS[kind]
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
        if physics:
            first = obs & ~known
            moved = obs & known & (np.abs(p - pos).max(-1) > 1e-3)
            vel = np.where(first[..., None], traj.vel[:, t], vel)
            vel = np.where((moved & prev_obs)[..., None], fd, vel)
            pos = np.where((first | moved)[..., None], p, pos)
        else:
            pos = np.where(obs[..., None], p, pos)
            vel = np.where(obs[..., None], fd, vel)
        known = (known | obs) if memory else obs
        prev_pos = np.where(obs[..., None], p, prev_pos)
        prev_obs = obs
        if physics:
            st = {"pos": pos.copy(), "vel": vel.copy(), "rad": sc.rad, "walls": sc.walls.copy(), "alive": known.copy()}
            W._step(st)
            pos, vel = st["pos"], st["vel"]
        else:
            pos = pos + vel
        out[:, t + 1] = render_state(sc, pos, known)
    return out


def frame_nll(frames, pred):
    """(B, T) Gaussian NLL in nats per frame under the true pixel noise; frame 0 has no prediction and gets 0."""
    z = (frames - pred) / W.NOISE
    nll = (0.5 * z * z).sum((-1, -2)) + W.SIZE * W.SIZE * (np.log(W.NOISE) + 0.5 * np.log(2 * np.pi))
    nll[:, 0] = 0
    return nll
