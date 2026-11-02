"""Procedural 2D physics: balls, walls and occluders, rendered to small noisy grayscale frames.

Everything is batched numpy. A world is a `Scene` (static layout plus initial ball state); `simulate`
steps it forward, optionally applying `Edit`s at chosen frames, and returns a `Trajectory` of
per-frame ball states. `render` turns a trajectory into frames. Lawful and impossible clips are the
same scene simulated with and without an edit, so they share every frame before the edit.
"""

from dataclasses import dataclass, field, replace

import numpy as np

SIZE = 32  # frames are SIZE x SIZE pixels
NOISE = 0.15  # std of the iid Gaussian pixel noise added to every rendered frame
MAX_BALLS = 2
MAX_OCC = 2
CROSSING = 0.5  # fraction of occluded scenes that start with a ball heading for the occluder

_YY, _XX = np.meshgrid(np.arange(SIZE) + 0.5, np.arange(SIZE) + 0.5, indexing="ij")


@dataclass
class Scene:
    pos: np.ndarray  # (B, N, 2) ball centres, (x, y) in pixels
    vel: np.ndarray  # (B, N, 2) pixels per frame
    rad: np.ndarray  # (B, N)
    lum: np.ndarray  # (B, N) ball brightness
    alive: np.ndarray  # (B, N) bool; False pads scenes with fewer balls
    occ: np.ndarray  # (B, K, 4) occluder rects x0, x1, y0, y1
    occ_on: np.ndarray  # (B, K) bool
    occ_lum: np.ndarray  # (B, K)
    walls: np.ndarray = field(default=None)  # (B, N, 4) xmin, xmax, ymin, ymax a ball's edge bounces off

    def __post_init__(self):
        if self.walls is None:
            B, N = self.rad.shape
            self.walls = np.tile(np.array([0, SIZE, 0, SIZE], np.float32), (B, N, 1))

    @property
    def batch(self):
        return self.rad.shape[0]


@dataclass
class Trajectory:
    scene: Scene
    pos: np.ndarray  # (B, T, N, 2)
    vel: np.ndarray  # (B, T, N, 2)
    alive: np.ndarray  # (B, T, N)
    bounced: np.ndarray  # (B, T, N) bool: a wall or ball contact changed this ball's velocity entering frame t


@dataclass
class Edit:
    """A state change applied to the simulation just before frame `frame` is recorded.

    `fn(state, rows)` mutates the dict of arrays in place for the batch rows in `rows`. Edits are how
    impossible events enter: a teleport moves `pos`, a vanish clears `alive`, a phantom wall moves `walls`.
    """

    frame: np.ndarray  # (B,) int, -1 means no edit for that row
    fn: callable


def random_scene(rng, B, n_balls=None, n_occ=None):
    """The training distribution. Every eval scene is built to lie inside it."""
    N, K = MAX_BALLS, MAX_OCC
    nb = rng.choice([1, 2], size=B) if n_balls is None else np.full(B, n_balls)
    nk = rng.choice([0, 1, 2], size=B, p=[0.2, 0.45, 0.35]) if n_occ is None else np.full(B, n_occ)

    rad = rng.uniform(1.5, 3.0, (B, N)).astype(np.float32)
    lum = rng.uniform(0.7, 1.0, (B, N)).astype(np.float32)
    alive = np.arange(N)[None, :] < nb[:, None]
    speed = np.exp(rng.uniform(np.log(0.15), np.log(1.4), (B, N)))
    ang = rng.uniform(0, 2 * np.pi, (B, N))
    vel = np.stack([speed * np.cos(ang), speed * np.sin(ang)], -1).astype(np.float32)
    pos = rng.uniform(0, 1, (B, N, 2)) * (SIZE - 2 * rad[..., None]) + rad[..., None]

    w = rng.uniform(3, 14, (B, K))
    x0 = rng.uniform(0, SIZE - w)
    snap = rng.random((B, K))  # one occluder in five sits flush with a side wall, so bounces happen out of sight
    x0 = np.where(snap < 0.1, 0.0, np.where(snap < 0.2, SIZE - w, x0))
    full = rng.random((B, K)) < 0.6

    # Half the scenes with an occluder start a visible ball on course for it, so a clip often shows a ball seen, hidden
    # and seen again. Everything stays lawful; only the initial conditions are chosen.
    cross = (rng.random(B) < CROSSING) & (nk >= 1)
    w[:, 0] = np.where(cross, rng.uniform(6, 14, B), w[:, 0])
    x0[:, 0] = np.where(snap[:, 0] < 0.2, x0[:, 0], np.minimum(x0[:, 0], SIZE - w[:, 0]))
    x0[:, 0] = np.where(cross & (snap[:, 0] >= 0.1) & (snap[:, 0] < 0.2), SIZE - w[:, 0], x0[:, 0])
    full[:, 0] |= cross
    r = rad[:, 0]
    room_left, room_right = x0[:, 0] - 2 * r - 1, SIZE - (x0[:, 0] + w[:, 0]) - 2 * r - 1
    from_left = room_left >= room_right
    room = np.maximum(room_left, room_right)
    cross &= room > 0
    arrive = rng.uniform(2, 8, B)
    s0 = np.exp(rng.uniform(np.log(0.15), np.log(1.2), B))
    a0 = rng.uniform(-0.45, 0.45, B)
    gap = np.minimum(arrive * s0 * np.cos(a0), room)
    vx0 = np.where(from_left, 1, -1) * s0 * np.cos(a0)
    xs = np.where(from_left, x0[:, 0] - r - 0.5 - gap, x0[:, 0] + w[:, 0] + r + 0.5 + gap)
    pos[cross, 0, 0] = xs[cross]
    pos[cross, 0, 1] = rng.uniform(r + 1, SIZE - r - 1)[cross]
    vel[cross, 0, 0] = vx0[cross]
    vel[cross, 0, 1] = (s0 * np.sin(a0))[cross]

    # Resolve initial overlap between the two balls by redrawing the second one.
    for _ in range(20):
        d = np.linalg.norm(pos[:, 0] - pos[:, 1], axis=-1)
        bad = alive[:, 1] & (d < rad[:, 0] + rad[:, 1] + 0.5)
        if not bad.any():
            break
        pos[bad, 1] = rng.uniform(0, 1, (bad.sum(), 2)) * (SIZE - 2 * rad[bad, 1:2]) + rad[bad, 1:2]
    alive[:, 1] &= np.linalg.norm(pos[:, 0] - pos[:, 1], axis=-1) >= rad[:, 0] + rad[:, 1] + 0.5

    h = np.where(full, SIZE, rng.uniform(10, 28, (B, K)))
    y0 = np.where(full, 0, rng.uniform(0, SIZE - h))
    occ = np.stack([x0, x0 + w, y0, y0 + h], -1).astype(np.float32)
    occ_on = np.arange(K)[None, :] < nk[:, None]
    occ_lum = rng.uniform(0.35, 0.55, (B, K)).astype(np.float32)
    return Scene(pos.astype(np.float32), vel, rad, lum, alive, occ, occ_on, occ_lum)


def _step(st, restitution=1.0):
    """One frame of motion: move, bounce off walls, resolve the ball-ball contact. Mutates `st`."""
    pos, vel, rad, walls, alive = st["pos"], st["vel"], st["rad"], st["walls"], st["alive"]
    v_before = vel.copy()
    pos += vel
    for axis, (lo, hi) in enumerate([(0, 1), (2, 3)]):
        low = walls[..., lo] + rad
        high = walls[..., hi] - rad
        under = pos[..., axis] < low
        over = pos[..., axis] > high
        pos[..., axis] = np.where(under, 2 * low - pos[..., axis], pos[..., axis])
        pos[..., axis] = np.where(over, 2 * high - pos[..., axis], pos[..., axis])
        vel[..., axis] = np.where(under | over, -vel[..., axis], vel[..., axis])

    if pos.shape[1] >= 2:
        d = pos[:, 1] - pos[:, 0]
        dist = np.linalg.norm(d, axis=-1)
        n = d / np.maximum(dist, 1e-6)[:, None]
        rel = np.sum((vel[:, 1] - vel[:, 0]) * n, -1)
        touch = alive[:, 0] & alive[:, 1] & (dist < rad[:, 0] + rad[:, 1]) & (rel < 0)
        m0, m1 = rad[:, 0] ** 2, rad[:, 1] ** 2
        j = np.where(touch, (1 + restitution) * rel / (m0 + m1), 0)[:, None] * n
        vel[:, 0] += j * m1[:, None]
        vel[:, 1] -= j * m0[:, None]
        overlap = np.where(touch, rad[:, 0] + rad[:, 1] - dist, 0)[:, None] * n
        pos[:, 0] -= overlap * 0.5
        pos[:, 1] += overlap * 0.5
    return np.any(np.abs(vel - v_before) > 1e-6, axis=-1)


def simulate(scene, T, edits=(), restitution=1.0):
    st = {
        "pos": scene.pos.copy(),
        "vel": scene.vel.copy(),
        "rad": scene.rad.copy(),
        "walls": scene.walls.copy(),
        "alive": scene.alive.copy(),
    }
    B, N = scene.rad.shape
    P = np.zeros((B, T, N, 2), np.float32)
    V = np.zeros((B, T, N, 2), np.float32)
    A = np.zeros((B, T, N), bool)
    BO = np.zeros((B, T, N), bool)
    for t in range(T):
        if t > 0:
            BO[:, t] = _step(st, restitution)
        for e in edits:
            rows = np.nonzero(e.frame == t)[0]
            if len(rows):
                e.fn(st, rows)
        P[:, t], V[:, t], A[:, t] = st["pos"], st["vel"], st["alive"]
    return Trajectory(scene, P, V, A, BO)


def occluder_cover(scene):
    """(B, H, W) fraction of each pixel covered by some occluder, and the occluder layer's brightness."""
    o = scene.occ[:, :, None, None, :]
    cx = np.clip(np.minimum(_XX + 0.5, o[..., 1]) - np.maximum(_XX - 0.5, o[..., 0]), 0, 1)
    cy = np.clip(np.minimum(_YY + 0.5, o[..., 3]) - np.maximum(_YY - 0.5, o[..., 2]), 0, 1)
    cov = cx * cy * scene.occ_on[:, :, None, None]
    total = 1 - np.prod(1 - cov, axis=1)
    lum = np.sum(cov * scene.occ_lum[:, :, None, None], 1) / np.maximum(np.sum(cov, 1), 1e-6)
    return total.astype(np.float32), lum.astype(np.float32)


def ball_cover(pos, rad, alive):
    """(..., N, H, W) antialiased disc coverage for balls at `pos` (..., N, 2)."""
    dx = _XX - pos[..., 0, None, None]
    dy = _YY - pos[..., 1, None, None]
    cov = np.clip(rad[..., None, None] + 0.5 - np.sqrt(dx * dx + dy * dy), 0, 1)
    return cov * alive[..., None, None]


def render(traj, rng=None, noise=NOISE):
    """Frames (B, T, H, W) and per-ball visible fraction (B, T, N)."""
    sc = traj.scene
    occ_cov, occ_lum = occluder_cover(sc)
    rad = np.broadcast_to(sc.rad[:, None], traj.alive.shape)
    lum = sc.lum[:, None, :, None, None]
    cov = ball_cover(traj.pos, rad, traj.alive)  # (B, T, N, H, W)
    balls = np.max(cov * lum, axis=2)
    oc = occ_cov[:, None]
    img = balls * (1 - oc) + occ_lum[:, None] * oc
    mass = cov.sum((-1, -2))
    seen = (cov * (1 - oc[:, :, None])).sum((-1, -2))
    visible = np.where(mass > 0, seen / np.maximum(mass, 1e-6), 0)
    if rng is not None and noise > 0:
        img = img + rng.normal(0, noise, img.shape)
    return img.astype(np.float32), visible.astype(np.float32)


def visible(traj):
    """(B, T) visible fraction of ball 0, without the cost of rendering frames."""
    sc = traj.scene
    occ_cov, _ = occluder_cover(sc)
    cov = ball_cover(traj.pos[:, :, 0], sc.rad[:, None, 0], traj.alive[:, :, 0])
    mass = cov.sum((-1, -2))
    seen = (cov * (1 - occ_cov[:, None])).sum((-1, -2))
    return np.where(mass > 0, seen / np.maximum(mass, 1e-6), 0)


def train_batch(rng, B, T):
    """A batch of lawful clips from the training distribution: noisy frames (B, T, H, W)."""
    traj = simulate(random_scene(rng, B), T)
    frames, _ = render(traj, rng)
    return frames


def blank_mask(rng, B, T, rate, max_len=24):
    """(B, T, 16) observation dropout: with probability `rate` a clip loses one stretch of frames, either whole
    frames or a tube of one or two patch columns, starting after the velocity could have been seen."""
    mask = np.zeros((B, T, 16), np.float32)
    for b in np.nonzero(rng.random(B) < rate)[0]:
        n = rng.integers(1, max_len + 1)
        t0 = rng.integers(4, max(T - 1 - n, 5))
        if rng.random() < 0.5:
            mask[b, t0 : t0 + n] = 1
        else:
            c = rng.integers(0, 4)
            cols = [c, min(c + 1, 3)] if rng.random() < 0.5 else [c]
            for col in cols:
                mask[b, t0 : t0 + n, col :: 4] = 1
    return mask


def with_scene(traj, **kw):
    return replace(traj, scene=replace(traj.scene, **kw))
