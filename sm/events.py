"""Matched lawful / impossible clip pairs for each kind of violation, at a chosen magnitude.

Both twins come from one scene and share the same pixel noise, so they are identical up to the event.
Each family puts its event inside the training distribution's range of speeds, sizes and layouts.

  teleport     a visible ball jumps d px along its own path (forward or back)
  deflect      a visible ball turns by theta degrees with nothing to hit
  phantom      a ball bounces d px short of the wall, touching nothing
  passthrough  a ball goes through the wall and leaves the frame
  collision    two balls meet and leave with restitution 1 - m instead of 1 (m = 2 passes through)
  vanish       a ball hidden for D frames behind an occluder never comes out
  hidden       a ball hidden for D frames jumps d px along its path while out of sight
  rebound      a ball bounces off a wall behind an occluder, hidden for D frames, and never comes back
  exit         a ball leaves through the wall in plain view, D frames before the scored empty frames: the trace's null
"""

from dataclasses import dataclass, replace

import numpy as np

from . import world as W

CONTROL_CLIPS = 192
REVEAL_PX = 3.0  # an occluded event is scored until the lawful ball has come this far out
REVEAL_FULL_PX = 9.0  # and again through full emergence: within 3 px "late" still explains an empty frame
HIDDEN_D_MAX = 3.0  # hidden jumps of every size use scenes that fit the largest, so speed does not change with the level


@dataclass
class Pairs:
    family: str
    magnitude: float
    duration: int  # frames fully hidden (vanish, hidden); 0 otherwise
    lawful: np.ndarray  # (B, T, H, W) noisy frames
    impossible: np.ndarray
    control: np.ndarray  # (C, T, H, W) occlusion families: same layout, no ball ever, same noise; else None
    law: W.Trajectory
    imp: W.Trajectory
    law_vis: np.ndarray  # (B, T, N) visible fraction
    imp_vis: np.ndarray
    window: np.ndarray  # (B, 2) [start, end) frames scored by the 2AFC
    window_full: np.ndarray = None  # occluded events: the same start, through full emergence
    sign: np.ndarray = None  # (B,) direction of a signed edit
    fork: np.ndarray = None  # (B,) the first frame where the twins differ; a scored window can start later


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-6)


def _one_ball(rng, n, speed_lo, speed_hi, margin=1.0):
    sc = W.random_scene(rng, n, n_balls=1, n_occ=0)
    r = sc.rad[:, 0]
    speed = np.exp(rng.uniform(np.log(speed_lo), np.log(speed_hi), n))
    ang = rng.uniform(0, 2 * np.pi, n)
    sc.vel[:, 0] = np.stack([speed * np.cos(ang), speed * np.sin(ang)], -1)
    sc.pos[:, 0] = rng.uniform(0, 1, (n, 2)) * (W.SIZE - 2 * (r[:, None] + margin)) + r[:, None] + margin
    return sc


def _subset(sc, idx):
    return W.Scene(**{k: getattr(sc, k)[idx] for k in ("pos", "vel", "rad", "lum", "alive", "occ", "occ_on", "occ_lum", "walls")})


def _first_diff(law_clean, imp_clean):
    diff = np.abs(law_clean - imp_clean).max((-1, -2)) > 1e-3  # (B, T)
    return np.where(diff.any(1), diff.argmax(1), law_clean.shape[1])


def _finish(family, mag, dur, sc, law, imp, rng_noise_seed, length, length_full=None, sign=None):
    law_clean, law_vis = W.render(law)
    imp_clean, imp_vis = W.render(imp)
    noise = np.random.default_rng(rng_noise_seed).normal(0, W.NOISE, law_clean.shape).astype(np.float32)
    td = _first_diff(law_clean, imp_clean)
    win = np.stack([td, np.minimum(td + length, law_clean.shape[1])], -1)
    control = None
    if dur:
        none = W.simulate(replace(sc, alive=np.zeros_like(sc.alive)), law_clean.shape[1])
        control = W.render(none)[0][:CONTROL_CLIPS] + noise[:CONTROL_CLIPS]
    full = win if length_full is None else np.stack([td, np.minimum(td + length_full, law_clean.shape[1])], -1)
    return Pairs(family, mag, dur, law_clean + noise, imp_clean + noise, control, law, imp, law_vis, imp_vis, win, full, sign, td)


def _collect(rng, B, propose):
    """Call `propose(n)` until B accepted rows exist. propose returns (scene, ok_mask, extra dict of arrays)."""
    scenes, extras, have = [], [], 0
    for _ in range(3000):
        sc, ok, extra = propose(max(4 * (B - have), 128))
        idx = np.nonzero(ok)[0][: B - have]
        if len(idx):
            scenes.append(_subset(sc, idx))
            extras.append({k: v[idx] for k, v in extra.items()})
            have += len(idx)
        if have >= B:
            break
    if have < B:
        raise RuntimeError(f"could only build {have} of {B} scenes")
    cat = lambda xs: np.concatenate(xs)
    sc = W.Scene(**{k: cat([getattr(s, k) for s in scenes]) for k in ("pos", "vel", "rad", "lum", "alive", "occ", "occ_on", "occ_lum", "walls")})
    return sc, {k: cat([e[k] for e in extras]) for k in extras[0]}


def _no_bounce(traj, lo, hi):
    return ~traj.bounced[:, lo:hi].any((1, 2))


def teleport(rng, B, d, t_e=12, T=20):
    def jump(sign):
        def fn(st, rows):
            st["pos"][rows, 0] += (sign[rows] * d)[:, None] * _unit(st["vel"][rows, 0])

        return fn

    def propose(n):
        sc = _one_ball(rng, n, 0.3, 1.0)
        sign = rng.choice([-1.0, 1.0], n)
        law = W.simulate(sc, T)
        imp = W.simulate(sc, T, [W.Edit(np.full(n, t_e), jump(sign))])
        ok = _no_bounce(law, t_e - 6, T) & _no_bounce(imp, t_e - 6, T)
        return sc, ok, {"sign": sign}

    sc, ex = _collect(rng, B, propose)
    law = W.simulate(sc, T)
    imp = W.simulate(sc, T, [W.Edit(np.full(B, t_e), jump(ex["sign"]))])
    return _finish("teleport", d, 0, sc, law, imp, rng.integers(1 << 31), 6, sign=ex["sign"])


def deflect(rng, B, deg, t_e=10, T=22):
    def turn(sign):
        def fn(st, rows):
            a = np.deg2rad(deg) * sign[rows]
            v = st["vel"][rows, 0].copy()
            st["vel"][rows, 0, 0] = v[:, 0] * np.cos(a) - v[:, 1] * np.sin(a)
            st["vel"][rows, 0, 1] = v[:, 0] * np.sin(a) + v[:, 1] * np.cos(a)

        return fn

    def propose(n):
        sc = _one_ball(rng, n, 0.5, 1.0)
        sign = rng.choice([-1.0, 1.0], n)
        law = W.simulate(sc, T)
        imp = W.simulate(sc, T, [W.Edit(np.full(n, t_e), turn(sign))])
        ok = _no_bounce(law, t_e - 6, T) & _no_bounce(imp, t_e - 6, T)
        return sc, ok, {"sign": sign}

    sc, ex = _collect(rng, B, propose)
    law = W.simulate(sc, T)
    imp = W.simulate(sc, T, [W.Edit(np.full(B, t_e), turn(ex["sign"]))])
    return _finish("deflect", deg, 0, sc, law, imp, rng.integers(1 << 31), 10, sign=ex["sign"])


def _wall_scene(rng, n, t_hit=12):
    """One ball heading for a random wall, arriving about t_hit frames in."""
    sc = _one_ball(rng, n, 0.3, 1.0)
    r = sc.rad[:, 0]
    side = rng.integers(0, 4, n)  # 0 left, 1 right, 2 top, 3 bottom
    vn = rng.uniform(0.4, 1.0, n)
    vt = rng.uniform(-0.4, 0.4, n)
    gap = vn * (t_hit + rng.uniform(0, 1, n))
    tang = rng.uniform(r + 1, W.SIZE - r - 1)
    axis = side // 2
    toward_max = side % 2 == 1
    normal = np.where(toward_max, W.SIZE - r - gap, r + gap)
    vnorm = np.where(toward_max, vn, -vn)
    sc.pos[:, 0, 0] = np.where(axis == 0, normal, tang)
    sc.pos[:, 0, 1] = np.where(axis == 0, tang, normal)
    sc.vel[:, 0, 0] = np.where(axis == 0, vnorm, vt)
    sc.vel[:, 0, 1] = np.where(axis == 0, vt, vnorm)
    return sc, side


def _wall_family(rng, B, family, change, T=26, length=8, settled=26):
    def edit(side):
        def fn(st, rows):
            change(st["walls"], rows, side[rows])

        return fn

    def propose(n):
        sc, side = _wall_scene(rng, n)
        law = W.simulate(sc, T)
        imp = W.simulate(sc, T, [W.Edit(np.zeros(n, int), edit(side))])
        ok = (law.bounced[:, 4:settled].sum((1, 2)) == 1) & (imp.bounced[:, 4:settled].sum((1, 2)) <= 1)
        return sc, ok, {"side": side}

    sc, ex = _collect(rng, B, propose)
    law = W.simulate(sc, T)
    imp = W.simulate(sc, T, [W.Edit(np.zeros(B, int), edit(ex["side"]))])
    return sc, law, imp, length


def phantom(rng, B, d):
    def change(walls, rows, side):
        walls[rows, 0, side] += np.where(side % 2 == 0, d, -d)

    sc, law, imp, length = _wall_family(rng, B, "phantom", change)
    return _finish("phantom", d, 0, sc, law, imp, rng.integers(1 << 31), length)


def passthrough(rng, B, d=1.0):
    def change(walls, rows, side):
        walls[rows, 0, side] = np.where(side % 2 == 0, -1e4, 1e4)

    sc, law, imp, length = _wall_family(rng, B, "passthrough", change)
    return _finish("passthrough", d, 0, sc, law, imp, rng.integers(1 << 31), length)


def _occluded_scene(rng, n, D, d_min=0.0):
    """One ball crossing one full-height occluder, built to be fully hidden for about D frames."""
    sc = W.random_scene(rng, n, n_balls=1, n_occ=1)
    r = rng.uniform(1.6, 2.4, n).astype(np.float32)
    sc.rad[:, 0] = r
    half = D // 2
    margin = np.minimum(0.25 + half, D - 0.75 - half)
    vlo = np.maximum(0.15, (d_min + 0.15) / max(margin, 0.25))
    vhi = np.minimum(1.3, (13 - 2 * r) / (D - 0.5))
    vx = np.exp(rng.uniform(np.log(vlo), np.log(np.maximum(vhi, vlo))))
    w = (D - 0.5) * vx + 2 * r + rng.uniform(0.5, 1.5, n)
    a = np.ceil(4 + (2 * r + 2) / vx)
    x0_lo = a * vx - 0.25 * vx - 0.8
    x0_hi = W.SIZE - 1 - 2 * r - (D + 5.25) * vx
    x0 = x0_lo + rng.uniform(0, 1, n) * np.maximum(x0_hi - x0_lo, 0)
    vy = vx * np.tan(np.deg2rad(rng.uniform(-15, 15, n)))
    x_start = x0 + r + 1 + 0.25 * vx - a * vx
    y_start = rng.uniform(r + 3, W.SIZE - r - 3)
    sc.pos[:, 0] = np.stack([x_start, y_start], -1)
    sc.vel[:, 0] = np.stack([vx, vy], -1)
    sc.occ[:, 0] = np.stack([x0, x0 + w, np.zeros(n), np.full(n, W.SIZE)], -1)
    feasible = (vhi >= vlo) & (x0_hi >= x0_lo) & (w <= 14) & (w >= 3)
    flip = rng.random(n) < 0.5
    sc.pos[flip, 0, 0] = W.SIZE - sc.pos[flip, 0, 0]
    sc.vel[flip, 0, 0] *= -1
    o = sc.occ[flip, 0].copy()
    sc.occ[flip, 0, 0], sc.occ[flip, 0, 1] = W.SIZE - o[:, 1], W.SIZE - o[:, 0]
    T = int(np.ceil(a.max())) + D + 8 + int(np.ceil(REVEAL_FULL_PX / vx.min()))
    return sc, feasible, T


def _hidden_run(vis, D):
    """Start frame of the first run of exactly D fully hidden frames after the ball was seen, else -1."""
    hidden = vis < 1e-3
    B, T = hidden.shape
    start = np.full(B, -1)
    for b in range(B):
        h = hidden[b]
        if h[0]:
            continue
        t = int(h.argmax()) if h.any() else -1
        if t <= 0:
            continue
        end = t
        while end < T and h[end]:
            end += 1
        if end - t == D and end + 6 <= T and not h[end : end + 6].any():
            start[b] = t
    return start


def _rebound_scene(rng, n, D, d_min=0.0):
    """One ball entering an occluder flush with a side wall, bouncing out of sight, hidden for about D frames."""
    sc = W.random_scene(rng, n, n_balls=1, n_occ=1)
    r = rng.uniform(1.6, 2.4, n).astype(np.float32)
    sc.rad[:, 0] = r
    w = rng.uniform(2 * r + 1.0, 14.0)
    vx = 2 * (w - 2 * r - 0.5) / max(D - 0.5, 0.5)
    feasible = (vx >= 0.15) & (vx <= 1.3)
    a = np.ceil(4 + (2 * r + 2) / np.maximum(vx, 0.05))
    x0 = W.SIZE - w
    vy = vx * np.tan(np.deg2rad(rng.uniform(-10, 10, n)))
    x_start = x0 + r + 0.25 * vx - a * vx
    feasible &= x_start - r >= 0.5
    y_start = rng.uniform(r + 3, W.SIZE - r - 3)
    sc.pos[:, 0] = np.stack([x_start, y_start], -1)
    sc.vel[:, 0] = np.stack([vx, vy], -1)
    sc.occ[:, 0] = np.stack([x0, np.full(n, W.SIZE), np.zeros(n), np.full(n, W.SIZE)], -1)
    flip = rng.random(n) < 0.5
    sc.pos[flip, 0, 0] = W.SIZE - sc.pos[flip, 0, 0]
    sc.vel[flip, 0, 0] *= -1
    o = sc.occ[flip, 0].copy()
    sc.occ[flip, 0, 0], sc.occ[flip, 0, 1] = W.SIZE - o[:, 1], W.SIZE - o[:, 0]
    live = vx[feasible] if feasible.any() else np.array([1.0])
    T = int(np.ceil(np.where(feasible, a, 0).max())) + D + 8 + int(np.ceil(REVEAL_FULL_PX / live.min()))
    return sc, feasible, T


def _reveal_frames(sc, px=REVEAL_PX):
    return np.ceil(px / np.abs(sc.vel[:, 0, 0])).astype(int)


def _occlusion_family(rng, B, D, family, edit_fn, d_min=0.0, scene_fn=None, bounces=0):
    scene_fn = scene_fn or _occluded_scene

    def propose(n):
        sc, feasible, T = scene_fn(rng, n, D, d_min)
        idx = np.nonzero(feasible)[0]
        sc = _subset(sc, idx)
        feasible = feasible[idx]
        n = len(idx)
        law = W.simulate(sc, T)
        a = _hidden_run(W.visible(law), D)
        ok = feasible & (a >= 0)
        after = _reveal_frames(sc)
        ok &= np.array([law.bounced[b, max(a[b] - 4, 0) : a[b] + D + after[b]].sum() == bounces if a[b] >= 0 else False for b in range(n)])
        return sc, ok, {"a": a, "T": np.full(n, T)}

    sc, ex = _collect(rng, B, propose)
    T = int(ex["T"].max())
    law = W.simulate(sc, T)
    a = _hidden_run(W.visible(law), D)
    assert (a >= 0).all()
    imp = W.simulate(sc, T, [W.Edit(a + D // 2, edit_fn)])
    return sc, law, imp, _reveal_frames(sc)


def vanish(rng, B, D):
    def fn(st, rows):
        st["alive"][rows, 0] = False

    sc, law, imp, length = _occlusion_family(rng, B, D, "vanish", fn)
    return _finish("vanish", D, D, sc, law, imp, rng.integers(1 << 31), length, _reveal_frames(sc, REVEAL_FULL_PX))


def hidden(rng, B, D, d):
    sign = rng.choice([-1.0, 1.0], B)

    def fn(st, rows):
        st["pos"][rows, 0] += (sign[rows] * d)[:, None] * _unit(st["vel"][rows, 0])

    sc, law, imp, length = _occlusion_family(rng, B, D, "hidden", fn, d_min=HIDDEN_D_MAX)
    return _finish("hidden", d, D, sc, law, imp, rng.integers(1 << 31), length + int(np.ceil(HIDDEN_D_MAX / 0.15)), sign=sign)


def rebound(rng, B, D):
    def fn(st, rows):
        st["alive"][rows, 0] = False

    sc, law, imp, length = _occlusion_family(rng, B, D, "rebound", fn, scene_fn=_rebound_scene, bounces=1)
    return _finish("rebound", D, D, sc, law, imp, rng.integers(1 << 31), length, _reveal_frames(sc, REVEAL_FULL_PX))


def exit_(rng, B, D, length=8):
    """Nothing is hidden here. If the empty frames still cost more than frames that never held a ball, that is not memory
    of a hidden object, and the occluded families' trace has to be read against it."""

    def change(walls, rows, side):
        walls[rows, 0, side] = np.where(side % 2 == 0, -1e4, 1e4)

    sc, law, imp, _ = _wall_family(rng, B, "exit", change, T=32 + D + length)
    p = _finish("exit", D, D, sc, law, imp, rng.integers(1 << 31), length)
    gone = np.array([np.nonzero(v[4:] < 1e-3)[0][0] + 4 for v in p.imp_vis[:, :, 0]])
    p.window = np.stack([gone + D, gone + D + length], -1)
    p.window_full = p.window
    return p


def _collision_scene(rng, n, t_hit=11):
    """Two balls on a course to meet about t_hit frames in, away from the walls and from each other until then."""
    sc = W.random_scene(rng, n, n_balls=2, n_occ=0)
    r = rng.uniform(1.8, 3.0, (n, 2)).astype(np.float32)
    sc.rad = r
    sc.alive[:] = True
    speed = rng.uniform(0.4, 1.0, (n, 2))
    ang = rng.uniform(0, 2 * np.pi, n)
    spread = rng.uniform(0.6, np.pi, n) * rng.choice([-1.0, 1.0], n)
    u0 = np.stack([np.cos(ang), np.sin(ang)], -1)
    u1 = np.stack([np.cos(ang + spread), np.sin(ang + spread)], -1)
    meet = rng.uniform(10, W.SIZE - 10, (n, 2))
    offset = rng.uniform(-0.6, 0.6, n)[:, None] * (r[:, 0] + r[:, 1])[:, None] * np.stack([-u0[:, 1], u0[:, 0]], -1)
    v0, v1 = speed[:, :1] * u0, speed[:, 1:] * u1
    sc.vel = np.stack([v0, v1], 1).astype(np.float32)
    sc.pos = np.stack([meet - t_hit * v0, meet + offset - t_hit * v1], 1).astype(np.float32)
    return sc


def collision(rng, B, m, T=24, length=8):
    """Restitution 1 - m: m = 0 lawful, m = 1 the balls stick along the contact normal, m = 2 they pass through, m < 0
    harder than elastic. The sign matters: under matched noise a directional bias shows as a constant proportion correct."""

    def ball_ok(traj, n):
        contact = traj.bounced[:, 5:T].any(-1)
        first = np.where(contact.any(1), contact.argmax(1) + 5, -1)
        return first, (contact.sum(1) >= 1)

    def propose(n):
        sc = _collision_scene(rng, n)
        inside = ((sc.pos - sc.rad[..., None]) > 0.5).all((1, 2)) & ((sc.pos + sc.rad[..., None]) < W.SIZE - 0.5).all((1, 2))
        law = W.simulate(sc, T)
        first, hit = ball_ok(law, n)
        both = law.bounced[:, 5:T].all(-1).sum(1) == 1  # exactly one frame where both balls changed velocity
        once = law.bounced[:, 5:T].any(-1).sum(1) == 1  # and nothing else: no wall in the clip
        sep = np.linalg.norm(law.pos[:, 0, 0] - law.pos[:, 0, 1], axis=-1) > sc.rad.sum(1) + 1
        return sc, inside & hit & both & once & sep & (first >= 8), {"first": first}

    sc, ex = _collect(rng, B, propose)
    law = W.simulate(sc, T)
    imp = W.simulate(sc, T, restitution=1.0 - m)
    return _finish("collision", m, 0, sc, law, imp, rng.integers(1 << 31), length)


FAMILIES = {
    "teleport": dict(fn=teleport, levels=[0.02, 0.04, 0.08, 0.15, 0.25, 0.5, 1.0, 2.0, 4.0], unit="px"),
    "deflect": dict(fn=deflect, levels=[0.12, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 180], unit="deg"),
    "phantom": dict(fn=phantom, levels=[0.01, 0.03, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0], unit="px"),
    "passthrough": dict(fn=passthrough, levels=[1.0], unit=""),
    "collision": dict(fn=collision, levels=[-0.5, -0.25, -0.1, -0.03, -0.01, -0.003, 0.003, 0.01, 0.03, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0], unit="restitution error"),
    "vanish": dict(fn=vanish, levels=[1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48], unit="frames"),
    "rebound": dict(fn=rebound, levels=[6, 8, 12, 16, 20, 24, 28, 32, 40, 48], unit="frames"),
    "exit": dict(fn=exit_, levels=[1, 4, 8, 16, 24, 32, 48], unit="frames"),
    "hidden": dict(fn=lambda rng, B, d: hidden(rng, B, 12, d), levels=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0], unit="px"),
}
