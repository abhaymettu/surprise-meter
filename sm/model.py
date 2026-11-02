"""Next-frame world models over pixel patches.

Each 32x32 frame becomes 16 tokens of 8x8 = 64 raw pixel values. Both models output a per-pixel Gaussian
over frame t+1 after seeing frame t, so they assign a proper log-likelihood, in nats, to every frame they
did not see yet.

  recurrent  a fixed set of state tokens is the only thing carried between frames, so anything remembered
             about a hidden ball is in the state; there is no clock and no access to old pixels
  window     tokens attend block-causally over the last W frames, a control whose memory is its window
"""

import math
from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn

from .world import NOISE, SIZE

PATCH = 8
GRID = SIZE // PATCH
TOKENS = GRID * GRID  # tokens per frame
PDIM = PATCH * PATCH
SIGMA_FLOOR = 0.9 * NOISE


@dataclass
class Config:
    d: int = 128
    layers: int = 5
    heads: int = 2
    arch: str = "recurrent"
    slots: int = 8  # recurrent: state tokens beyond the one per patch position
    window: int = 32  # window: attention span in frames
    rope_base: float = 500.0


SIZES = {
    "1m": Config(d=128, layers=5, heads=2),
    "5m": Config(d=256, layers=6, heads=4),
    "20m": Config(d=384, layers=11, heads=6),
}


def patchify(frames):
    """(B, T, H, W) -> (B, T * TOKENS, PDIM), tokens ordered by frame, then patch row, then patch column."""
    B, T = frames.shape[:2]
    x = frames.reshape(B, T, GRID, PATCH, GRID, PATCH).transpose(0, 1, 2, 4, 3, 5)
    return x.reshape(B, T * TOKENS, PDIM)


def unpatchify(x, T):
    B = x.shape[0]
    x = x.reshape(B, T, GRID, GRID, PATCH, PATCH).transpose(0, 1, 2, 4, 3, 5)
    return x.reshape(B, T, SIZE, SIZE)


def frame_mask(T, window, dtype=mx.float32):
    t = mx.arange(T * TOKENS) // TOKENS
    dt = t[:, None] - t[None, :]
    ok = (dt >= 0) & (dt < window)
    return mx.where(ok, mx.array(0.0, dtype), mx.array(-1e9, dtype))


def rope_tables(T, dh, base, offset=0):
    t = (mx.arange(T * TOKENS) // TOKENS + offset).astype(mx.float32)
    inv = base ** (-mx.arange(0, dh, 2).astype(mx.float32) / dh)
    ang = t[:, None] * inv[None, :]
    return mx.cos(ang), mx.sin(ang)


def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return mx.stack([y1, y2], -1).reshape(x.shape)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.n2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)

    def __call__(self, x, mask=None, cos=None, sin=None):
        B, L, d = x.shape
        h = self.heads
        q, k, v = mx.split(self.qkv(self.n1(x)), 3, axis=-1)
        q, k, v = (a.reshape(B, L, h, d // h).transpose(0, 2, 1, 3) for a in (q, k, v))
        if cos is not None:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / math.sqrt(d // h), mask=mask)
        x = x + self.proj(a.transpose(0, 2, 1, 3).reshape(B, L, d))
        return x + self.fc2(nn.gelu(self.fc1(self.n2(x))))


def _gaussian(out):
    mean = out[..., :PDIM] + 0.25
    sigma = SIGMA_FLOOR + nn.softplus(out[..., PDIM:] - 1.0) * 0.5
    return mean, sigma


def _head(d):
    head = nn.Linear(d, 2 * PDIM)
    head.weight = mx.zeros_like(head.weight)
    head.bias = mx.zeros_like(head.bias)
    return head


class WindowModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.embed = nn.Linear(PDIM, d)
        self.where = mx.random.normal((TOKENS, d)) * 0.02
        self.blocks = [Block(d, cfg.heads) for _ in range(cfg.layers)]
        self.norm = nn.LayerNorm(d)
        self.head = _head(d)

    def __call__(self, frames):
        """frames (B, T, H, W) -> mean, sigma (B, T, H, W); index t predicts frame t+1."""
        B, T = frames.shape[:2]
        x = self.embed(patchify(frames) - 0.25) + mx.tile(self.where, (T, 1))[None]
        mask = frame_mask(T, self.cfg.window, x.dtype)
        cos, sin = rope_tables(T, self.cfg.d // self.cfg.heads, self.cfg.rope_base)
        for blk in self.blocks:
            x = blk(x, mask, cos, sin)
        mean, sigma = _gaussian(self.head(self.norm(x)))
        return unpatchify(mean, T), unpatchify(sigma, T)


class RecurrentModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.n_state = TOKENS + cfg.slots
        self.embed = nn.Linear(PDIM, d)
        self.where = mx.random.normal((TOKENS, d)) * 0.02
        self.start = mx.random.normal((self.n_state, d)) * 0.02
        self.blocks = [Block(d, cfg.heads) for _ in range(cfg.layers)]
        self.norm = nn.LayerNorm(d)
        self.head = _head(d)

    def initial_state(self, B):
        return mx.broadcast_to(self.start[None], (B, self.n_state, self.cfg.d))

    def step(self, state, patches):
        """state (B, S, d), patches (B, TOKENS, PDIM) -> new state and the Gaussian over the next frame's patches."""
        seen = self.embed(patches - 0.25) + self.where
        x = mx.concatenate([state[:, :TOKENS] + seen, state[:, TOKENS:]], 1)
        for blk in self.blocks:
            x = blk(x)
        state = self.norm(x)
        mean, sigma = _gaussian(self.head(state[:, :TOKENS]))
        return state, mean, sigma

    def __call__(self, frames, state=None):
        """frames (B, T, H, W) -> mean, sigma (B, T, H, W); index t predicts frame t+1."""
        B, T = frames.shape[:2]
        p = patchify(frames)
        state = self.initial_state(B) if state is None else state
        means, sigmas = [], []
        for t in range(T):
            state, mean, sigma = self.step(state, p[:, t * TOKENS : (t + 1) * TOKENS])
            means.append(mean)
            sigmas.append(sigma)
        return unpatchify(mx.concatenate(means, 1), T), unpatchify(mx.concatenate(sigmas, 1), T)


def build(cfg: Config):
    return RecurrentModel(cfg) if cfg.arch == "recurrent" else WindowModel(cfg)


def frame_nll(model, frames):
    """Negative log-likelihood in nats of each frame given the frames before it: (B, T-1) for frames 1..T-1."""
    mean, sigma = model(frames[:, :-1])
    x = frames[:, 1:]
    z = (x - mean) / sigma
    per_pixel = 0.5 * z * z + mx.log(sigma) + 0.5 * math.log(2 * math.pi)
    return per_pixel.sum((-1, -2))


def noise_floor_nll():
    """Expected per-frame NLL of a perfect predictor: the entropy of the pixel noise, in nats."""
    return SIZE * SIZE * (0.5 * math.log(2 * math.pi * math.e) + math.log(NOISE))


def count_params(model):
    from mlx.utils import tree_flatten

    return sum(v.size for _, v in tree_flatten(model.parameters()))


def config_dict(cfg):
    return asdict(cfg)
