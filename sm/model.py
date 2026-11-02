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
    mix: int = 1  # mixture components per patch
    state_norm: bool = True  # recurrent: LayerNorm the state between frames
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


def _head(d, mix):
    head = nn.Linear(d, mix * (2 * PDIM + 1))
    head.bias = mx.zeros_like(head.bias)
    head.weight = mx.zeros_like(head.weight) if mix == 1 else mx.random.normal(head.weight.shape) * 0.01
    return head


def _mixture(out, mix):
    """out (..., mix * (2 * PDIM + 1)) -> mean and sigma (..., mix, PDIM), logit (..., mix)."""
    logit = out[..., :mix]
    rest = out[..., mix:].reshape(*out.shape[:-1], mix, 2 * PDIM)
    mean = rest[..., :PDIM] + 0.25
    sigma = SIGMA_FLOOR + nn.softplus(rest[..., PDIM:] - 1.0) * 0.5
    return mean, sigma, logit


def patch_nll(mean, sigma, logit, target):
    """target (..., TOKENS, PDIM) under the per-patch mixture -> nll and model entropy, nats, (..., TOKENS)."""
    c = 0.5 * math.log(2 * math.pi)
    log_sigma = mx.log(sigma)
    z = (target[..., None, :] - mean) / sigma
    comp = -(0.5 * z * z + log_sigma + c).sum(-1)
    logp = logit - mx.logsumexp(logit, axis=-1, keepdims=True)
    nll = -mx.logsumexp(logp + comp, axis=-1)
    entropy = (mx.exp(logp) * (log_sigma + 0.5 + c).sum(-1)).sum(-1)
    return nll, entropy


class WindowModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.embed = nn.Linear(PDIM, d)
        self.where = mx.random.normal((TOKENS, d)) * 0.02
        self.blocks = [Block(d, cfg.heads) for _ in range(cfg.layers)]
        self.norm = nn.LayerNorm(d)
        self.head = _head(d, cfg.mix)

    def __call__(self, frames):
        """frames (B, T, H, W) -> mean, sigma, logit over (B, T, TOKENS, ...); index t predicts frame t+1."""
        B, T = frames.shape[:2]
        x = self.embed(patchify(frames) - 0.25) + mx.tile(self.where, (T, 1))[None]
        mask = frame_mask(T, self.cfg.window, x.dtype)
        cos, sin = rope_tables(T, self.cfg.d // self.cfg.heads, self.cfg.rope_base)
        for blk in self.blocks:
            x = blk(x, mask, cos, sin)
        out = self.head(self.norm(x)).reshape(B, T, TOKENS, -1)
        return _mixture(out, self.cfg.mix)


class RecurrentModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.n_state = TOKENS + cfg.slots
        self.embed = nn.Linear(PDIM, d)
        self.where = mx.random.normal((TOKENS, d)) * 0.02
        self.start = mx.random.normal((self.n_state, d)) * 0.02
        self.blank = mx.random.normal((1, d)) * 0.02
        self.blocks = [Block(d, cfg.heads) for _ in range(cfg.layers)]
        self.norm = nn.LayerNorm(d)
        self.head = _head(d, cfg.mix)

    def initial_state(self, B):
        return mx.broadcast_to(self.start[None], (B, self.n_state, self.cfg.d))

    def step(self, state, patches, blank=None):
        """state (B, S, d), patches (B, TOKENS, PDIM) -> new state and the mixture over the next frame's patches.

        blank (B, TOKENS), 1 where the patch is withheld: the model gets a learned token instead of those pixels.
        """
        seen = self.embed(patches - 0.25)
        if blank is not None:
            seen = mx.where(blank[..., None] > 0, self.blank, seen)
        seen = seen + self.where
        x = mx.concatenate([state[:, :TOKENS] + seen, state[:, TOKENS:]], 1)
        for blk in self.blocks:
            x = blk(x)
        read = self.norm(x)
        mean, sigma, logit = _mixture(self.head(read[:, :TOKENS]), self.cfg.mix)
        return (read if self.cfg.state_norm else x), mean, sigma, logit

    def __call__(self, frames, blank=None):
        """frames (B, T, H, W), blank (B, T, TOKENS) -> mean, sigma, logit over (B, T, TOKENS, ...); index t predicts t+1."""
        B, T = frames.shape[:2]
        p = patchify(frames)
        state = self.initial_state(B)
        out = []
        for t in range(T):
            state, mean, sigma, logit = self.step(state, p[:, t * TOKENS : (t + 1) * TOKENS], None if blank is None else blank[:, t])
            out.append((mean, sigma, logit))
        return tuple(mx.stack(a, 1) for a in zip(*out))


def build(cfg: Config):
    return RecurrentModel(cfg) if cfg.arch == "recurrent" else WindowModel(cfg)


def frame_nll(model, frames, blank=None):
    """Negative log-likelihood in nats of each frame given the frames before it: (B, T-1) for frames 1..T-1."""
    mean, sigma, logit = model(frames[:, :-1], blank) if blank is not None else model(frames[:, :-1])
    B, T = frames.shape[:2]
    target = patchify(frames[:, 1:]).reshape(B, T - 1, TOKENS, PDIM)
    return patch_nll(mean, sigma, logit, target)[0].sum(-1)


def noise_floor_nll():
    """Expected per-frame NLL of a perfect predictor: the entropy of the pixel noise, in nats."""
    return SIZE * SIZE * (0.5 * math.log(2 * math.pi * math.e) + math.log(NOISE))


def count_params(model):
    from mlx.utils import tree_flatten

    return sum(v.size for _, v in tree_flatten(model.parameters()))


def config_dict(cfg):
    return asdict(cfg)
