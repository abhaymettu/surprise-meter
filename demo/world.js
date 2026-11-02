// The same 2D world the model was trained on (sm/world.py), one scene at a time, in JS.
export const SIZE = 32, NOISE = 0.04;

const uniform = (a, b) => a + Math.random() * (b - a);
const gauss = () => Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());

export function randomScene({ balls = Math.random() < 0.5 ? 1 : 2, occluders = pick([0, 1, 1, 2]) } = {}) {
  const list = [];
  for (let i = 0; i < balls; i++) {
    const r = uniform(1.5, 3), speed = Math.exp(uniform(Math.log(0.25), Math.log(1.0))), a = uniform(0, 2 * Math.PI);
    let x, y, tries = 0;
    do {
      x = uniform(r, SIZE - r); y = uniform(r, SIZE - r);
    } while (list.some((b) => Math.hypot(b.x - x, b.y - y) < b.r + r + 1) && ++tries < 40);
    list.push({ x, y, vx: speed * Math.cos(a), vy: speed * Math.sin(a), r, lum: uniform(0.7, 1), alive: true });
  }
  const occ = [];
  for (let i = 0; i < occluders; i++) {
    const w = uniform(5, 12), x0 = uniform(0, SIZE - w);
    occ.push({ x0, x1: x0 + w, y0: 0, y1: SIZE, lum: uniform(0.35, 0.55) });
  }
  return { balls: list, occ, wallsOn: true };
}

function pick(a) { return a[Math.floor(Math.random() * a.length)]; }

export function step(world) {
  const { balls } = world;
  for (const b of balls) {
    if (!b.alive || b.held) continue;
    b.x += b.vx; b.y += b.vy;
    if (!world.wallsOn) continue;
    if (b.x < b.r) { b.x = 2 * b.r - b.x; b.vx = -b.vx; }
    if (b.x > SIZE - b.r) { b.x = 2 * (SIZE - b.r) - b.x; b.vx = -b.vx; }
    if (b.y < b.r) { b.y = 2 * b.r - b.y; b.vy = -b.vy; }
    if (b.y > SIZE - b.r) { b.y = 2 * (SIZE - b.r) - b.y; b.vy = -b.vy; }
  }
  if (balls.length === 2 && balls[0].alive && balls[1].alive && !balls[0].held && !balls[1].held && world.collide !== false) {
    const [a, b] = balls;
    const dx = b.x - a.x, dy = b.y - a.y, dist = Math.max(Math.hypot(dx, dy), 1e-6);
    const nx = dx / dist, ny = dy / dist;
    const rel = (b.vx - a.vx) * nx + (b.vy - a.vy) * ny;
    if (dist < a.r + b.r && rel < 0) {
      const m0 = a.r * a.r, m1 = b.r * b.r, j = (2 * rel) / (m0 + m1);
      a.vx += j * m1 * nx; a.vy += j * m1 * ny;
      b.vx -= j * m0 * nx; b.vy -= j * m0 * ny;
      const o = (a.r + b.r - dist) * 0.5;
      a.x -= o * nx; a.y -= o * ny; b.x += o * nx; b.y += o * ny;
    }
  }
}

function coverage(world, px, py) {
  let keep = 1, sum = 0, lum = 0;
  for (const o of world.occ) {
    const cx = Math.max(0, Math.min(1, Math.min(px + 1, o.x1) - Math.max(px, o.x0)));
    const cy = Math.max(0, Math.min(1, Math.min(py + 1, o.y1) - Math.max(py, o.y0)));
    const c = cx * cy;
    keep *= 1 - c; sum += c; lum += c * o.lum;
  }
  return { cover: 1 - keep, lum: sum > 0 ? lum / sum : 0 };
}

export function render(world, out, noise = NOISE) {
  for (let py = 0; py < SIZE; py++) {
    for (let px = 0; px < SIZE; px++) {
      let v = 0;
      for (const b of world.balls) {
        if (!b.alive) continue;
        const dx = px + 0.5 - b.x, dy = py + 0.5 - b.y;
        const d = b.r + 0.5 - Math.sqrt(dx * dx + dy * dy);
        if (d > 0) v = Math.max(v, Math.min(d, 1) * b.lum);
      }
      const { cover, lum } = coverage(world, px, py);
      out[py * SIZE + px] = v * (1 - cover) + lum * cover + (noise > 0 ? gauss() * noise : 0);
    }
  }
  return out;
}
