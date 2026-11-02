// The browser's CPU engine against MLX on the clip that sm/export.py saved. Run from the repo root.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
const root = resolve("demo") + "/";
const name = process.argv[2] ?? "model";
global.fetch = async (u) => ({
  json: async () => JSON.parse(readFileSync(root + u)),
  arrayBuffer: async () => { const b = readFileSync(root + u); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); },
});
const { loadWeights, CpuEngine } = await import(root + "engine.js");
const engine = new CpuEngine(await loadWeights(name));
const clip = JSON.parse(readFileSync(root + name + ".parity.json"));
let worst = 0;
const t0 = performance.now();
for (let t = 0; t < clip.frames.length; t++) {
  const res = await engine.step(clip.frames[t].flat());
  const m = clip.mean[t].flat(), s = clip.sigma[t].flat();
  for (let i = 0; i < 1024; i++) worst = Math.max(worst, Math.abs(res.mean[i] - m[i]), Math.abs(res.sigma[i] - s[i]));
}
console.log(`${name}: worst difference from MLX ${worst.toExponential(2)}, ${((performance.now() - t0) / clip.frames.length).toFixed(0)} ms per frame on the CPU`);
process.exit(worst < 1e-3 ? 0 : 1);
