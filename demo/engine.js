// The world model's forward step, by hand: one WebGPU compute implementation and one CPU implementation of
// sm/model.py RecurrentModel.step, with no framework. Both take a 32x32 frame and return the Gaussian over the next one.
const PATCH = 8, GRID = 4, TOKENS = 16, PDIM = 64, NOISE = 0.15, SIGMA_FLOOR = 0.9 * NOISE;

export async function loadWeights(url) {
  const [meta, buf] = await Promise.all([fetch(`${url}.json`).then((r) => r.json()), fetch(`${url}.bin`).then((r) => r.arrayBuffer())]);
  const half = new Uint16Array(buf), tensors = {};
  for (const [name, t] of Object.entries(meta.tensors)) tensors[name] = { shape: t.shape, data: halfToFloat(half, t.offset, t.count) };
  return { ...meta, tensors };
}

function halfToFloat(src, offset, count) {
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const h = src[offset + i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
    out[i] = s * (e === 0 ? m * 2 ** -24 : e === 31 ? (m ? NaN : Infinity) : (1 + m / 1024) * 2 ** (e - 15));
  }
  return out;
}

function patches(frame) {
  const p = new Float32Array(TOKENS * PDIM);
  for (let g = 0; g < TOKENS; g++) {
    const gy = g >> 2, gx = g & 3;
    for (let i = 0; i < PATCH; i++) for (let j = 0; j < PATCH; j++) p[g * PDIM + i * PATCH + j] = frame[(gy * PATCH + i) * 32 + gx * PATCH + j] - 0.25;
  }
  return p;
}

function decode(out, xrayOut) {
  const mean = new Float32Array(1024), sigma = new Float32Array(1024), xray = new Float32Array(1024);
  for (let g = 0; g < TOKENS; g++) {
    const gy = g >> 2, gx = g & 3;
    for (let k = 0; k < PDIM; k++) {
      const y = gy * PATCH + (k >> 3), x = gx * PATCH + (k & 7), at = y * 32 + x;
      mean[at] = out[g * 128 + k] + 0.25;
      const raw = out[g * 128 + PDIM + k] - 1;
      sigma[at] = SIGMA_FLOOR + 0.5 * Math.log1p(Math.exp(raw));
      if (xrayOut) xray[at] = xrayOut[g * PDIM + k] + 0.25;
    }
  }
  return { mean, sigma, xray: xrayOut ? xray : null };
}

// ---- CPU ----------------------------------------------------------------------------------------------------------

function erf(x) {
  const t = 1 / (1 + 0.5 * Math.abs(x));
  const r = 1 - t * Math.exp(-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : -r;
}

function linear(x, n, W, bias, nOut, nIn) {
  const y = new Float32Array(n * nOut);
  for (let t = 0; t < n; t++) {
    for (let o = 0; o < nOut; o++) {
      let s = bias ? bias[o] : 0;
      const w = o * nIn, xo = t * nIn;
      for (let i = 0; i < nIn; i++) s += W[w + i] * x[xo + i];
      y[t * nOut + o] = s;
    }
  }
  return y;
}

function layerNorm(x, n, d, g, b) {
  const y = new Float32Array(n * d);
  for (let t = 0; t < n; t++) {
    let m = 0;
    for (let i = 0; i < d; i++) m += x[t * d + i];
    m /= d;
    let v = 0;
    for (let i = 0; i < d; i++) v += (x[t * d + i] - m) ** 2;
    const r = 1 / Math.sqrt(v / d + 1e-5);
    for (let i = 0; i < d; i++) y[t * d + i] = (x[t * d + i] - m) * r * g[i] + b[i];
  }
  return y;
}

function attention(qkv, n, d, heads) {
  const dh = d / heads, y = new Float32Array(n * d), scale = 1 / Math.sqrt(dh), s = new Float32Array(n);
  for (let h = 0; h < heads; h++) {
    const qo = h * dh, ko = d + h * dh, vo = 2 * d + h * dh;
    for (let i = 0; i < n; i++) {
      let top = -Infinity;
      for (let j = 0; j < n; j++) {
        let dot = 0;
        for (let c = 0; c < dh; c++) dot += qkv[i * 3 * d + qo + c] * qkv[j * 3 * d + ko + c];
        s[j] = dot * scale;
        if (s[j] > top) top = s[j];
      }
      let z = 0;
      for (let j = 0; j < n; j++) z += s[j] = Math.exp(s[j] - top);
      for (let j = 0; j < n; j++) {
        const p = s[j] / z;
        for (let c = 0; c < dh; c++) y[i * d + qo + c] += p * qkv[j * 3 * d + vo + c];
      }
    }
  }
  return y;
}

export class CpuEngine {
  constructor(w) {
    this.w = w; this.d = w.d; this.n = TOKENS + w.slots; this.kind = "cpu";
    this.reset();
  }
  reset() { this.state = Float32Array.from(this.w.tensors.start.data); }
  async step(frame) {
    const { d, n, w } = this, T = w.tensors;
    const seen = linear(patches(frame), TOKENS, T["embed.weight"].data, T["embed.bias"].data, d, PDIM);
    let x = Float32Array.from(this.state);
    for (let i = 0; i < TOKENS * d; i++) x[i] += seen[i] + T.where.data[i];
    for (let l = 0; l < w.layers; l++) {
      const B = (s) => T[`blocks.${l}.${s}`].data;
      const qkv = linear(layerNorm(x, n, d, B("n1.weight"), B("n1.bias")), n, B("qkv.weight"), null, 3 * d, d);
      const p = linear(attention(qkv, n, d, w.heads), n, B("proj.weight"), null, d, d);
      for (let i = 0; i < x.length; i++) x[i] += p[i];
      const h = linear(layerNorm(x, n, d, B("n2.weight"), B("n2.bias")), n, B("fc1.weight"), B("fc1.bias"), 4 * d, d);
      for (let i = 0; i < h.length; i++) h[i] = 0.5 * h[i] * (1 + erf(h[i] * Math.SQRT1_2));
      const q = linear(h, n, B("fc2.weight"), B("fc2.bias"), d, 4 * d);
      for (let i = 0; i < x.length; i++) x[i] += q[i];
    }
    this.state = layerNorm(x, n, d, T["norm.weight"].data, T["norm.bias"].data);
    const spatial = this.state.subarray(0, TOKENS * d);
    const out = linear(spatial, TOKENS, T["head.weight"].data, T["head.bias"].data, 128, d);
    const xr = T["xray.weight"] ? linear(spatial, TOKENS, T["xray.weight"].data, T["xray.bias"].data, PDIM, d) : null;
    return decode(out, xr);
  }
}

// ---- WebGPU -------------------------------------------------------------------------------------------------------

const WGSL = /* wgsl */ `
struct P { n: u32, a: u32, b: u32, c: u32, w: u32, bias: u32, heads: u32, flag: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

// y[t, o] = sum_i W[o, i] x[t, i] + bias[o]. a = input width, b = output width, flag 1 adds bias.
@compute @workgroup_size(1) fn linear(@builtin(global_invocation_id) id: vec3u) {
  let t = id.x;
  for (var o = 0u; o < p.b; o++) {
    var s = 0.0;
    if (p.flag == 1u) { s = W[p.bias + o]; }
    let wo = p.w + o * p.a; let xo = t * p.a;
    for (var i = 0u; i < p.a; i++) { s += W[wo + i] * x[xo + i]; }
    y[t * p.b + o] = s;
  }
}

@compute @workgroup_size(1) fn layernorm(@builtin(global_invocation_id) id: vec3u) {
  let t = id.x; let d = p.a; var m = 0.0;
  for (var i = 0u; i < d; i++) { m += x[t * d + i]; }
  m /= f32(d);
  var v = 0.0;
  for (var i = 0u; i < d; i++) { let e = x[t * d + i] - m; v += e * e; }
  let r = 1.0 / sqrt(v / f32(d) + 1e-5);
  for (var i = 0u; i < d; i++) { y[t * d + i] = (x[t * d + i] - m) * r * W[p.w + i] + W[p.bias + i]; }
}

@compute @workgroup_size(1) fn attention(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x; let d = p.a; let n = p.n; let dh = d / p.heads; let scale = 1.0 / sqrt(f32(dh));
  for (var c = 0u; c < d; c++) { y[i * d + c] = 0.0; }
  for (var h = 0u; h < p.heads; h++) {
    let qo = h * dh; let ko = d + h * dh; let vo = 2u * d + h * dh;
    var top = -1e30;
    for (var j = 0u; j < n; j++) {
      var dot = 0.0;
      for (var c = 0u; c < dh; c++) { dot += x[i * 3u * d + qo + c] * x[j * 3u * d + ko + c]; }
      top = max(top, dot * scale);
    }
    var z = 0.0;
    for (var j = 0u; j < n; j++) {
      var dot = 0.0;
      for (var c = 0u; c < dh; c++) { dot += x[i * 3u * d + qo + c] * x[j * 3u * d + ko + c]; }
      z += exp(dot * scale - top);
    }
    for (var j = 0u; j < n; j++) {
      var dot = 0.0;
      for (var c = 0u; c < dh; c++) { dot += x[i * 3u * d + qo + c] * x[j * 3u * d + ko + c]; }
      let pr = exp(dot * scale - top) / z;
      for (var c = 0u; c < dh; c++) { y[i * d + qo + c] += pr * x[j * 3u * d + vo + c]; }
    }
  }
}

fn erf(v: f32) -> f32 {
  let t = 1.0 / (1.0 + 0.5 * abs(v));
  let r = 1.0 - t * exp(-v * v - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return select(-r, r, v >= 0.0);
}

@compute @workgroup_size(32) fn gelu(@builtin(global_invocation_id) id: vec3u) {
  if (id.x < p.n) { let v = x[id.x]; y[id.x] = 0.5 * v * (1.0 + erf(v * 0.70710678)); }
}

// y[i] += x[i] over the first n values; with flag 1, y[i] = x[i] + W[w + i].
@compute @workgroup_size(32) fn add(@builtin(global_invocation_id) id: vec3u) {
  if (id.x < p.n) {
    if (p.flag == 1u) { y[id.x] = x[id.x] + W[p.w + id.x]; } else { y[id.x] += x[id.x]; }
  }
}
`;

export class GpuEngine {
  static async create(w) {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) return null;
    const device = await adapter.requestDevice({
      requiredLimits: { maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 1 << 30), maxStorageBuffersPerShaderStage: 4 },
    });
    const e = new GpuEngine();
    e.device = device; e.w = w; e.kind = "webgpu";
    e.build();
    e.reset();
    return e;
  }

  build() {
    const { device, w } = this, d = w.d, n = TOKENS + w.slots;
    this.n = n; this.d = d;
    let total = 0; const at = {};
    for (const [k, t] of Object.entries(w.tensors)) { at[k] = total; total += t.data.length; }
    const weights = device.createBuffer({ size: total * 4, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
    const mapped = new Float32Array(weights.getMappedRange());
    for (const [k, t] of Object.entries(w.tensors)) mapped.set(t.data, at[k]);
    weights.unmap();

    const buf = (words) => device.createBuffer({ size: words * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const b = {
      input: buf(TOKENS * PDIM), seen: buf(n * d), X: buf(n * d), N: buf(n * d), Q: buf(n * 3 * d), A: buf(n * d), P: buf(n * d), H: buf(n * 4 * d), H2: buf(n * 4 * d),
      state: buf(n * d), out: buf(TOKENS * 128), xray: buf(TOKENS * PDIM),
    };
    this.b = b;
    const mod = device.createShaderModule({ code: WGSL });
    const layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const pipe = {};
    for (const entryPoint of ["linear", "layernorm", "attention", "gelu", "add"]) pipe[entryPoint] = device.createComputePipeline({ layout: pl, compute: { module: mod, entryPoint } });

    const ops = (this.ops = []);
    const op = (entry, src, dst, count, u) => {
      const data = new Uint32Array([u.n ?? n, u.a ?? 0, u.b ?? 0, 0, u.w ?? 0, u.bias ?? 0, w.heads, u.flag ?? 0]);
      const ub = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true });
      new Uint32Array(ub.getMappedRange()).set(data); ub.unmap();
      const bg = device.createBindGroup({
        layout,
        entries: [{ binding: 0, resource: { buffer: ub } }, { binding: 1, resource: { buffer: weights } }, { binding: 2, resource: { buffer: src } }, { binding: 3, resource: { buffer: dst } }],
      });
      ops.push({ pipeline: pipe[entry], bg, count });
    };
    const lin = (src, dst, tokens, nameW, nameB, nIn, nOut) => op("linear", src, dst, tokens, { a: nIn, b: nOut, w: at[nameW], bias: nameB ? at[nameB] : 0, flag: nameB ? 1 : 0 });
    const norm = (src, dst, g, bb) => op("layernorm", src, dst, n, { a: d, w: at[g], bias: at[bb] });
    const wg32 = (words) => Math.ceil(words / 32);

    lin(b.input, b.seen, TOKENS, "embed.weight", "embed.bias", PDIM, d);
    op("add", b.seen, b.P, wg32(TOKENS * d), { n: TOKENS * d, flag: 1, w: at.where }); // P = seen + where
    op("add", b.P, b.X, wg32(TOKENS * d), { n: TOKENS * d }); // X, a copy of the state, += P
    for (let l = 0; l < w.layers; l++) {
      const T = (s) => `blocks.${l}.${s}`;
      norm(b.X, b.N, T("n1.weight"), T("n1.bias"));
      lin(b.N, b.Q, n, T("qkv.weight"), null, d, 3 * d);
      op("attention", b.Q, b.A, n, { a: d });
      lin(b.A, b.P, n, T("proj.weight"), null, d, d);
      op("add", b.P, b.X, wg32(n * d), { n: n * d });
      norm(b.X, b.N, T("n2.weight"), T("n2.bias"));
      lin(b.N, b.H, n, T("fc1.weight"), T("fc1.bias"), d, 4 * d);
      op("gelu", b.H, b.H2, wg32(n * 4 * d), { n: n * 4 * d });
      lin(b.H2, b.P, n, T("fc2.weight"), T("fc2.bias"), 4 * d, d);
      op("add", b.P, b.X, wg32(n * d), { n: n * d });
    }
    norm(b.X, b.state, "norm.weight", "norm.bias");
    lin(b.state, b.out, TOKENS, "head.weight", "head.bias", d, 128);
    if (w.tensors["xray.weight"]) lin(b.state, b.xray, TOKENS, "xray.weight", "xray.bias", d, PDIM);
    this.readOut = device.createBuffer({ size: TOKENS * 128 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this.readX = device.createBuffer({ size: TOKENS * PDIM * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  }

  reset() {
    this.device.queue.writeBuffer(this.b.state, 0, this.w.tensors.start.data);
  }

  async step(frame) {
    const { device, b } = this;
    device.queue.writeBuffer(b.input, 0, patches(frame));
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(b.state, 0, b.X, 0, b.state.size);
    const pass = enc.beginComputePass();
    this.ops.forEach((o) => {
      pass.setPipeline(o.pipeline); pass.setBindGroup(0, o.bg); pass.dispatchWorkgroups(o.count);
    });
    pass.end();
    enc.copyBufferToBuffer(b.out, 0, this.readOut, 0, this.readOut.size);
    const hasX = !!this.w.tensors["xray.weight"];
    if (hasX) enc.copyBufferToBuffer(b.xray, 0, this.readX, 0, this.readX.size);
    device.queue.submit([enc.finish()]);
    await this.readOut.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this.readOut.getMappedRange().slice(0));
    this.readOut.unmap();
    let xr = null;
    if (hasX) {
      await this.readX.mapAsync(GPUMapMode.READ);
      xr = new Float32Array(this.readX.getMappedRange().slice(0));
      this.readX.unmap();
    }
    return decode(out, xr);
  }
}
