import { SIZE, NOISE, randomScene, step as stepWorld, render as renderWorld } from "./world.js";
import { loadWeights, GpuEngine, WorkerEngine } from "./engine.js";

const $ = (id) => document.getElementById(id);
const TICK_MS = 50, BURN_IN = 4, REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;

// ---- surprise, as defined in sm/surprise.py ------------------------------------------------------------------------

let nullTable = null;

function evidence(frame, mean, sigma) {
  const excess = new Float32Array(1024), patch = new Float32Array(16);
  for (let i = 0; i < 1024; i++) {
    const z = (frame[i] - mean[i]) / sigma[i];
    excess[i] = 0.5 * (z * z - 1);
    patch[((i >> 8) & 3) * 4 + ((i >> 3) & 3)] += excess[i];
  }
  let best = -Infinity, bx = 16, by = 16;
  for (let y = 0; y < 3; y++) {
    for (let x = 0; x < 3; x++) {
      const s = patch[y * 4 + x] + patch[y * 4 + x + 1] + patch[(y + 1) * 4 + x] + patch[(y + 1) * 4 + x + 1];
      if (s > best) { best = s; bx = (x + 1) * 8; by = (y + 1) * 8; }
    }
  }
  return { best, bx, by, excess };
}

function oneIn(m) {
  const { evidence: e, one_in: o } = nullTable, n = e.length;
  if (m <= e[n - 1]) return o[n - 1];
  if (m >= e[0]) return o[0]; // stranger than every frame in the table, and the table is all that was measured
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (e[mid] >= m) lo = mid; else hi = mid;
  }
  const f = (e[lo] - m) / Math.max(e[lo] - e[hi], 1e-6);
  return Math.exp(Math.log(o[lo]) + f * (Math.log(o[hi]) - Math.log(o[lo])));
}

let LOG_TOP = 6, TOP = 1e6; // the scale ends where the calibration table does
const level = (n) => Math.min(Math.log10(Math.max(n, 1)) / LOG_TOP, 1);
const SPIKE = 3000; // about one lawful frame in three minutes gets this far

function twoFigures(n) {
  const d = 10 ** Math.max(0, Math.floor(Math.log10(n)) - 1);
  return (Math.floor(n / d) * d).toLocaleString("en-US");
}

function fmtOneIn(n) {
  return n >= TOP ? `1 in ${twoFigures(TOP)}+` : `1 in ${twoFigures(Math.max(n, 1))}`;
}

function meaning(n) {
  if (n >= TOP) return `Stranger than all ${twoFigures(TOP)} lawful frames this was calibrated on.`;
  if (n >= 1e4) return "Almost never happens when the physics is real.";
  if (n >= 300) return "Unusual. Lawful physics gets here only now and then.";
  return "Nothing unusual. One in this many lawful frames looks at least this strange.";
}

// ---- world loop ----------------------------------------------------------------------------------------------------

let engine = null, world = randomScene({ balls: 2, occluders: 1 }), prev = null, curr = null;
let pending = null, last = null, age = 0, needsReset = false;
let frameIdx = 0;
const frame = new Float32Array(1024);
const trace = new Float32Array(240).fill(0); // level per tick, newest at traceHead
const marks = [];
let traceHead = 0, peakLevel = 0, shownN = 1, shownLevel = 0, surpriseSpot = { x: 16, y: 16, a: 0 };

const snapshot = () => world.balls.map((b) => ({ x: b.x, y: b.y, r: b.r, lum: b.lum, alive: b.alive, a: b.alive ? 1 : 0, id: b.id }));

let nextId = 0;
const newWorld = (opts) => {
  world = randomScene(opts);
  world.balls.forEach((b) => (b.id = nextId++));
  needsReset = true;
  prev = curr = null;
  wantVanish = false;
  uiState();
};

let wantVanish = false, frozen = false;

function hiddenFully(b) {
  return world.occ.some((o) => b.x - b.r - 1 >= o.x0 && b.x + b.r + 1 <= o.x1);
}

async function tick() {
  if (needsReset) { engine?.reset(); age = 0; last = null; needsReset = false; }
  if (!frozen) stepWorld(world);
  if (wantVanish) {
    const b = world.balls.find((b) => b.alive && hiddenFully(b));
    if (b) { b.alive = false; wantVanish = false; b.goneAt = performance.now(); mark("deleted while hidden"); uiState(); caption("The ball is gone, but the model was not told. Its picture of the ball keeps moving.", 0); }
  }
  respawnIfEmpty();
  renderWorld(world, frame);
  prev = curr ?? snapshot();
  curr = snapshot();
  lastTickAt = performance.now();

  if (engine) {
    let n = 1;
    if (last && age >= BURN_IN) {
      const e = evidence(frame, last.mean, last.sigma);
      n = oneIn(e.best);
      if (n >= SPIKE) surpriseSpot = { x: e.bx, y: e.by, a: Math.max(surpriseSpot.a, level(n)) };
      drawViews(frame, last.mean, e.excess);
    }
    pushTrace(n);
    const res = await engine.step(frame);
    last = res;
    uploadXray(res.xray);
    age++;
  }
  frameIdx++;
}

let emptySince = 0;
function respawnIfEmpty() {
  if (attract || world.balls.some((b) => b.alive)) { emptySince = 0; return; }
  if (!emptySince) emptySince = performance.now();
  if (performance.now() - emptySince < 2600) return;
  const r = 2 + Math.random(), a = Math.random() * 6.28, s = 0.3 + Math.random() * 0.5;
  world.balls = [{ x: 16 + (Math.random() - 0.5) * 16, y: 16 + (Math.random() - 0.5) * 16, vx: s * Math.cos(a), vy: s * Math.sin(a), r, lum: 0.9, alive: true, id: nextId++ }];
  mark("a ball from nowhere");
  caption("A new ball, out of nothing. That is what the spike is.");
  emptySince = 0;
}

function pushTrace(n) {
  traceHead = (traceHead + 1) % trace.length;
  trace[traceHead] = level(n);
  const l = level(n);
  // fast attack, slow release: spikes register at once and stay readable
  if (l >= peakLevel) { peakLevel = l; shownN = n; }
  else { peakLevel = Math.max(l, peakLevel - 0.012); shownN = Math.max(n, 10 ** (peakLevel * LOG_TOP)); }
}

function mark(label) { marks.push({ head: traceHead, label, ticks: 0 }); if (marks.length > 6) marks.shift(); }

async function loop() {
  for (;;) {
    const t0 = performance.now();
    if (!document.hidden) await tick();
    await new Promise((r) => setTimeout(r, Math.max(0, TICK_MS - (performance.now() - t0))));
  }
}

// ---- stage: all drawn in one fragment shader ------------------------------------------------------------------------

const stage = $("stage");
const gl = stage.getContext("webgl2", { antialias: false, alpha: false, powerPreference: "high-performance" });
const VS = `#version 300 es
in vec2 p; out vec2 uv; void main(){ uv = (p*0.5+0.5)*vec2(1.,1.); uv.y = 1.-uv.y; gl_Position = vec4(p,0,1);} `;
const FS = `#version 300 es
precision highp float;
in vec2 uv; out vec4 o;
uniform vec4 uBall[2];      // x, y, r, alpha in world pixels
uniform float uLum[2];
uniform vec4 uOcc[2];       // x0, x1, y0, y1
uniform float uOccOn[2];
uniform sampler2D uXray;    // the model's occluder-free picture of the next frame
uniform sampler2D uErr;     // where it was wrong
uniform vec3 uSpot;         // surprise bloom x, y, strength
uniform float uRes, uTime, uHasX, uPeak, uSure;

float box(vec2 q, vec4 r, float aa){
  float d = max(max(r.x-q.x, q.x-r.y), max(r.z-q.y, q.y-r.w));
  return 1.-smoothstep(-aa, aa, d);
}
// Catmull-Rom sampling so 32x32 fields upscale without visible cells
vec4 cubic(sampler2D t, vec2 q){
  vec2 s = q*32.-.5, i = floor(s), f = s-i;
  vec2 w0 = f*(-.5+f*(1.-.5*f)), w1 = 1.+f*f*(-2.5+1.5*f), w2 = f*(.5+f*(2.-1.5*f)), w3 = f*f*(-.5+.5*f);
  vec2 w12 = w1+w2, off = w2/w12;
  vec2 a=(i-.5)/32., b=(i+.5+off)/32., c=(i+2.5)/32.;
  vec4 r = vec4(0);
  r += texture(t, vec2(a.x,a.y))*w0.x*w0.y + texture(t, vec2(b.x,a.y))*w12.x*w0.y + texture(t, vec2(c.x,a.y))*w3.x*w0.y;
  r += texture(t, vec2(a.x,b.y))*w0.x*w12.y + texture(t, vec2(b.x,b.y))*w12.x*w12.y + texture(t, vec2(c.x,b.y))*w3.x*w12.y;
  r += texture(t, vec2(a.x,c.y))*w0.x*w3.y + texture(t, vec2(b.x,c.y))*w12.x*w3.y + texture(t, vec2(c.x,c.y))*w3.x*w3.y;
  return r;
}
void main(){
  vec2 q = uv*32.;
  float aa = 32./uRes*.8;
  vec3 c = vec3(.012,.013,.016);
  // grid of the model's own patches, barely there
  vec2 g = abs(fract(q/8.+.5)-.5)*8.;
  c += .012*(1.-smoothstep(0.,aa*1.5,min(g.x,g.y)));

  // balls
  for(int i=0;i<2;i++){
    vec4 b = uBall[i];
    if(b.w<=0.) continue;
    float d = length(q-b.xy)-b.z;
    float m = (1.-smoothstep(-aa,aa,d))*b.w;
    vec2 n = (q-b.xy)/max(b.z,.01);
    float shade = .78+.22*clamp(dot(normalize(vec3(n,.7)),normalize(vec3(-.4,-.6,.7))),0.,1.);
    c = mix(c, vec3(.97,.98,1.)*uLum[i]*shade, m);
    c += .05*b.w*exp(-max(d,0.)*.9);
  }

  // occluders, in front of the balls
  float cover = 0.;
  for(int i=0;i<2;i++){
    if(uOccOn[i]<.5) continue;
    float m = box(q,uOcc[i],aa);
    cover = max(cover,m);
  }
  if(cover>0.){
    vec3 slab = vec3(.205,.215,.235);
    float hatch = smoothstep(.46,.5,abs(fract((q.x+q.y)*.5)-.5))*.018;
    c = mix(c, slab+hatch, cover);
  }

  // the model's belief, drawn only behind slabs: a thin contour around a faint fill, never mistaken for a solid ball
  if(uHasX>.5 && cover>0. && uSure>0.){
    float v = cubic(uXray, uv).r;
    float iso = uPeak*.58;
    float grad = max(length(vec2(dFdx(v),dFdy(v))),1e-4);
    float line = exp(-pow((v-iso)/grad/(uRes/820.),2.));
    c += cover*uSure*(line*vec3(.92,.95,1.)*.85 + smoothstep(iso*.55,iso*1.15,v)*vec3(.07,.08,.095));
  }

  // where the model was wrong
  float e = cubic(uErr, uv).r;
  c += vec3(1.,.72,.34)*smoothstep(.05,1.,e)*.9;

  // a ring that opens on a spike; brightness and shape carry the meaning, not only hue
  if(uSpot.z>0.){
    float d = length(q-uSpot.xy);
    float radius = 3.+ (1.-uSpot.z)*13.;
    float ring = exp(-pow((d-radius)/.55,2.))*uSpot.z;
    c += vec3(1.,.82,.55)*ring;
  }
  // vignette and grain so it does not look like a flat fill
  vec2 v = uv-.5;
  c *= 1.-.32*dot(v,v);
  float n = fract(sin(dot(gl_FragCoord.xy+uTime*60.,vec2(12.9898,78.233)))*43758.5453);
  c += (n-.5)*.012;
  o = vec4(pow(c,vec3(1./2.2))*.0+c,1.);
}`;

function shader(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
const prog = gl.createProgram();
gl.attachShader(prog, shader(gl.VERTEX_SHADER, VS));
gl.attachShader(prog, shader(gl.FRAGMENT_SHADER, FS));
gl.linkProgram(prog);
gl.useProgram(prog);
const U = (n) => gl.getUniformLocation(prog, n);
const vb = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, vb);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
const loc = gl.getAttribLocation(prog, "p");
gl.enableVertexAttribArray(loc);
gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

function fieldTexture(unit, uniform) {
  const t = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, 32, 32, 0, gl.RED, gl.UNSIGNED_BYTE, new Uint8Array(1024));
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.uniform1i(U(uniform), unit);
  return t;
}
const texX = fieldTexture(0, "uXray"), texE = fieldTexture(1, "uErr");
gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
const xrayTarget = new Float32Array(1024), xrayShown = new Float32Array(1024), errTarget = new Float32Array(1024), errShown = new Float32Array(1024);
const bytes = new Uint8Array(1024);
let hasX = 0;

function uploadXray(x) {
  if (!x) return;
  hasX = 1;
  xrayTarget.set(x);
}

function uploadField(tex, unit, a) {
  for (let i = 0; i < 1024; i++) bytes[i] = Math.max(0, Math.min(255, a[i] * 255));
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 32, 32, gl.RED, gl.UNSIGNED_BYTE, bytes);
}

let lastTickAt = 0, lastDraw = performance.now(), shownPeak = 0, shownMass = 0;

function resize() {
  const r = stage.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
  const w = Math.round(r.width * dpr), h = Math.round(r.height * dpr);
  if (stage.width !== w || stage.height !== h) { stage.width = w; stage.height = h; gl.viewport(0, 0, w, h); }
  gl.uniform1f(U("uRes"), w);
}

function draw(now) {
  resize();
  const dt = Math.min(now - lastDraw, 64) / 1000;
  lastDraw = now;
  // balls glide between 20 Hz physics ticks, so the world moves at the display's rate
  const f = REDUCED ? 1 : Math.min((now - lastTickAt) / TICK_MS, 1);
  const balls = curr ?? snapshot();
  const from = prev ?? balls;
  const ball = new Float32Array(8), lum = new Float32Array(2);
  balls.slice(0, 2).forEach((b, i) => {
    const p = from.find((q) => q.id === b.id) ?? b;
    const held = world.balls.find((w) => w.id === b.id)?.held;
    const live = world.balls.find((w) => w.id === b.id);
    const x = held ? live.x : p.x + (b.x - p.x) * f, y = held ? live.y : p.y + (b.y - p.y) * f;
    ball.set([x, y, b.r, b.alive ? 1 : 0], i * 4);
    lum[i] = b.lum;
  });
  gl.uniform4fv(U("uBall"), ball);
  gl.uniform1fv(U("uLum"), lum);
  const occ = new Float32Array(8), on = new Float32Array(2);
  world.occ.slice(0, 2).forEach((o, i) => { occ.set([o.x0, o.x1, o.y0, o.y1], i * 4); on[i] = 1; });
  gl.uniform4fv(U("uOcc"), occ);
  gl.uniform1fv(U("uOccOn"), on);

  // fields ease toward their newest value so nothing strobes at the tick rate
  const k = 1 - Math.exp(-dt / 0.045);
  for (let i = 0; i < 1024; i++) {
    xrayShown[i] += (xrayTarget[i] - xrayShown[i]) * k;
    errShown[i] += (errTarget[i] - errShown[i]) * k;
    errTarget[i] *= Math.exp(-dt / 0.22);
  }
  // How much ball the state holds behind the slabs, which gates the outline. Measured on held-out clips: 0.9 for a ball
  // hidden 16 frames, 0.2 six frames after the reveal that never came, zero by ten, under 0.13 with no ball at all.
  let peak = 0, mass = 0;
  for (const o of world.occ) {
    for (let y = 0; y < 32; y++) for (let x = Math.max(0, Math.ceil(o.x0)); x < Math.min(32, o.x1); x++) {
      const v = xrayShown[y * 32 + x];
      peak = Math.max(peak, v);
      mass += Math.max(0, v - 0.25);
    }
  }
  shownMass += (mass - shownMass) * (1 - Math.exp(-dt / 0.12));
  shownPeak = peak;
  const t = Math.max(0, Math.min(1, (shownMass - 0.2) / 0.4));
  gl.uniform1f(U("uPeak"), peak);
  gl.uniform1f(U("uSure"), t * t * (3 - 2 * t));
  uploadField(texX, 0, xrayShown);
  uploadField(texE, 1, errShown);
  surpriseSpot.a *= Math.exp(-dt / 0.42);
  gl.uniform3f(U("uSpot"), surpriseSpot.x, surpriseSpot.y, surpriseSpot.a > 0.02 ? surpriseSpot.a : 0);
  gl.uniform1f(U("uTime"), now / 1000 % 10);
  gl.uniform1f(U("uHasX"), hasX);
  gl.drawArrays(gl.TRIANGLES, 0, 3);

  drawMeter(now, f);
  requestAnimationFrame(draw);
}

// ---- meter and views ------------------------------------------------------------------------------------------------

const traceCanvas = $("trace"), tctx = traceCanvas.getContext("2d");
const elN = $("oneIn"), elM = $("meaning");
let lastText = 0;

function drawMeter(now, f) {
  const r = traceCanvas.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
  const W = Math.round(r.width * dpr), H = Math.round(r.height * dpr);
  if (traceCanvas.width !== W || traceCanvas.height !== H) { traceCanvas.width = W; traceCanvas.height = H; }
  const c = tctx;
  c.clearRect(0, 0, W, H);
  const pad = 2 * dpr, top = 2 * dpr, bottom = H - 3 * dpr, y = (l) => bottom - Math.min(l, 1.08) / 1.08 * (bottom - top);

  c.font = `${11 * dpr}px ui-sans-serif, -apple-system, sans-serif`;
  c.textBaseline = "bottom";
  for (const [l, label] of [[2, "1 in 100"], [4, "1 in 10,000"], [LOG_TOP, `all ${twoFigures(TOP)} lawful frames`]]) {
    const yy = Math.round(y(l / LOG_TOP)) + 0.5;
    c.strokeStyle = "#23262b";
    c.lineWidth = dpr;
    c.beginPath(); c.moveTo(0, yy); c.lineTo(W, yy); c.stroke();
    c.fillStyle = "#6c727a";
    const isTop = l === LOG_TOP;
    c.textBaseline = isTop ? "top" : "bottom";
    c.fillText(label, pad, yy + (isTop ? 2 : -2) * dpr);
  }

  const N = trace.length, step = W / (N - 1);
  const shift = REDUCED ? 0 : f * step;
  c.beginPath();
  for (let i = 0; i < N; i++) {
    const v = trace[(traceHead + 1 + i) % N], px = i * step - shift;
    if (i === 0) c.moveTo(px, y(v)); else c.lineTo(px, y(v));
  }
  c.lineTo(W, bottom); c.lineTo(-shift, bottom); c.closePath();
  const g = c.createLinearGradient(0, top, 0, bottom);
  g.addColorStop(0, "rgba(255,190,110,0.55)");
  g.addColorStop(0.55, "rgba(200,205,215,0.20)");
  g.addColorStop(1, "rgba(200,205,215,0.05)");
  c.fillStyle = g;
  c.fill();
  c.beginPath();
  for (let i = 0; i < N; i++) {
    const v = trace[(traceHead + 1 + i) % N], px = i * step - shift;
    if (i === 0) c.moveTo(px, y(v)); else c.lineTo(px, y(v));
  }
  c.strokeStyle = "#e8eaee";
  c.lineWidth = 1.4 * dpr;
  c.lineJoin = "round";
  c.stroke();

  c.textBaseline = "top";
  c.fillStyle = "#9aa0a8";
  for (const m of marks) {
    const back = (traceHead - m.head + N) % N;
    if (m.head === -1 || back >= N - 1) continue;
    const px = W - back * step - shift;
    c.strokeStyle = "#5a6068";
    c.lineWidth = dpr;
    c.beginPath(); c.moveTo(px + 0.5, top); c.lineTo(px + 0.5, bottom); c.stroke();
    const tw = c.measureText(m.label).width;
    c.fillText(m.label, Math.max(px - tw - 4 * dpr, pad), bottom - 13 * dpr);
  }

  if (now - lastText > 90 && engine) {
    lastText = now;
    const t = fmtOneIn(shownN);
    if (elN.textContent !== t) elN.textContent = t;
    const mean = meaning(shownN);
    if (elM.textContent !== mean) elM.textContent = mean;
    elN.style.color = peakLevel > level(SPIKE) ? "var(--spike)" : "var(--ink)";
  }
}

const views = ["vSaw", "vExp", "vErr"].map((id) => $(id).getContext("2d"));
const vImg = views.map((c) => c.createImageData(32, 32));

function drawViews(saw, mean, excess) {
  for (let i = 0; i < 1024; i++) {
    const a = Math.max(0, Math.min(1, saw[i])) * 255, b = Math.max(0, Math.min(1, mean[i])) * 255;
    const e = Math.max(0, Math.min(1, excess[i] / 30));
    vImg[0].data.set([a, a, a, 255], i * 4);
    vImg[1].data.set([b, b, b, 255], i * 4);
    vImg[2].data.set([e * 255, e * 200, e * 130, 255], i * 4);
    errTarget[i] = Math.max(errTarget[i], Math.min(excess[i] / 60, 1));
  }
  views.forEach((c, i) => c.putImageData(vImg[i], 0, 0));
}

// ---- direct manipulation ---------------------------------------------------------------------------------------------

let grab = null;

function toWorld(e) {
  const r = stage.getBoundingClientRect();
  return { x: ((e.clientX - r.left) / r.width) * SIZE, y: ((e.clientY - r.top) / r.height) * SIZE };
}

function ballAt(p) {
  const b = world.balls.filter((b) => b.alive).sort((a, c) => Math.hypot(a.x - p.x, a.y - p.y) - Math.hypot(c.x - p.x, c.y - p.y))[0];
  return b && Math.hypot(b.x - p.x, b.y - p.y) <= b.r + 1.6 ? b : null;
}

// a finger on a ball drags it; a finger anywhere else still scrolls the page
stage.addEventListener("touchstart", (e) => { if (ballAt(toWorld(e.touches[0]))) e.preventDefault(); }, { passive: false });

stage.addEventListener("pointerdown", (e) => {
  if (grab) return;
  const p = toWorld(e);
  const b = ballAt(p);
  if (!b) return;
  stage.setPointerCapture(e.pointerId);
  stage.classList.add("grabbing");
  b.held = true;
  grab = { b, dx: b.x - p.x, dy: b.y - p.y, id: e.pointerId, moved: 0, hist: [{ t: e.timeStamp, x: b.x, y: b.y }], startedAt: e.timeStamp };
  stopAttract();
});

stage.addEventListener("pointermove", (e) => {
  if (!grab || e.pointerId !== grab.id) return;
  const p = toWorld(e);
  const nx = p.x + grab.dx, ny = p.y + grab.dy;
  grab.moved += Math.hypot(nx - grab.b.x, ny - grab.b.y);
  grab.b.x = Math.max(grab.b.r, Math.min(SIZE - grab.b.r, nx));
  grab.b.y = Math.max(grab.b.r, Math.min(SIZE - grab.b.r, ny));
  grab.hist.push({ t: e.timeStamp, x: grab.b.x, y: grab.b.y });
  if (grab.hist.length > 6) grab.hist.shift();
  if (grab.moved > 1 && !grab.marked) { grab.marked = true; mark("you moved it"); caption("Nothing in its world can do that."); }
});

function release(e) {
  if (!grab || e.pointerId !== grab.id) return;
  const { b, hist, moved } = grab;
  b.held = false;
  stage.classList.remove("grabbing");
  if (moved < 0.6 && e.timeStamp - grab.startedAt < 350) {
    b.alive = false;
    mark("you deleted it");
    caption(hiddenFully(b) ? "Gone while out of sight. Watch where it should reappear." : "Gone in plain sight. Things do not do that here.");
  } else {
    // the ball leaves the hand at the hand's speed, limited to what the world's physics allows
    const a = hist[0], z = hist[hist.length - 1], ms = Math.max(z.t - a.t, 1);
    let vx = ((z.x - a.x) / ms) * TICK_MS, vy = ((z.y - a.y) / ms) * TICK_MS;
    const s = Math.hypot(vx, vy);
    if (s > 1.4) { vx *= 1.4 / s; vy *= 1.4 / s; }
    if (s > 0.1) { b.vx = vx; b.vy = vy; }
  }
  grab = null;
}
stage.addEventListener("pointerup", release);
stage.addEventListener("pointercancel", release);

// ---- buttons -----------------------------------------------------------------------------------------------------------

function uiState() {
  $("aWalls").setAttribute("aria-pressed", String(!world.wallsOn));
  $("aFreeze").setAttribute("aria-pressed", String(frozen));
  $("aGhost").setAttribute("aria-pressed", String(world.collide === false));
  $("aVanish").setAttribute("aria-pressed", String(wantVanish));
}

const act = (id, fn) => $(id).addEventListener("pointerdown", (e) => { if (e.button === 0) { stopAttract(); fn(); uiState(); } });
const key = (id, fn) => $(id).addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); stopAttract(); fn(); uiState(); } });
const both = (id, fn) => { act(id, fn); key(id, fn); };

both("aVanish", () => {
  wantVanish = !wantVanish;
  if (wantVanish && !crossingSoon()) { occludedScene(); wantVanish = true; }
  caption(wantVanish ? "Waiting for a ball to go fully behind a slab, then it is deleted." : "");
});
both("aReverse", () => { for (const b of world.balls) { b.vx *= -1; b.vy *= -1; } mark("reversed"); caption("Everything turned around with nothing to hit."); });
both("aWalls", () => { world.wallsOn = !world.wallsOn; mark(world.wallsOn ? "walls back" : "walls off"); caption(world.wallsOn ? "" : "Now watch a ball reach the edge. It keeps going, and the model expected a bounce."); });
both("aFreeze", () => { frozen = !frozen; mark(frozen ? "froze" : "released"); caption(frozen ? "Stopped dead, mid-flight. It expected the next frame to be different." : ""); });
both("aGhost", () => { world.collide = world.collide === false ? true : false; mark(world.collide === false ? "no collisions" : "collisions back"); caption(world.collide === false ? "Solid things now pass through each other." : ""); });
both("aNew", () => { frozen = false; newWorld({ occluders: 1 + (Math.random() < 0.25) }); caption(""); });

// ---- first visit: show the idea without asking for anything ----------------------------------------------------------

const capEl = $("caption");
let capTimer = 0;
function caption(text, ms = 5200) {
  capEl.textContent = text;
  capEl.style.opacity = text ? 1 : 0;
  clearTimeout(capTimer);
  if (text && ms) capTimer = setTimeout(() => (capEl.style.opacity = 0), ms);
}

let attract = true;
function stopAttract() {
  if (!attract) return;
  attract = false;
  wantVanish = false; // the tour may have armed it; from here on it is only armed by the reader
  uiState();
}

function crossingSoon() {
  return world.balls.some((b) => b.alive && (hiddenFully(b) || world.occ.some((o) => {
    const enter = b.vx > 0 ? (o.x0 + b.r + 1 - b.x) / b.vx : b.vx < 0 ? (o.x1 - b.r - 1 - b.x) / b.vx : Infinity;
    return enter > 0 && enter < 70 && o.x1 - o.x0 > 2 * b.r + 2;
  })));
}

function occludedScene() {
  // one ball crossing one slab, the demo's central case
  const w = 10 + Math.random() * 2, x0 = 11 + Math.random() * 2;
  const dir = Math.random() < 0.5 ? 1 : -1;
  const v = 0.42 + Math.random() * 0.2;
  const r = 2.4;
  const x = dir > 0 ? r + 1 : SIZE - r - 1;
  world = { balls: [{ x, y: 8 + Math.random() * 16, vx: dir * v, vy: (Math.random() - 0.5) * 0.18, r, lum: 0.92, alive: true, id: nextId++ }], occ: [{ x0, x1: x0 + w, y0: 0, y1: SIZE, lum: 0.45 }], wallsOn: true };
  needsReset = true;
  prev = curr = null;
  wantVanish = false;
}

async function attractLoop() {
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  while (attract) {
    occludedScene();
    caption("A ball goes behind the slab. The outline is read straight out of the model's state.", 0);
    let hid = false;
    for (let i = 0; i < 300 && attract && !hid; i++) { await wait(50); hid = world.balls[0].alive && hiddenFully(world.balls[0]); }
    if (!attract) break;
    for (let i = 0; i < 300 && attract && hiddenFully(world.balls[0]); i++) await wait(50);
    if (!attract) break;
    caption("It comes out about where the outline said it would.", 0);
    await wait(2600);
    if (!attract) break;
    occludedScene();
    caption("Again, but this time the ball is deleted once it is out of sight.", 0);
    wantVanish = true;
    uiState();
    for (let i = 0; i < 400 && attract && wantVanish; i++) await wait(50);
    if (!attract) break;
    await wait(2400);
    if (!attract) break;
    caption("Nothing came out. With no ball to confirm it, the outline faded within about a second.", 0);
    await wait(2800);
    if (!attract) break;
    newWorld({ balls: 2, occluders: 1 });
    caption("Your turn. Drag a ball somewhere it could not have gone and watch the meter.", 0);
    await wait(6000);
  }
  caption("", 0);
}

// ---- start -----------------------------------------------------------------------------------------------------------

async function start() {
  requestAnimationFrame(draw);
  world.balls.forEach((b) => (b.id = nextId++));
  uiState();
  loop();
  try {
    const params = new URLSearchParams(location.search);
    let weights = null;
    if (params.get("engine") !== "cpu" && navigator.gpu) {
      weights = await loadWeights(params.get("model") ?? "./model");
      engine = await GpuEngine.create(weights).catch(() => null);
    }
    if (!engine) {
      const url = params.get("cpuModel") ?? "./model-cpu";
      weights = await fetch(`${url}.json`).then((r) => r.json());
      engine = new WorkerEngine(url);
    }
    nullTable = weights.null;
    TOP = nullTable.one_in[0];
    LOG_TOP = Math.log10(TOP);
    needsReset = true;
    const m = (weights.params / 1e6).toFixed(1);
    $("status").textContent = engine.kind === "webgpu" ? `${m}M parameters, WebGPU, on your GPU.` : `${m}M parameters on your CPU, because this browser has no WebGPU.`;
    window.__ready = true;
    if (!params.has("noattract")) attractLoop();
  } catch (err) {
    $("status").textContent = `Could not load the model: ${err.message}`;
  }
}
start();
window.__sm = { get world() { return world; }, get last() { return last; }, get n() { return shownN; }, get peak() { return shownPeak; }, get mass() { return shownMass; }, engineKind: () => engine?.kind };
