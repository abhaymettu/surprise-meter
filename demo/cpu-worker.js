import { loadWeights, CpuEngine } from "./engine.js";

let ready = null;
onmessage = async ({ data }) => {
  if (data.url) ready = loadWeights(data.url).then((w) => new CpuEngine(w));
  const engine = await ready;
  if (data.reset) engine.reset();
  if (data.frame) {
    const r = await engine.step(data.frame);
    postMessage(r, [r.mean.buffer, r.sigma.buffer, ...(r.xray ? [r.xray.buffer] : [])]);
  }
};
