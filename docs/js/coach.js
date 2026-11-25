// Typed tactics without the local server. The keyword parser, prompt and schema
// come from football/coach.py (exported to coach.json); phrases the keywords miss
// go to a small model that WebLLM runs on the visitor's GPU, in a worker.
export const WEBLLM_URL = "https://esm.run/@mlc-ai/web-llm@0.2.89";
export const MODEL = { name: "Qwen2.5 1.5B", mb: 880 };
const MODEL_IDS = { f16: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC", f32: "Qwen2.5-1.5B-Instruct-q4f32_1-MLC" };

let data = null;
export function setCoachData(d) { data = d; }
export async function loadCoachData() {
  const res = await fetch(new URL("./coach.json", import.meta.url));
  if (!res.ok) throw new Error(`coach.json answered ${res.status}`);
  data = await res.json();
  return data;
}

// Same rules as keyword_tactic in football/coach.py: the first match per dimension wins,
// a negation in front flips it, mild wording halves it.
export function keywordTactic(text, d = data) {
  const t = text.toLowerCase();
  const z = Object.fromEntries(d.tactics.map((k) => [k, 0]));
  const done = new Set();
  for (const [pat, dim, val] of d.keywords) {
    if (done.has(dim) || !new RegExp(pat).test(t)) continue;
    const v = new RegExp(`${d.negation}(${pat})`).test(t) ? -val : val;
    z[dim] = v * (new RegExp(d.mild).test(t) ? 0.5 : 1);
    done.add(dim);
  }
  return z;
}

// Same as _clamp: one number per dimension in [-1, 1], anything unusable becomes 0.
export function clamp(z, d = data) {
  const out = {};
  for (const k of d.tactics) {
    const raw = z && typeof z === "object" ? z[k] : undefined;
    let v = ["number", "string", "boolean"].includes(typeof raw) ? Number(raw) : 0;
    if (Number.isNaN(v)) v = 0;
    out[k] = Math.round(Math.max(-1, Math.min(1, v)) * 100) / 100;
  }
  return out;
}

let engine = null; // Promise of { engine, id } once loading has started

async function pickModel() {
  if (!navigator.gpu) throw new Error("this browser has no WebGPU");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("WebGPU found no usable graphics adapter");
  return adapter.features.has("shader-f16") ? MODEL_IDS.f16 : MODEL_IDS.f32;
}

export function modelLoading() { return engine !== null; }

export function loadModel(onProgress) {
  engine ??= (async () => {
    const id = await pickModel();
    const { CreateWebWorkerMLCEngine } = await import(WEBLLM_URL);
    const worker = new Worker(new URL("./coach_worker.js", import.meta.url), { type: "module" });
    const failed = new Promise((_, reject) => {
      worker.addEventListener("error", (e) => reject(new Error(e.message || "the model worker did not start")));
    });
    const e = await Promise.race([
      CreateWebWorkerMLCEngine(worker, id, { initProgressCallback: (r) => onProgress?.(r.progress, r.text) }),
      failed,
    ]);
    return { engine: e, id };
  })();
  engine.catch(() => { engine = null; });
  return engine;
}

export async function askModel(text, onProgress) {
  const { engine: e, id } = await loadModel(onProgress);
  const t0 = performance.now();
  const reply = await e.chat.completions.create({
    messages: [{ role: "system", content: data.system }, { role: "user", content: text }],
    temperature: 0,
    seed: 0,
    max_tokens: 100,
    response_format: { type: "json_object", schema: JSON.stringify(data.schema) },
  });
  let raw;
  try {
    raw = JSON.parse(reply.choices[0].message.content);
  } catch {
    throw Object.assign(new Error("the model's answer was not valid JSON"), { loaded: true });
  }
  return { raw, z: clamp(raw), model: id, latency_s: (performance.now() - t0) / 1000 };
}

// Keywords first, the in-browser model only when they find nothing (like Coach in coach.py).
// Returns { z, source: "keywords" | "browser", model?, latency_s?, error?, loaded? }.
export async function browserCoach(text, onProgress) {
  text = text.trim().slice(0, 300);
  const kw = keywordTactic(text);
  if (Object.values(kw).some((v) => v !== 0)) return { z: kw, source: "keywords" };
  try {
    return { ...(await askModel(text, onProgress)), source: "browser" };
  } catch (err) {
    return { z: kw, source: "keywords", error: err.message || String(err), loaded: Boolean(err.loaded) };
  }
}
