import { CFG, Game, MOVE_DIRS, N_ACTIONS, N_MOVES, TEAM_CFG } from "./game.js";
import { ChaseBot, Policy, RandomBot, TeamChaseBot } from "./policy.js";

// Tactic dimensions, in the order the conditioned network expects (football/tactics.py).
const TACTICS = [
  { id: "press", label: "Press", lo: "deep", hi: "high" },
  { id: "flank", label: "Flank", lo: "right", hi: "left" },
  { id: "spread", label: "Spread", lo: "together", hi: "apart" },
  { id: "possession", label: "Ball", lo: "let them", hi: "keep it" },
];
const PRESETS = [
  { label: "Neutral", z: [0, 0, 0, 0] },
  { label: "Press high", z: [1, 0, 0, 0] },
  { label: "Park the bus", z: [-1, 0, 0, 0] },
  { label: "Left wing", z: [0, 1, 0, 0] },
  { label: "Right wing", z: [0, -1, 0, 0] },
  { label: "Spread out", z: [0, 0, 1, 0] },
  { label: "Stick together", z: [0, 0, -1, 0] },
  { label: "Keep the ball", z: [0, 0, 0, 1] },
];

const $ = (id) => document.getElementById(id);
const canvas = $("pitch");
const ctx = canvas.getContext("2d");
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// ------------------------------------------------------------------ geometry
// Everything is drawn in "logical" pixels (SCALE per game unit). The canvas is then
// scaled to its on-screen size and the device pixel ratio, so it stays sharp.
const SCALE = 8;
const MARGIN = 24;
const LW = CFG.field_l * SCALE + 2 * MARGIN;
const LH = CFG.field_h * SCALE + 2 * MARGIN;
const X = (x) => MARGIN + x * SCALE;
const Y = (y) => MARGIN + (CFG.field_h - y) * SCALE;

const COLORS = {
  surround: "#25703a",
  grassA: "#2e8b3f",
  grassB: "#287c38",
  line: "rgba(246, 252, 244, 0.92)",
  team: ["#d6382a", "#1d5bd6"],
  ink: "#14221b",
};

// ---------------------------------------------------------------------- state
const state = {
  game: new Game(),
  prev: null, // positions before the latest step, for smooth drawing between steps
  opponents: {},
  info: {}, // id -> { label, short, note, format, tactics }
  format: "1v1",
  agent: null, // right team (blue)
  left: null, // left player or team (red); null means the human
  mate: null, // 2v2: the human's tactic-conditioned AI teammate
  z: new Float32Array(TACTICS.length), // the red team's current tactic
  names: ["You", "Pro"],
  tags: ["You", "Pro"],
  score: [0, 0],
  firstTo: 5,
  fps: 30,
  phase: "ready", // ready | playing | paused | goal | over
  watch: false,
  greedy: false,
  acc: 0,
  last: performance.now(),
  holdUntil: 0,
  trail: [],
  pulses: [],
  spin: 0,
};

const brain = { moves: new Float32Array(N_MOVES), kick: 0, conf: 0, picked: 0, live: false };

// ------------------------------------------------------------------ canvas size
const pitchLayer = document.createElement("canvas");
let k = 1;

function resize() {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
  const w = Math.round(rect.width * dpr);
  const h = Math.round((w * LH) / LW);
  if (canvas.width === w && canvas.height === h) return;
  canvas.width = w;
  canvas.height = h;
  k = w / LW;
  pitchLayer.width = w;
  pitchLayer.height = h;
  const p = pitchLayer.getContext("2d");
  p.setTransform(k, 0, 0, k, 0, 0);
  paintPitch(p);
}
new ResizeObserver(resize).observe(canvas);

// The static pitch is painted once per size, then copied every frame.
function paintPitch(p) {
  const c = CFG;
  const fw = c.field_l * SCALE;
  const fh = c.field_h * SCALE;

  p.fillStyle = COLORS.surround;
  p.fillRect(0, 0, LW, LH);
  for (let i = 0; i < 10; i++) {
    p.fillStyle = i % 2 ? COLORS.grassB : COLORS.grassA;
    p.fillRect(MARGIN + (i * fw) / 10, MARGIN, fw / 10 + 0.6, fh);
  }

  // markings
  p.strokeStyle = COLORS.line;
  p.fillStyle = COLORS.line;
  p.lineWidth = 2;
  p.strokeRect(X(0), Y(c.field_h), fw, fh);
  p.beginPath();
  p.moveTo(X(c.field_l / 2), Y(c.field_h));
  p.lineTo(X(c.field_l / 2), Y(0));
  p.stroke();
  p.beginPath();
  p.arc(X(c.field_l / 2), Y(c.field_h / 2), 8 * SCALE, 0, 2 * Math.PI);
  p.stroke();
  p.beginPath();
  p.arc(X(c.field_l / 2), Y(c.field_h / 2), 3, 0, 2 * Math.PI);
  p.fill();
  // penalty and goal areas (cosmetic: the physics has no special rules there)
  const mid = c.field_h / 2;
  for (const dir of [0, 1]) {
    const edge = dir === 0 ? 0 : c.field_l;
    const sign = dir === 0 ? 1 : -1;
    for (const [depth, half] of [[14, 20], [5, 12]]) {
      p.strokeRect(
        Math.min(X(edge), X(edge + sign * depth)),
        Y(mid + half),
        depth * SCALE,
        half * 2 * SCALE
      );
    }
    p.beginPath();
    p.arc(X(edge + sign * 10), Y(mid), 3, 0, 2 * Math.PI);
    p.fill();
  }
  // corner arcs
  for (const [cx, cy, a0] of [[0, 0, 0], [c.field_l, 0, Math.PI / 2], [c.field_l, c.field_h, Math.PI], [0, c.field_h, -Math.PI / 2]]) {
    p.beginPath();
    p.arc(X(cx), Y(cy), 2 * SCALE, a0 === 0 ? -Math.PI / 2 : a0 - Math.PI / 2, a0 === 0 ? 0 : a0);
    p.stroke();
  }

  // goals with a net
  const gTop = Y(mid + c.goal_w / 2);
  const gh = c.goal_w * SCALE;
  const depth = 2.5 * SCALE;
  [[X(0) - depth, COLORS.team[0]], [X(c.field_l), COLORS.team[1]]].forEach(([gx, col]) => {
    p.fillStyle = "rgba(255,255,255,0.2)";
    p.fillRect(gx, gTop, depth, gh);
    p.save();
    p.beginPath();
    p.rect(gx, gTop, depth, gh);
    p.clip();
    p.strokeStyle = "rgba(255,255,255,0.45)";
    p.lineWidth = 1;
    for (let d = -gh; d < depth + gh; d += 6) {
      p.beginPath(); p.moveTo(gx + d, gTop); p.lineTo(gx + d - gh, gTop + gh); p.stroke();
      p.beginPath(); p.moveTo(gx + d, gTop); p.lineTo(gx + d + gh, gTop + gh); p.stroke();
    }
    p.restore();
    p.strokeStyle = col;
    p.lineWidth = 3.5;
    p.strokeRect(gx, gTop, depth, gh);
  });

  // soft vignette so the pitch feels lit from the middle
  const v = p.createRadialGradient(LW / 2, LH / 2, LH * 0.35, LW / 2, LH / 2, LW * 0.62);
  v.addColorStop(0, "rgba(0,0,0,0)");
  v.addColorStop(1, "rgba(0,18,8,0.28)");
  p.fillStyle = v;
  p.fillRect(0, 0, LW, LH);
}

// ---------------------------------------------------------------------- input
// Physical key positions (event.code), so WASD works on any keyboard layout.
const held = new Set();
const virtual = new Set(); // keys pressed on screen with a finger or mouse
const GAME_CODES = new Set(["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "KeyW", "KeyA", "KeyS", "KeyD", "Space"]);
const isDown = (...codes) => codes.some((c) => held.has(c) || virtual.has(c));

function humanAction() {
  const dx = (isDown("ArrowRight", "KeyD") ? 1 : 0) - (isDown("ArrowLeft", "KeyA") ? 1 : 0);
  const dy = (isDown("ArrowUp", "KeyW") ? 1 : 0) - (isDown("ArrowDown", "KeyS") ? 1 : 0);
  return Game.keyboardAction(dx, dy, isDown("Space"), 0);
}

function isFormControl(t) {
  return t instanceof HTMLElement && t.matches("select, input, textarea");
}

window.addEventListener("keydown", (e) => {
  if (e.key === "Meta") held.clear(); // macOS drops key-up events while Cmd is held
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (isFormControl(e.target)) return;

  if (e.code === "KeyP" || e.code === "Escape") {
    if (!e.repeat) togglePause();
    return;
  }
  if (e.code === "KeyR") {
    // Only outside live play, so a slip next to WASD never throws the match away.
    if (!e.repeat && state.phase !== "playing" && state.phase !== "goal") startMatch();
    return;
  }
  if (!GAME_CODES.has(e.code)) return;

  const onButton = e.target instanceof HTMLElement && e.target.closest("button");
  if (e.code === "Space" && onButton) return; // let a focused button handle its own Space
  e.preventDefault();
  held.add(e.code);
  if (e.code === "Space" && !e.repeat) {
    if (state.phase === "ready" || state.phase === "over") startMatch();
    else if (state.phase === "paused") togglePause();
  }
});
window.addEventListener("keyup", (e) => held.delete(e.code));
window.addEventListener("blur", () => held.clear());
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    held.clear();
    if (state.phase === "playing") togglePause();
  }
});

// On-screen keys: the same keycaps that light up on the keyboard also work as a touch controller.
const keycaps = [...document.querySelectorAll(".key[data-code]")];
keycaps.forEach((el) => {
  const code = el.dataset.code;
  const release = () => virtual.delete(code);
  el.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    try { el.setPointerCapture(e.pointerId); } catch { /* not every pointer can be captured */ }
    virtual.add(code);
  });
  el.addEventListener("pointerup", release);
  el.addEventListener("pointercancel", release);
  el.addEventListener("lostpointercapture", release);
  el.addEventListener("contextmenu", (e) => e.preventDefault());
});
function paintKeycaps() {
  for (const el of keycaps) el.classList.toggle("down", isDown(el.dataset.code));
}

// ------------------------------------------------------------------ simulation
function snapshot() {
  const g = state.game;
  state.prev = { pos: g.pos.map((p) => [...p]), ball: [...g.ball] };
}

function kickoff() {
  state.game.reset();
  snapshot();
  state.trail.length = 0;
  state.pulses.length = 0;
  state.acc = 0;
  setClock();
}

function setClock() {
  $("clock").firstElementChild.style.transform = `scaleX(${Math.min(1, state.game.t / CFG.max_steps)})`;
}

function sample(p) {
  let u = Math.random();
  for (let i = 0; i < p.length; i++) { u -= p[i]; if (u <= 0) return i; }
  return p.length - 1;
}
function argmax(p) {
  let best = 0;
  for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
  return best;
}

// The blue agent's action, plus a smoothed copy of its odds for the side panel.
function agentAct(obs) {
  const player = state.agent;
  if (!player.probs) return player.act(obs, state.greedy);
  const p = player.probs(obs);
  const action = state.greedy ? argmax(p) : sample(p);

  const moves = new Float32Array(N_MOVES);
  let kick = 0, entropy = 0;
  for (let i = 0; i < N_ACTIONS; i++) {
    moves[i % N_MOVES] += p[i];
    if (i >= N_MOVES) kick += p[i];
    if (p[i] > 1e-12) entropy -= p[i] * Math.log(p[i]);
  }
  const conf = 1 - entropy / Math.log(N_ACTIONS);
  const s = brain.live ? 0.35 : 1; // ease between steps so the panel does not flicker
  for (let m = 0; m < N_MOVES; m++) brain.moves[m] += s * (moves[m] - brain.moves[m]);
  brain.kick += s * (kick - brain.kick);
  brain.conf += s * (conf - brain.conf);
  brain.picked = action % N_MOVES;
  brain.live = true;
  return action;
}

// Red team in 2v2: the human (or the watched player) in slot 0, the coached teammate in slot 1.
function teamActions(g) {
  const z = state.z;
  const red0 = state.left ? state.left.act(g.observe(0), state.greedy, Math.random, z) : humanAction();
  const red1 = (state.left || state.mate).act(g.observe(1), state.greedy, Math.random, z);
  return [red0, red1, state.agent.act(g.observe(2), state.greedy), state.agent.act(g.observe(3), state.greedy)];
}

function tick() {
  const g = state.game;
  snapshot();
  let r;
  if (g.P === 1) {
    const a0 = state.left ? state.left.act(g.observe(0), state.greedy) : humanAction();
    const a1 = agentAct(g.observe(1));
    r = g.step([a0, a1]);
  } else {
    r = g.step(teamActions(g));
  }

  const speed = Math.hypot(g.ballVel[0], g.ballVel[1]);
  state.spin += speed * 0.14;
  state.trail.push([g.ball[0], g.ball[1]]);
  if (state.trail.length > 12) state.trail.shift();
  for (let i = 0; i < g.M; i++) {
    if (g.cooldown[i] === CFG.kick_cooldown) state.pulses.push({ i, x: g.pos[i][0], y: g.pos[i][1], born: performance.now() });
  }
  setClock();
  if (g.P === 1) paintBrain();

  if (!r.done) return;
  if (r.goal !== 0) {
    const side = r.goal === 1 ? 0 : 1;
    state.score[side] += 1;
    state.phase = "goal";
    state.holdUntil = performance.now() + 1300;
    showGoal(side);
    updateScore();
  } else {
    say("Time. Nobody scored, so it is back to kickoff.");
    kickoff();
  }
}

function afterGoal() {
  $("goal-banner").classList.remove("show");
  if (Math.max(...state.score) >= state.firstTo) return finishMatch();
  kickoff();
  state.phase = "playing";
}

// ------------------------------------------------------------------ match flow
function startMatch() {
  applySettings();
  state.score = [0, 0];
  state.phase = "playing";
  kickoff();
  hideOverlay();
  $("goal-banner").classList.remove("show");
  updateScore();
  setPauseButton();
  const team = state.format === "2v2";
  say(state.watch ? `Watching two ${team ? "teams" : "players"}.`
    : team ? "Kick off. You and your AI teammate are red and attack to the right."
    : "Kick off. You are red and attack to the right.");
  focusPitch();
}

function togglePause() {
  if (state.phase === "playing") {
    state.phase = "paused";
    showOverlay("paused");
  } else if (state.phase === "paused") {
    state.phase = "playing";
    state.acc = 0;
    hideOverlay();
    focusPitch();
  }
  setPauseButton();
}

function finishMatch() {
  state.phase = "over";
  const [a, b] = state.score;
  const leftWon = a > b;
  if (!state.watch) recordResult(leftWon);
  showOverlay("over");
  setPauseButton();
  say(`Final score ${a} to ${b}.`);
}

function resetToReady() {
  state.phase = "ready";
  state.score = [0, 0];
  kickoff();
  showOverlay("ready");
  updateScore();
  setPauseButton();
  say("Pick an opponent and kick off.");
}

function focusPitch() {
  canvas.focus({ preventScroll: true });
}

function setPauseButton() {
  const b = $("btn-pause");
  b.disabled = state.phase !== "playing" && state.phase !== "paused";
  b.textContent = state.phase === "paused" ? "Resume" : "Pause";
}

// -------------------------------------------------------------------- records
const recordKey = () => `record:${state.format === "2v2" ? "2v2:" : ""}${$("opponent").value}`;
function readRecord(key) {
  try {
    return JSON.parse(localStorage.getItem(key) || '{"w":0,"l":0}');
  } catch {
    return { w: 0, l: 0 };
  }
}
function recordResult(won) {
  const key = recordKey();
  const rec = readRecord(key);
  if (won) rec.w += 1; else rec.l += 1;
  try { localStorage.setItem(key, JSON.stringify(rec)); } catch { /* storage can be blocked; the session still works */ }
  updateRecord();
}
function updateRecord() {
  const el = $("record");
  if (state.watch) { el.textContent = ""; return; }
  const rec = readRecord(recordKey());
  el.textContent = `Your record against ${state.names[1]}: ${rec.w} won, ${rec.l} lost.`;
}

// ------------------------------------------------------------------------ DOM
function say(text) { $("status").textContent = text; }

function updateScore() {
  $("score-left").textContent = state.score[0];
  $("score-right").textContent = state.score[1];
  $("name-left").textContent = state.names[0];
  $("name-right").textContent = state.names[1];
  $("round-label").textContent = `First to ${state.firstTo}`;
  $("score-left").setAttribute("aria-label", `${state.names[0]} goals`);
  $("score-right").setAttribute("aria-label", `${state.names[1]} goals`);
}

function showGoal(side) {
  const b = $("goal-banner");
  b.className = `goal-banner ${side === 0 ? "red" : "blue"}`;
  const who = side === 0 ? (state.watch ? state.names[0] : "You") : state.names[1];
  $("goal-who").textContent = side === 0 && !state.watch ? "You scored" : `${who} scored`;
  void b.offsetWidth; // restart the transition
  b.classList.add("show");
  say(`${who} scored. ${state.score[0]} to ${state.score[1]}.`);
}

function showOverlay(kind) {
  const [a, b] = state.score;
  const t = $("ov-title"), s = $("ov-sub"), p = $("ov-primary"), q = $("ov-secondary"), h = $("ov-hint");
  q.hidden = true;
  if (kind === "ready") {
    const team = state.format === "2v2";
    t.textContent = state.watch ? `Watch two ${team ? "teams" : "players"}`
      : team ? "Play 2v2 with a teammate you can coach" : "Beat the agent that taught itself football";
    s.textContent = state.watch
      ? `${state.names[0]} on the left, ${state.names[1]} on the right. First to ${state.firstTo}.`
      : team ? `You and your AI teammate are red and attack to the right. Tell it how to play in the coach panel. First to ${state.firstTo}.`
      : `You are red and attack to the right. First to ${state.firstTo} goals wins.`;
    p.textContent = state.watch ? "Start match" : "Kick off";
    h.textContent = "Or press Space.";
  } else if (kind === "paused") {
    t.textContent = "Paused";
    s.textContent = `${state.names[0]} ${a}, ${state.names[1]} ${b}. First to ${state.firstTo}.`;
    p.textContent = "Resume";
    q.hidden = false;
    h.textContent = "Press P or Space to carry on, R to restart.";
  } else {
    const leftWon = a > b;
    t.textContent = state.watch
      ? `${leftWon ? state.names[0] : state.names[1]} wins ${Math.max(a, b)} to ${Math.min(a, b)}`
      : leftWon ? `You win ${a} to ${b}` : `${state.names[1]} wins ${b} to ${a}`;
    const rec = readRecord(recordKey());
    s.textContent = state.watch ? "Run it again, or change the players below." : `Your record against ${state.names[1]}: ${rec.w} won, ${rec.l} lost.`;
    p.textContent = "Play again";
    h.textContent = "Or press Space.";
  }
  $("overlay").hidden = false;
}
function hideOverlay() { $("overlay").hidden = true; }

// What the blue agent is "thinking": a 3x3 compass of move odds plus kick and confidence.
const compass = $("compass");
const GLYPH = { "-1,1": "↖", "0,1": "↑", "1,1": "↗", "-1,0": "←", "0,0": "stay", "1,0": "→", "-1,-1": "↙", "0,-1": "↓", "1,-1": "↘" };
const cells = new Array(N_MOVES);
for (let row = 0; row < 3; row++) {
  for (let col = 0; col < 3; col++) {
    // Find which move points to this cell. The agent plays the right side, so its x is mirrored on screen.
    const sx = col - 1, sy = 1 - row;
    const m = MOVE_DIRS.findIndex((d) => Math.sign(Math.round(-d[0] * 10)) === sx && Math.sign(Math.round(d[1] * 10)) === sy);
    const cell = document.createElement("div");
    cell.className = "cell";
    cell.innerHTML = `<span class="g">${GLYPH[`${sx},${sy}`]}</span><span class="n">0%</span>`;
    compass.append(cell);
    cells[m] = cell;
  }
}

function paintBrain(force = false) {
  if (!brain.live && !force) return;
  for (let m = 0; m < N_MOVES; m++) {
    const v = brain.moves[m];
    const a = 0.06 + 0.94 * Math.min(1, v / 0.55);
    const cell = cells[m];
    cell.style.setProperty("--a", a.toFixed(3));
    cell.classList.toggle("hot", a > 0.55);
    cell.classList.toggle("picked", m === brain.picked);
    cell.lastElementChild.textContent = brain.live ? `${Math.round(v * 100)}%` : "–";
  }
  $("kick-bar").style.transform = `scaleX(${brain.kick.toFixed(3)})`;
  $("kick-val").textContent = `${Math.round(brain.kick * 100)}%`;
  $("conf-bar").style.transform = `scaleX(${Math.max(0, brain.conf).toFixed(3)})`;
  $("conf-val").textContent = `${Math.round(Math.max(0, brain.conf) * 100)}%`;
}

// ------------------------------------------------------------------- drawing
function draw(now) {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.drawImage(pitchLayer, 0, 0);
  ctx.setTransform(k, 0, 0, k, 0, 0);

  const g = state.game;
  const c = CFG;
  const a = state.phase === "playing" && state.prev ? Math.min(1, state.acc / (1000 / state.fps)) : 1;
  const lerp = (p0, p1) => p0 + (p1 - p0) * a;
  const at = (cur, old) => (state.prev ? [lerp(old[0], cur[0]), lerp(old[1], cur[1])] : cur);
  const players = g.pos.map((p, i) => at(p, state.prev ? state.prev.pos[i] : p));
  const ball = at(g.ball, state.prev ? state.prev.ball : g.ball);

  // ball trail
  if (!reduceMotion && state.trail.length > 1 && Math.hypot(g.ballVel[0], g.ballVel[1]) > 0.8) {
    const n = state.trail.length;
    for (let j = 0; j < n; j++) {
      const [tx, ty] = state.trail[j];
      ctx.beginPath();
      ctx.arc(X(tx), Y(ty), Math.max(2, c.ball_r * SCALE * (0.35 + 0.5 * (j / n))), 0, 2 * Math.PI);
      ctx.fillStyle = `rgba(255,255,255,${(0.28 * (j + 1)) / n})`;
      ctx.fill();
    }
  }

  // kick pulses
  if (!reduceMotion) {
    state.pulses = state.pulses.filter((p) => now - p.born < 320);
    for (const p of state.pulses) {
      const t = (now - p.born) / 320;
      ctx.beginPath();
      ctx.arc(X(p.x), Y(p.y), (c.player_r + 0.8 + t * 3.2) * SCALE, 0, 2 * Math.PI);
      ctx.strokeStyle = `rgba(255,255,255,${0.75 * (1 - t)})`;
      ctx.lineWidth = 3 * (1 - t) + 1;
      ctx.stroke();
    }
  }

  for (let i = 0; i < g.M; i++) drawPlayer(i, players[i], g);
  drawBall(ball);

  // name tags above the players so you always know who is who
  for (let i = 0; i < g.M; i++) if (state.tags[i]) drawTag(state.tags[i], players[i]);

  if (state.phase === "paused" || state.phase === "over") {
    ctx.fillStyle = "rgba(10,28,18,0.0)";
    ctx.fillRect(0, 0, LW, LH);
  }
}

function drawPlayer(i, [px, py], g) {
  const r = CFG.player_r * SCALE;
  const cx = X(px), cy = Y(py);

  ctx.fillStyle = "rgba(0,20,8,0.28)";
  ctx.beginPath();
  ctx.ellipse(cx + 2, cy + 4, r, r * 0.88, 0, 0, 2 * Math.PI);
  ctx.fill();

  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, 2 * Math.PI);
  ctx.fillStyle = COLORS.team[g.teamOf(i)];
  ctx.fill();
  ctx.beginPath();
  ctx.arc(cx - r * 0.25, cy - r * 0.3, r * 0.55, 0, 2 * Math.PI);
  ctx.fillStyle = "rgba(255,255,255,0.18)";
  ctx.fill();

  // kick-ready ring: fills back up after each kick
  const ready = 1 - g.cooldown[i] / CFG.kick_cooldown;
  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(255,255,255,0.28)";
  ctx.beginPath();
  ctx.arc(cx, cy, r + 3.5, 0, 2 * Math.PI);
  ctx.stroke();
  if (ready > 0) {
    ctx.strokeStyle = ready >= 1 ? "#ffffff" : "#f5c518";
    ctx.beginPath();
    ctx.arc(cx, cy, r + 3.5, -Math.PI / 2, -Math.PI / 2 + ready * 2 * Math.PI);
    ctx.stroke();
  }

  // heading dot
  const [vx, vy] = g.vel[i];
  const sp = Math.hypot(vx, vy);
  if (sp > 0.15) {
    ctx.beginPath();
    ctx.arc(cx + (vx / sp) * r * 0.6, cy - (vy / sp) * r * 0.6, 3.2, 0, 2 * Math.PI);
    ctx.fillStyle = "#fff";
    ctx.fill();
  }
}

function drawBall([bx, by]) {
  const br = Math.max(4.5, CFG.ball_r * SCALE);
  const cx = X(bx), cy = Y(by);
  ctx.fillStyle = "rgba(0,20,8,0.3)";
  ctx.beginPath();
  ctx.ellipse(cx + 1.5, cy + 3, br, br * 0.85, 0, 0, 2 * Math.PI);
  ctx.fill();

  ctx.beginPath();
  ctx.arc(cx, cy, br, 0, 2 * Math.PI);
  ctx.fillStyle = "#fbfcfa";
  ctx.fill();
  ctx.lineWidth = 1.2;
  ctx.strokeStyle = "rgba(20,34,27,0.7)";
  ctx.stroke();

  // a rolling patch, so you can see the ball spin as it travels
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(state.spin);
  ctx.beginPath();
  for (let j = 0; j < 5; j++) {
    const ang = (j / 5) * 2 * Math.PI - Math.PI / 2;
    const rr = br * 0.5;
    ctx[j ? "lineTo" : "moveTo"](Math.cos(ang) * rr, Math.sin(ang) * rr);
  }
  ctx.closePath();
  ctx.fillStyle = "rgba(20,34,27,0.85)";
  ctx.fill();
  ctx.restore();
}

function drawTag(label, [px, py]) {
  ctx.font = '700 12px "Hanken Grotesk", system-ui, sans-serif';
  const w = ctx.measureText(label).width + 14;
  const x = X(px) - w / 2;
  const y = Y(py) - CFG.player_r * SCALE - 26;
  ctx.fillStyle = "rgba(20,34,27,0.78)";
  ctx.beginPath();
  ctx.roundRect(x, y, w, 18, 4);
  ctx.fill();
  ctx.fillStyle = "#fff";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(label, X(px), y + 9.5);
}

// ----------------------------------------------------------------------- loop
function frame(now) {
  const dt = Math.min(now - state.last, 100);
  state.last = now;
  if (state.phase === "playing" && state.agent) {
    state.acc += dt;
    const stepMs = 1000 / state.fps;
    while (state.acc >= stepMs && state.phase === "playing") {
      tick();
      state.acc -= stepMs;
    }
  } else if (state.phase === "goal" && now >= state.holdUntil) {
    afterGoal();
  }
  draw(now);
  paintKeycaps();
  requestAnimationFrame(frame);
}

// ------------------------------------------------------------------- settings
function playerFor(id) {
  const team = state.format === "2v2";
  if (id === "chase") return team ? new TeamChaseBot(TEAM_CFG) : new ChaseBot();
  if (id === "random") return new RandomBot();
  return state.opponents[id];
}
const shortName = (id) => (id === "chase" ? "Chase bot" : id === "random" ? "Random" : state.info[id]?.short || id);
const noteFor = (id) =>
  id === "chase"
    ? state.format === "2v2"
      ? "Scripted rules: whoever is nearer the ball chases it and shoots, the other guards the goal."
      : "Scripted rules: run behind the ball, line up with the goal, then kick."
  : id === "random" ? "Picks a random action every step."
  : state.info[id]?.note || "";

// Fill the player pickers with the agents trained for the current format.
function fillPickers() {
  const def = state.format === "2v2" ? state.defaults.team : state.defaults.solo;
  for (const s of [$("opponent"), $("left-player")]) {
    const keep = s.value;
    s.length = 0;
    for (const [id, m] of Object.entries(state.info)) {
      if ((m.format || "1v1") === state.format && !m.hidden) s.add(new Option(m.label, id));
    }
    s.add(new Option(state.format === "2v2" ? "Scripted chase team" : "Scripted chase bot", "chase"));
    s.add(new Option(state.format === "2v2" ? "Random team" : "Random player", "random"));
    const ok = [...s.options].some((o) => o.value === keep);
    s.value = ok ? keep : s.id === "opponent" ? def : "chase";
  }
}

function applySettings() {
  const format = $("format").value;
  if (format !== state.format || !state.game) {
    state.format = format;
    state.game = new Game(format === "2v2" ? TEAM_CFG : CFG);
    fillPickers();
  }
  const team = format === "2v2";
  const oppId = $("opponent").value;
  state.agent = playerFor(oppId);
  state.watch = $("mode").value === "watch";
  state.left = state.watch ? playerFor($("left-player").value) : null;
  state.mate = team ? state.opponents[state.defaults.mate] : null;
  state.greedy = $("greedy").checked;
  const opp = shortName(oppId);
  state.names = [state.watch ? shortName($("left-player").value) : team ? "You + AI" : "You", opp];
  state.tags = team ? [state.watch ? "" : "You", state.watch ? "" : "AI mate", "", ""] : [state.watch ? state.names[0] : "You", opp];
  $("mode").options[0].text = team ? "You + AI mate vs agents" : "You vs agent";
  $("mode").options[1].text = team ? "Watch two teams" : "Watch two players";
  $("left-player-wrap").hidden = !state.watch;
  $("left-player-wrap").querySelector("label").textContent = team ? "Left team (red)" : "Left player (red)";
  $("opponent").previousElementSibling.textContent = team ? "Opponent team" : "Opponent";
  $("opponent-note").textContent = noteFor(oppId);
  $("brain").hidden = team;
  $("coach").hidden = !team;
  canvas.setAttribute("aria-label", team
    ? "Football pitch, 2v2. You and your AI teammate are red and attack to the right. Move with W A S D or the arrow keys, kick with Space."
    : "Football pitch. You are red and attack to the right. Move with W A S D or the arrow keys, kick with Space.");
  brain.live = false;
  brain.moves.fill(0);
  brain.kick = brain.conf = 0;
  brain.picked = -1;
  $("brain").dataset.mode = state.agent.probs ? "on" : "off";
  paintBrain(true);
  updateRecord();
}

function onSettingsChange() {
  applySettings();
  resetToReady();
}

$("speed").addEventListener("input", (e) => {
  state.fps = Number(e.target.value);
  $("speed-val").textContent = `${+(state.fps / 30).toFixed(2)}×`;
});
$("speed").addEventListener("pointerup", focusPitch);
["format", "opponent", "mode", "left-player", "greedy"].forEach((id) => $(id).addEventListener("change", onSettingsChange));
document.querySelectorAll("#first-to button").forEach((btn) =>
  btn.addEventListener("click", () => {
    state.firstTo = Number(btn.dataset.n);
    document.querySelectorAll("#first-to button").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
    onSettingsChange();
  })
);

$("ov-primary").addEventListener("click", () => (state.phase === "paused" ? togglePause() : startMatch()));
$("ov-secondary").addEventListener("click", startMatch);
$("btn-pause").addEventListener("click", () => { togglePause(); });
$("btn-restart").addEventListener("click", startMatch);

const fsTarget = $("match");
if (document.fullscreenEnabled && fsTarget.requestFullscreen) {
  $("btn-fs").hidden = false;
  $("btn-fs").addEventListener("click", () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else fsTarget.requestFullscreen().then(resize).catch(() => {});
  });
  document.addEventListener("fullscreenchange", () => {
    $("btn-fs").textContent = document.fullscreenElement ? "Exit full screen" : "Full screen";
    requestAnimationFrame(resize);
  });
}

// ---------------------------------------------------------------------- coach
// Presets work anywhere. Typed tactics need scripts/serve_demo.py, which runs the LLM coach locally.
function setTactic(z, label) {
  state.z = Float32Array.from(z);
  document.querySelectorAll("#presets button").forEach((b) => b.setAttribute("aria-pressed", String(b.textContent === label)));
  paintTactic();
}

function paintTactic() {
  TACTICS.forEach((t, i) => {
    const v = state.z[i];
    const bar = $(`tac-${t.id}`);
    bar.style.left = `${50 + Math.min(v, 0) * 50}%`;
    bar.style.width = `${Math.abs(v) * 50}%`;
    $(`tac-${t.id}-val`).textContent = v === 0 ? "0" : `${v > 0 ? "+" : ""}${v.toFixed(1)}`;
  });
}

function setupCoach() {
  $("tactic-bars").innerHTML = TACTICS.map((t) => `
    <div>
      <dt>${t.label}</dt>
      <dd><span class="meter bi"><i id="tac-${t.id}"></i></span><output id="tac-${t.id}-val">0</output></dd>
      <span class="ends" aria-hidden="true"><span>${t.lo}</span><span>${t.hi}</span></span>
    </div>`).join("");
  for (const p of PRESETS) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = p.label;
    b.addEventListener("click", () => {
      setTactic(p.z, p.label);
      say(`Tactic for your teammate: ${p.label.toLowerCase()}.`);
    });
    $("presets").append(b);
  }
  setTactic(PRESETS[0].z, PRESETS[0].label);
  $("coach-note").textContent = "Typing a tactic needs the local coach server (python scripts/serve_demo.py). The presets work anywhere.";

  fetch("coach/health").then((r) => (r.ok ? r.json() : null)).then((h) => {
    if (!h) return;
    $("coach-form").hidden = false;
    $("coach-note").textContent = `Coach model: ${h.model}, running on this machine.`;
  }).catch(() => {});

  $("coach-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = $("coach-text").value.trim();
    if (!text) return;
    $("coach-send").disabled = true;
    say("Asking the coach…");
    try {
      const res = await fetch("coach", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
      if (!res.ok) throw new Error(`coach answered ${res.status}`);
      const out = await res.json();
      setTactic(TACTICS.map((t) => out.z[t.id]), "");
      const how = out.source === "llm" ? `${out.model}, ${out.latency_s.toFixed(1)} s` : "matched keywords";
      say(`Coach heard "${text}" (${how}).`);
    } catch (err) {
      say(`The coach did not answer: ${err.message}. Use a preset instead.`);
    } finally {
      $("coach-send").disabled = false;
      focusPitch();
    }
  });
}

// ----------------------------------------------------------------------- init
async function init() {
  const manifest = await (await fetch("models/manifest.json")).json();
  const policies = await Promise.all(manifest.models.map((m) => Policy.load(`models/${m.id}.json`)));
  manifest.models.forEach((m, i) => {
    state.opponents[m.id] = policies[i];
    state.info[m.id] = m;
  });
  state.defaults = { solo: manifest.default, team: manifest.default_2v2, mate: manifest.teammate };
  state.format = "";
  if (!state.defaults.mate) $("format").disabled = true;
  setupCoach();
  resize();
  applySettings();
  resetToReady();
  requestAnimationFrame(frame);
}

init().catch((err) => {
  console.error(err);
  say(
    location.protocol === "file:"
      ? "The agents could not load from a file. Run `python -m http.server` in the docs folder and open localhost:8000."
      : `Could not load the agents: ${err.message}`
  );
});
