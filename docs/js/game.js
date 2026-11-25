// JavaScript port of football/core.py for a single game (1v1 or 2v2).
// Must stay in sync with the Python physics (tests/test_web_parity.py checks it).
// Players are team-major: 0..P-1 are team 0, P..2P-1 are team 1.

export const CFG = {
  field_l: 100.0,
  field_h: 64.0,
  goal_w: 20.0,
  player_r: 2.0,
  player_accel: 0.35,
  player_damping: 0.85,
  player_max_speed: 2.0,
  ball_r: 1.2,
  ball_friction: 0.97,
  ball_wall_restitution: 0.8,
  ball_player_restitution: 0.3,
  kick_speed: 4.5,
  kick_reach: 1.5,
  kick_cooldown: 8,
  max_steps: 400,
  kickoff_noise: 2.0,
  players_per_team: 1,
};
CFG.kick_range = CFG.player_r + CFG.ball_r + CFG.kick_reach;

export const TEAM_CFG = { ...CFG, players_per_team: 2 };

export const N_MOVES = 9;
export const N_ACTIONS = 18;
export const OBS_DIM = 16;
export const obsDim = (cfg) => OBS_DIM + 8 * (cfg.players_per_team - 1);

const KICKOFF = { 1: [[0.25, 0.5]], 2: [[0.35, 0.35], [0.15, 0.65]] };

const S = Math.SQRT1_2;
export const MOVE_DIRS = [
  [0, 0], [1, 0], [S, S], [0, 1], [-S, S], [-1, 0], [-S, -S], [0, -1], [S, -S],
];

const uniform = (rng) => rng() * 2 - 1;

export class Game {
  constructor(cfg = CFG, rng = Math.random) {
    this.cfg = cfg;
    this.rng = rng;
    this.P = cfg.players_per_team;
    this.M = 2 * this.P;
    this.reset();
  }

  teamOf(i) { return i < this.P ? 0 : 1; }

  reset() {
    const c = this.cfg;
    const n = c.kickoff_noise;
    const r = this.rng;
    const spots = KICKOFF[this.P];
    this.pos = [];
    for (let i = 0; i < this.M; i++) {
      const [fx, fy] = spots[i % this.P];
      const x = (i < this.P ? fx : 1 - fx) * c.field_l;
      this.pos.push([x + uniform(r) * n, fy * c.field_h + uniform(r) * n]);
    }
    this.vel = this.pos.map(() => [0, 0]);
    this.ball = [0.5 * c.field_l + uniform(r) * 0.5, 0.5 * c.field_h + uniform(r) * 0.5];
    this.ballVel = [0, 0];
    this.cooldown = this.pos.map(() => 0);
    this.t = 0;
  }

  // actions: one per player in 0..17. Returns {goal, done, truncated, touch, kick};
  // goal is +1 when team 0 scores.
  step(actions) {
    const c = this.cfg;
    this.touch = this.pos.map(() => false);
    this._movePlayers(actions.map((a) => a % N_MOVES));
    this._collidePlayers();
    this._ballContact();
    const kick = this._kick(actions.map((a) => a >= N_MOVES));
    const goal = this._moveBall();
    this.t += 1;
    const truncated = this.t >= c.max_steps && goal === 0;
    return { goal, done: goal !== 0 || truncated, truncated, touch: this.touch, kick };
  }

  _movePlayers(moves) {
    const c = this.cfg;
    const r = c.player_r;
    const hi = [c.field_l - r, c.field_h - r];
    for (let i = 0; i < this.M; i++) {
      const d = MOVE_DIRS[moves[i]];
      const dx = i >= this.P ? -d[0] : d[0];
      const v = this.vel[i];
      v[0] = v[0] * c.player_damping + dx * c.player_accel;
      v[1] = v[1] * c.player_damping + d[1] * c.player_accel;
      const speed = Math.hypot(v[0], v[1]);
      const k = Math.min(1, c.player_max_speed / Math.max(speed, 1e-8));
      v[0] *= k;
      v[1] *= k;
      const p = this.pos[i];
      p[0] += v[0];
      p[1] += v[1];
      for (let ax = 0; ax < 2; ax++) {
        if (p[ax] < r || p[ax] > hi[ax]) {
          p[ax] = Math.min(Math.max(p[ax], r), hi[ax]);
          v[ax] = 0;
        }
      }
    }
  }

  _collidePlayers() {
    const r = this.cfg.player_r;
    for (let i = 0; i < this.M; i++) {
      for (let j = i + 1; j < this.M; j++) {
        const a = this.pos[i], b = this.pos[j];
        const dx = b[0] - a[0];
        const dy = b[1] - a[1];
        const dist = Math.hypot(dx, dy);
        const overlap = 2 * r - dist;
        if (overlap <= 0) continue;
        let nx = 1, ny = 0;
        if (dist > 1e-8) { nx = dx / dist; ny = dy / dist; }
        const push = 0.5 * overlap;
        a[0] -= nx * push; a[1] -= ny * push;
        b[0] += nx * push; b[1] += ny * push;
      }
    }
  }

  _ballContact() {
    const c = this.cfg;
    const minD = c.player_r + c.ball_r;
    const e = c.ball_player_restitution;
    for (let k = 0; k < this.M; k++) {
      const who = (this.t + k) % this.M;
      const p = this.pos[who];
      const v = this.vel[who];
      const rx = this.ball[0] - p[0];
      const ry = this.ball[1] - p[1];
      const dist = Math.hypot(rx, ry);
      if (dist >= minD) continue;
      this.touch[who] = true;
      let nx = 1, ny = 0;
      if (dist >= 1e-8) { nx = rx / dist; ny = ry / dist; }
      this.ball[0] = p[0] + nx * minD;
      this.ball[1] = p[1] + ny * minD;
      const vn = (this.ballVel[0] - v[0]) * nx + (this.ballVel[1] - v[1]) * ny;
      if (vn < 0) {
        this.ballVel[0] -= (1 + e) * vn * nx;
        this.ballVel[1] -= (1 + e) * vn * ny;
      }
    }
  }

  _kick(kicks) {
    const c = this.cfg;
    let ix = 0, iy = 0, kicked = false;
    const can = this.pos.map(() => false);
    for (let i = 0; i < this.M; i++) {
      this.cooldown[i] = Math.max(this.cooldown[i] - 1, 0);
      const rx = this.ball[0] - this.pos[i][0];
      const ry = this.ball[1] - this.pos[i][1];
      const dist = Math.hypot(rx, ry);
      can[i] = kicks[i] && this.cooldown[i] === 0 && dist <= c.kick_range;
      if (can[i]) {
        const m = Math.max(dist, 1e-8);
        ix += (rx / m) * c.kick_speed;
        iy += (ry / m) * c.kick_speed;
        kicked = true;
        this.touch[i] = true;
      }
    }
    if (!kicked) return can;
    this.ballVel = [ix, iy];
    for (let i = 0; i < this.M; i++) if (can[i]) this.cooldown[i] = c.kick_cooldown;
    return can;
  }

  _moveBall() {
    const c = this.cfg;
    const b = this.ball;
    const bv = this.ballVel;
    b[0] += bv[0];
    b[1] += bv[1];
    bv[0] *= c.ball_friction;
    bv[1] *= c.ball_friction;

    const r = c.ball_r;
    const rest = c.ball_wall_restitution;
    if (b[1] < r) { b[1] = 2 * r - b[1]; bv[1] *= -rest; }
    else if (b[1] > c.field_h - r) { b[1] = 2 * (c.field_h - r) - b[1]; bv[1] *= -rest; }

    const inMouth = Math.abs(b[1] - 0.5 * c.field_h) < 0.5 * c.goal_w;
    if (!inMouth) {
      if (b[0] < r) { b[0] = 2 * r - b[0]; bv[0] *= -rest; }
      else if (b[0] > c.field_l - r) { b[0] = 2 * (c.field_l - r) - b[0]; bv[0] *= -rest; }
    }
    if (inMouth && b[0] > c.field_l) return 1;
    if (inMouth && b[0] < 0) return -1;
    return 0;
  }

  // Ego-centric observation for player i (team 1 is mirrored along x); layout in core.py.
  observe(i) {
    const c = this.cfg;
    const hl = 0.5 * c.field_l;
    const hh = 0.5 * c.field_h;
    const team = this.teamOf(i);
    const sx = team === 0 ? 1 : -1;
    const P = (p) => [((p[0] - hl) / hl) * sx, (p[1] - hh) / hh];
    const V = (v) => [(v[0] / c.player_max_speed) * sx, v[1] / c.player_max_speed];
    const me = this.pos[i];
    const opp = [];
    for (let j = 0; j < this.M; j++) if (this.teamOf(j) !== team) opp.push(j);
    const d = (j) => Math.hypot(this.pos[j][0] - me[0], this.pos[j][1] - me[1]);
    opp.sort((a, b) => d(a) - d(b) || a - b);
    const mates = [];
    for (let j = team * this.P; j < (team + 1) * this.P; j++) if (j !== i) mates.push(j);

    const o = new Float32Array(obsDim(c));
    const mp = P(me), ball = P(this.ball);
    const put = (k, xy) => { o[k] = xy[0]; o[k + 1] = xy[1]; };
    put(0, mp); put(2, V(this.vel[i]));
    put(4, P(this.pos[opp[0]])); put(6, V(this.vel[opp[0]]));
    put(8, ball);
    o[10] = (this.ballVel[0] / c.kick_speed) * sx; o[11] = this.ballVel[1] / c.kick_speed;
    o[12] = ball[0] - mp[0]; o[13] = ball[1] - mp[1];
    o[14] = this.cooldown[i] / c.kick_cooldown;
    o[15] = this.t / c.max_steps;
    let k = 16;
    for (const j of [...mates, ...opp.slice(1)]) {
      put(k, P(this.pos[j])); put(k + 2, V(this.vel[j]));
      k += 4;
    }
    return o;
  }

  // Turn screen-space input (dx right, dy up) into an action for a player on team `side`.
  static keyboardAction(dx, dy, kick, side) {
    if (side === 1) dx = -dx;
    let move = 0;
    if (dx || dy) {
      let best = -Infinity;
      for (let m = 1; m < N_MOVES; m++) {
        const s = MOVE_DIRS[m][0] * dx + MOVE_DIRS[m][1] * dy;
        if (s > best) { best = s; move = m; }
      }
    }
    return move + (kick ? N_MOVES : 0);
  }
}
