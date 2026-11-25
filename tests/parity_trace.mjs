// Replays a fixed state + action sequence through docs/js/game.js and prints
// the trajectory as JSON. Used by tests/test_web_parity.py.
import { readFileSync } from "node:fs";
import { CFG, Game } from "../docs/js/game.js";
import { ChaseBot, TeamChaseBot } from "../docs/js/policy.js";

const input = JSON.parse(readFileSync(process.argv[2], "utf8"));
const cfg = { ...CFG, players_per_team: input.players_per_team };
const game = new Game(cfg);
game.pos = input.pos;
game.vel = input.vel;
game.ball = input.ball;
game.ballVel = input.ball_vel;
game.cooldown = input.cooldown;
game.t = 0;

const bot = cfg.players_per_team === 1 ? new ChaseBot() : new TeamChaseBot();
const trace = [];
for (const a of input.actions) {
  const chase = bot.act(game.observe(0));
  const r = game.step(a);
  trace.push({
    pos: game.pos.map((p) => [...p]),
    ball: [...game.ball],
    goal: r.goal,
    touch: r.touch,
    obs: Array.from({ length: game.M }, (_, i) => Array.from(game.observe(i))),
    chase,
  });
  if (r.done) break;
}
console.log(JSON.stringify(trace));
