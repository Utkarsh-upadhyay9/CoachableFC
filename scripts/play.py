"""Play against a trained agent with the keyboard (or watch two players).

Controls: arrow keys / WASD to move, Space to kick, +/- to change speed,
P to pause, R to restart the match, Esc to quit. In --team mode, T cycles
your AI teammate through preset tactics.

Examples:
    python scripts/play.py --agent models/final.pt
    python scripts/play.py --agent models/final.pt --fps 15 --goals 3
    python scripts/play.py --watch --agent models/final.pt --opponent chase
    python scripts/play.py --team --agent models/2v2/pool.pt --mate models/2v2/coach.pt
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from football.config import DEFAULT_CONFIG, N_MOVES, TEAM_CONFIG  # noqa: E402
from football.core import MOVE_DIRS, FootballCore  # noqa: E402
from football.evaluate import team_act  # noqa: E402
from football.players import make_player  # noqa: E402
from football.policy import PolicyPlayer  # noqa: E402
from football.render import Renderer  # noqa: E402

# Same presets as the web demo; order is press, flank, spread, possession.
PRESETS = [
    ("neutral", [0, 0, 0, 0]),
    ("press high", [1, 0, 0, 0]),
    ("park the bus", [-1, 0, 0, 0]),
    ("left wing", [0, 1, 0, 0]),
    ("right wing", [0, -1, 0, 0]),
    ("spread out", [0, 0, 1, 0]),
    ("stick together", [0, 0, -1, 0]),
    ("keep the ball", [0, 0, 0, 1]),
]


def keyboard_action(pg, side: int) -> int:
    keys = pg.key.get_pressed()
    dx = (keys[pg.K_RIGHT] or keys[pg.K_d]) - (keys[pg.K_LEFT] or keys[pg.K_a])
    dy = (keys[pg.K_UP] or keys[pg.K_w]) - (keys[pg.K_DOWN] or keys[pg.K_s])
    if side == 1:
        dx = -dx
    move = 0
    if dx or dy:
        v = np.array([dx, dy], dtype=float)
        move = int(np.argmax(MOVE_DIRS[1:] @ v)) + 1
    kick = bool(keys[pg.K_SPACE])
    return move + N_MOVES * kick


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="models/final.pt", help="checkpoint, 'chase' or 'random'")
    ap.add_argument("--opponent", default=None, help="second player for --watch mode")
    ap.add_argument("--watch", action="store_true", help="agent vs --opponent, no human")
    ap.add_argument("--team", action="store_true", help="2v2: you and --mate against two copies of --agent")
    ap.add_argument("--mate", default="models/2v2/coach.pt", help="your teammate in --team mode")
    ap.add_argument("--side", type=int, default=0, choices=[0, 1], help="0 = you are red (left)")
    ap.add_argument("--fps", type=int, default=30, help="simulation steps per second (lower = easier)")
    ap.add_argument("--goals", type=int, default=5, help="first to this many goals wins")
    ap.add_argument("--greedy", action="store_true", help="agent picks argmax actions")
    ap.add_argument("--log", default="results/human_games.csv")
    args = ap.parse_args()

    cfg = TEAM_CONFIG if args.team else DEFAULT_CONFIG
    P = cfg.players_per_team
    agent = make_player(args.agent, greedy=args.greedy, config=cfg)
    other = make_player(args.opponent, greedy=args.greedy, config=cfg) if args.watch else None
    mate = make_player(args.mate, greedy=args.greedy, config=cfg) if args.team and not args.watch else None
    preset = 0
    human_side = args.side
    agent_side = 1 - human_side
    hs, ags = slice(human_side * P, human_side * P + P), slice(agent_side * P, agent_side * P + P)

    def set_preset(i):
        if isinstance(mate, PolicyPlayer):
            mate.z = np.asarray(PRESETS[i][1], dtype=np.float32)

    set_preset(preset)
    renderer = Renderer(cfg, scale=8.0)
    pg = renderer.pg
    clock = pg.time.Clock()
    core = FootballCore(1, cfg, seed=int(time.time()), auto_reset=False)
    obs = core.reset()

    score = [0, 0]  # [human/left player, agent]
    fps = args.fps
    paused = False
    over = False
    left_name = Path(args.opponent).stem if args.watch else ("You + mate" if args.team else "You")
    right_name = Path(args.agent).stem

    running = True
    while running:
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                running = False
            elif ev.type == pg.KEYDOWN:
                if ev.key == pg.K_ESCAPE:
                    running = False
                elif ev.key in (pg.K_PLUS, pg.K_EQUALS, pg.K_KP_PLUS):
                    fps = min(fps + 5, 120)
                elif ev.key in (pg.K_MINUS, pg.K_KP_MINUS):
                    fps = max(fps - 5, 5)
                elif ev.key == pg.K_p:
                    paused = not paused
                elif ev.key == pg.K_t and mate is not None:
                    preset = (preset + 1) % len(PRESETS)
                    set_preset(preset)
                elif ev.key == pg.K_r:
                    score, over = [0, 0], False
                    obs = core.reset()

        if not paused and not over:
            actions = np.zeros((1, 2 * P), dtype=np.int64)
            actions[:, ags] = team_act(agent, obs[:, ags])
            if other is not None:
                actions[:, hs] = team_act(other, obs[:, hs])
            else:
                actions[0, hs.start] = keyboard_action(pg, human_side)
                if mate is not None:
                    actions[0, hs.start + 1] = mate(obs[:, hs.start + 1])[0]
            obs, _, _, done, info = core.step(actions)
            if done[0]:
                goal = int(info["goal"][0])
                if goal != 0:
                    scorer_side = 0 if goal == 1 else 1
                    score[0 if scorer_side == human_side else 1] += 1
                obs = core.reset()
                if max(score) >= args.goals:
                    over = True
                    if not args.watch:
                        log_result(args, score, fps)

        hud = f"{left_name} {score[0]} - {score[1]} {right_name}   |   first to {args.goals}   |   {fps} fps"
        if mate is not None:
            hud += f"   |   mate: {PRESETS[preset][0]} (T)"
        if paused:
            hud += "   |   PAUSED"
        if over:
            hud = f"{'You win' if score[0] > score[1] else 'Agent wins'} {score[0]}-{score[1]}  (R = rematch, Esc = quit)"
        renderer.draw(core, 0, hud)
        renderer.flip()
        clock.tick(fps)

    renderer.close()


def log_result(args, score, fps):
    path = Path(args.log)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    agent = f"{args.agent} (2v2, mate {args.mate})" if args.team else args.agent
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "agent", "human_goals", "agent_goals", "human_won", "fps", "first_to"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), agent, score[0], score[1], int(score[0] > score[1]), fps, args.goals])


if __name__ == "__main__":
    main()
