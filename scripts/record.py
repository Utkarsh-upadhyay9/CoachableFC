"""Record matches between two players (or 2v2 teams) to an animated GIF.

A 2v2 checkpoint makes the match 2v2 (use --players 2 for bots only). With
--tactic, a tactic-conditioned red team plays that tactic.

Examples:
    python scripts/record.py models/final.pt chase --games 3 --out media/final_vs_chase.gif
    python scripts/record.py models/2v2/cond.pt chase --tactic press=1 --out media/2v2_press.gif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from football.config import DEFAULT_CONFIG, TEAM_CONFIG  # noqa: E402
from football.core import FootballCore  # noqa: E402
from football.evaluate import team_act  # noqa: E402
from football.players import chase_for, make_player  # noqa: E402
from football.policy import PolicyPlayer  # noqa: E402
from football.render import Renderer  # noqa: E402
from football.tactics import TACTICS  # noqa: E402


def parse_tactic(s: str) -> np.ndarray:
    z = np.zeros(len(TACTICS), dtype=np.float32)
    for part in filter(None, s.split(",")):
        k, v = part.split("=")
        z[TACTICS.index(k.strip())] = float(v)
    return z


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("left", help="red player or team (checkpoint, chase or random)")
    ap.add_argument("right", help="blue player or team")
    ap.add_argument("--players", type=int, default=None, choices=[1, 2], help="players per team (default: from the checkpoints)")
    ap.add_argument("--tactic", default="", help="red team tactic, e.g. press=1,flank=-0.5")
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=2, help="keep every k-th frame")
    ap.add_argument("--scale", type=float, default=5.0)
    ap.add_argument("--out", default="media/match.gif")
    args = ap.parse_args()

    left, right = make_player(args.left, seed=args.seed), make_player(args.right, seed=args.seed + 1)
    nets = [p.net for p in (left, right) if isinstance(p, PolicyPlayer)]
    P = args.players or (2 if any(n.obs_dim > DEFAULT_CONFIG.obs_dim for n in nets) else 1)
    cfg = TEAM_CONFIG if P == 2 else DEFAULT_CONFIG
    left, right = (chase_for(cfg) if s == "chase" else p for s, p in ((args.left, left), (args.right, right)))
    if args.tactic and isinstance(left, PolicyPlayer):
        left.z = parse_tactic(args.tactic)
    names = (Path(args.left).stem + (f" [{args.tactic}]" if args.tactic else ""), Path(args.right).stem)
    core = FootballCore(1, cfg, seed=args.seed, auto_reset=False)
    renderer = Renderer(cfg, scale=args.scale, headless=True, hud_h=28)

    frames = []
    score = [0, 0]
    for game in range(args.games):
        obs = core.reset()
        t = 0
        while True:
            if t % args.every == 0:
                renderer.draw(core, 0, f"{names[0]} {score[0]} - {score[1]} {names[1]}   game {game + 1}/{args.games}")
                frames.append(Image.fromarray(renderer.rgb_array()))
            acts = np.concatenate([team_act(left, obs[:, :P]), team_act(right, obs[:, P:])], axis=1)
            obs, _, _, done, info = core.step(acts)
            t += 1
            if done[0]:
                g = int(info["goal"][0])
                if g == 1:
                    score[0] += 1
                elif g == -1:
                    score[1] += 1
                renderer.draw(core, 0, f"{names[0]} {score[0]} - {score[1]} {names[1]}   game {game + 1}/{args.games}")
                frames.extend([Image.fromarray(renderer.rgb_array())] * 10)
                break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = [f.convert("P", palette=Image.ADAPTIVE, colors=32) for f in frames]
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / 30 * args.every), loop=0, optimize=True)
    print(f"wrote {out} ({len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB), final score {score}")


if __name__ == "__main__":
    main()
