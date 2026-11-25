"""2v2 baselines, coordination metrics and head-to-head for final agents.

For each run's final.pt (tactic-conditioned agents play the neutral tactic):

- W/D/L vs the random team and the chase team.
- Ad-hoc partner: the agent teams up with a scripted chase player (a partner
  it never trained with) against the chase team.
- Coordination stats vs the chase team (`football.stats`), with the chase
  team itself as a reference.
- Head-to-head round robin between the final agents.

Example:
    python scripts/eval_2v2.py runs/2v2_pool runs/2v2_pool_seed1 runs/2v2_latest runs/2v2_cond
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from football.bots import RandomBot, TeamChaseBot  # noqa: E402
from football.config import TEAM_CONFIG  # noqa: E402
from football.evaluate import crossplay, play_matches  # noqa: E402
from football.policy import load_player  # noqa: E402
from football.stats import team_stats  # noqa: E402


def wdl(r) -> dict:
    return {"win": r.wins / r.games, "draw": r.draws / r.games, "loss": r.losses / r.games}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--out", default="results/2v2")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    c = TEAM_CONFIG
    chase = TeamChaseBot(c)

    names = [Path(r).name for r in args.runs]
    torch.manual_seed(0)
    agents = [load_player(Path(r) / "final.pt", name=n) for r, n in zip(args.runs, names)]

    baselines, coordination, heatmaps = [], {}, {}
    ref = team_stats(chase, chase, args.games, seed=21, config=c)
    heatmaps["chase team"] = ref.pop("heatmap")
    coordination["chase team"] = ref
    for name, a in zip(names, agents):
        rr = play_matches(a, RandomBot(seed=1), args.games, seed=11, config=c)
        rc = play_matches(a, chase, args.games, seed=12, config=c)
        rm = play_matches(a, [a, chase], args.games, seed=14, config=c)
        rp = play_matches([a, chase], chase, args.games, seed=13, config=c)
        baselines.append({"run": name, "vs_random": wdl(rr), "vs_chase": wdl(rc), "vs_mixed": wdl(rm),
                          "with_chase_partner_vs_chase": wdl(rp)})
        s = team_stats(a, chase, args.games, seed=21, config=c)
        heatmaps[name] = s.pop("heatmap")
        coordination[name] = s
        print(f"{name:>15s}  vs random {rr.wins}/{rr.draws}/{rr.losses}  vs chase {rc.wins}/{rc.draws}/{rc.losses}  "
              f"vs mixed {rm.wins}/{rm.draws}/{rm.losses}  ad-hoc partner {rp.wins}/{rp.draws}/{rp.losses}  passes/game {s['passes_per_game']:.2f}  "
              f"swarm {s['swarm']['mean']:.3f}  depth gap {s['depth_gap']['mean']:.1f}")
    print(f"{'chase team':>15s}  passes/game {ref['passes_per_game']:.2f}  swarm {ref['swarm']['mean']:.3f}  "
          f"depth gap {ref['depth_gap']['mean']:.1f}")

    score, win = crossplay(agents, n_games=args.games, seed=99, config=c)
    print("\nhead-to-head score (row vs column):")
    print(" " * 17 + "".join(f"{n:>16s}" for n in names))
    for n, row in zip(names, score):
        print(f"{n:>15s}  " + "".join(f"{v:16.2f}" for v in row))

    (out / "eval.json").write_text(json.dumps({
        "games": args.games, "baselines": baselines, "coordination": coordination,
        "names": names, "head_to_head_score": score.tolist(), "head_to_head_win": np.nan_to_num(win).tolist(),
    }, indent=2))
    (out / "heatmaps_neutral.json").write_text(json.dumps(heatmaps))

    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(names))
    w = 0.2
    keys = (("vs_random", "tab:gray", "vs random team"), ("vs_chase", "tab:orange", "vs chase team"),
            ("vs_mixed", "tab:blue", "vs itself + chase player"),
            ("with_chase_partner_vs_chase", "tab:green", "with chase partner vs chase team"))
    for k, (key, color, label) in enumerate(keys):
        wins = [r[key]["win"] for r in baselines]
        draws = [r[key]["draw"] for r in baselines]
        ax.bar(x + (k - 1.5) * w, wins, w, color=color, label=f"win, {label}")
        ax.bar(x + (k - 1.5) * w, draws, w, bottom=wins, color=color, alpha=0.35)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of games (light = draws)")
    ax.set_title(f"2v2 final agents vs scripted teams ({args.games} games each)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "baselines.png", dpi=150)
    print(f"\nwrote {out}/eval.json and baselines.png")


if __name__ == "__main__":
    main()
