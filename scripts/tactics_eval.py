"""Does the tactic-conditioned agent follow the tactic it is given?

1. Sweep: for each tactic dimension, set it to -1, -0.5, 0, 0.5, 1 (others 0)
   and measure that dimension's behavior metric and the win rate vs the
   chase team. The unconditioned pool agent is the reference.
2. End to end: phrases go through the coach (keywords and each LLM), the
   resulting tactic drives the agent, and we check that every dimension the
   phrase asks for moves the matching metric in the right direction compared
   with the neutral tactic.
3. Position heatmaps under contrasting tactics.

Example:
    python scripts/tactics_eval.py runs/2v2_cond --reference runs/2v2_pool
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

from football.bots import TeamChaseBot  # noqa: E402
from football.coach import Coach, z_vector  # noqa: E402
from football.config import TEAM_CONFIG  # noqa: E402
from football.policy import load_checkpoint, load_player, PolicyPlayer  # noqa: E402
from football.stats import team_stats  # noqa: E402
from football.tactics import K, TACTICS  # noqa: E402

LEVELS = (-1.0, -0.5, 0.0, 0.5, 1.0)
PHRASES = [
    ("press high", {"press": 1}),
    ("park the bus", {"press": -1}),
    ("attack down the left", {"press": 1, "flank": 1}),
    ("overload the right side", {"flank": -1}),
    ("spread out", {"spread": 1}),
    ("stay tight as a pair", {"spread": -1}),
    ("hold possession", {"possession": 1}),
    ("let them come at us and break fast", {"press": -1, "possession": -1}),
]
HEAT_TACTICS = {
    "neutral": {},
    "press high (press +1)": {"press": 1.0},
    "park the bus (press -1)": {"press": -1.0},
    "left wing (flank +1)": {"flank": 1.0},
}


def vec(d: dict) -> np.ndarray:
    return np.array([d.get(k, 0.0) for k in TACTICS], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--reference", default=None, help="unconditioned run to compare with")
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--models", nargs="*", default=["qwen2.5:3b-instruct"])
    ap.add_argument("--out", default="results/2v2")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    c = TEAM_CONFIG
    chase = TeamChaseBot(c)
    net, _ = load_checkpoint(Path(args.run_dir) / "final.pt")

    def measure(z, seed=31):
        torch.manual_seed(seed)  # same z must give the same games, so coaches differ only by their z
        return team_stats(PolicyPlayer(net, z=z), chase, args.games, seed=seed, config=c)

    # 1. sweep
    sweep = {}
    for k, dim in enumerate(TACTICS):
        rows = []
        for v in LEVELS:
            z = np.zeros(K, dtype=np.float32)
            z[k] = v
            s = measure(z)
            rows.append({"z": v, "metric": s["tactic"][dim], "all_metrics": s["tactic"], "win": s["win"],
                         "loss": s["loss"], "score": s["score"]})
        sweep[dim] = rows
        lo, hi = rows[0]["metric"]["mean"], rows[-1]["metric"]["mean"]
        print(f"{dim:>10s}: metric {lo:+.3f} at z=-1 -> {hi:+.3f} at z=+1   "
              f"win vs chase {rows[0]['win']:.2f} / {rows[2]['win']:.2f} / {rows[-1]['win']:.2f}")
    reference = None
    if args.reference:
        r = team_stats(load_player(Path(args.reference) / "final.pt"), chase, args.games, seed=31, config=c)
        reference = {"run": Path(args.reference).name, "tactic": r["tactic"], "win": r["win"], "score": r["score"]}

    # 2. end to end through the coach
    coaches = {"keywords": Coach(use_llm=False)}
    for m in args.models:
        coaches[f"keywords+{m}"] = Coach(model=m, log_path=out / "coach_log.jsonl")
    neutral = measure(np.zeros(K, dtype=np.float32), seed=41)["tactic"]
    e2e = {}
    for name, coach in coaches.items():
        rows, hits, total = [], 0, 0
        for text, want in PHRASES:
            z = coach(text)["z"]
            s = measure(np.array(z_vector(z), dtype=np.float32), seed=41)
            moved = {}
            for dim, sign in want.items():
                delta = s["tactic"][dim]["mean"] - neutral[dim]["mean"]
                ok = bool(np.sign(delta) == sign and abs(delta) > s["tactic"][dim]["ci95"])
                moved[dim] = {"delta": delta, "ok": ok}
                hits += ok
                total += 1
            rows.append({"text": text, "z": z, "moved": moved, "win": s["win"]})
        e2e[name] = {"followed": hits / total, "rows": rows}
        print(f"end to end via {name}: {hits}/{total} requested dimensions moved the right way")

    # 3. heatmaps
    heat = {}
    for label, d in HEAT_TACTICS.items():
        heat[label] = measure(vec(d), seed=51)["heatmap"]

    (out / "tactics.json").write_text(json.dumps({
        "games": args.games, "sweep": sweep, "reference": reference, "neutral": neutral, "end_to_end": e2e,
    }, indent=2))
    (out / "heatmaps_tactics.json").write_text(json.dumps(heat))

    fig, axes = plt.subplots(1, K, figsize=(3.2 * K, 3.2))
    for ax, dim in zip(axes, TACTICS):
        rows = sweep[dim]
        m = np.array([r["metric"]["mean"] for r in rows])
        e = np.array([r["metric"]["ci95"] for r in rows])
        ax.errorbar(LEVELS, m, yerr=e, fmt="o-", capsize=3, label="conditioned agent")
        if reference:
            ax.axhline(reference["tactic"][dim]["mean"], color="gray", ls="--", lw=1, label="unconditioned agent")
        ax.set_title(dim)
        ax.set_xlabel(f"tactic z[{dim}]")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("behavior metric (game mean)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Tactic sweep: each dimension alone, vs the chase team", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "tactic_sweep.png", dpi=150, bbox_inches="tight")

    fig, axes = plt.subplots(1, len(heat), figsize=(3.6 * len(heat), 2.7))
    for ax, (label, h) in zip(axes, heat.items()):
        ax.imshow(np.array(h), origin="lower", extent=[0, c.field_l, 0, c.field_h], cmap="magma", aspect="equal")
        ax.set_title(label, fontsize=9)
        ax.set_xticks([0, 50, 100])
        ax.set_yticks([0, 32, 64])
        ax.axvline(50, color="white", lw=0.5, alpha=0.5)
    fig.suptitle("Where the conditioned team stands (attacking right)", y=1.03)
    fig.tight_layout()
    fig.savefig(out / "tactic_heatmaps.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out}/tactics.json, tactic_sweep.png, tactic_heatmaps.png")


if __name__ == "__main__":
    main()
