"""Per-run summary numbers for the 2v2 report: Elo, progress and draw collapse.

- later_beats_earlier: share of checkpoint pairs where the later checkpoint
  scores above 0.5 against the earlier one (cross-play, bots excluded).
- draw_lock_step: first env step after which the self-play draw rate
  (mean over 20 updates) never drops below 0.99 again, or null.

Example:
    python scripts/summarize_2v2.py pool pool_seed1 latest cond cond_w10
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


def main():
    out = {}
    for r in sys.argv[1:]:
        elo = json.loads(Path(f"results/2v2/{r}/elo.json").read_text())
        ck = {k: v for k, v in elo.items() if k not in ("random", "chase")}
        score = np.load(f"results/2v2/{r}/crossplay.npz")["score"][2:, 2:]
        i, j = np.tril_indices(len(score), -1)
        rows = list(csv.DictReader(Path(f"runs/2v2_{r}/metrics.csv").open()))
        steps = np.array([float(x["env_steps"]) for x in rows])[19:]
        draw = np.convolve([float(x["train_draw"]) for x in rows], np.ones(20) / 20, mode="valid")
        below = np.nonzero(draw < 0.99)[0]
        lock = None if len(below) and below[-1] == len(draw) - 1 else float(steps[below[-1] + 1 if len(below) else 0])
        peak = max(ck, key=ck.get)
        out[r] = {
            "final_elo": list(ck.values())[-1], "peak_elo": ck[peak], "peak_at": peak, "chase_elo": elo["chase"],
            "later_beats_earlier": float((score[i, j] > 0.5).mean()),
            "final_selfplay_draw": float(np.mean([float(x["train_draw"]) for x in rows[-50:]])),
            "draw_lock_step": lock,
        }
    Path("results/2v2/summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
