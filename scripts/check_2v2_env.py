"""2v2 environment checks: side fairness, bot sanity and simulation speed.

Example:
    python scripts/check_2v2_env.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from football.bots import RandomBot, TeamChaseBot  # noqa: E402
from football.config import DEFAULT_CONFIG, N_ACTIONS, TEAM_CONFIG  # noqa: E402
from football.core import FootballCore  # noqa: E402
from football.evaluate import run_games  # noqa: E402


def by_side(a, b, n, seed):
    """Win rates of the team on side 0 and side 1 (A plays side 0 in even games)."""
    r = run_games(a, b, n, seed, TEAM_CONFIG)
    side0 = np.where(np.arange(n) % 2 == 0, r, -r)
    return {"side0_win": float((side0 == 1).mean()), "side1_win": float((side0 == -1).mean()), "draw": float((side0 == 0).mean())}


def speed(config, n_envs=256, steps=2000):
    core = FootballCore(n_envs, config, seed=0)
    core.reset()
    rng = np.random.default_rng(0)
    acts = rng.integers(0, N_ACTIONS, (steps, n_envs, core.M))
    t0 = time.perf_counter()
    for a in acts:
        core.step(a)
    return n_envs * steps / (time.perf_counter() - t0)


def main():
    chase = TeamChaseBot(TEAM_CONFIG)
    out = {
        "chase_vs_chase_4000": by_side(chase, chase, 4000, 0),
        "random_vs_random_1000": by_side(RandomBot(seed=1), RandomBot(seed=2), 1000, 1),
        "chase_vs_random_1000": {k: float((run_games(chase, RandomBot(seed=3), 1000, 2, TEAM_CONFIG) == v).mean())
                                 for k, v in (("win", 1), ("draw", 0), ("loss", -1))},
        "env_steps_per_s": {"1v1": speed(DEFAULT_CONFIG), "2v2": speed(TEAM_CONFIG)},
    }
    Path("results/2v2").mkdir(parents=True, exist_ok=True)
    Path("results/2v2/env_checks.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
