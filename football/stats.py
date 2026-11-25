"""Behavior statistics for 2v2 matches: tactics, passing, roles, positions.

Everything is measured for team A (the first team passed to `team_stats`),
in A's ego frame (attacking towards +x), only while a game is still running.

- ``tactic``: per-game means of the `football.tactics` features.
- ``passes``: A kicks the ball and the next player to touch it is A's other
  player. ``intercepted``: the next touch is an opponent's.
- ``depth_gap``: mean |x difference| between A's two players, field units.
- ``role_persistence``: share of steps where the deeper player is the slot
  that is deeper most often in that game (1.0 = fixed defender/attacker).
- ``swarm``: share of steps where both A players are within 8 units of the ball.
- ``heatmap``: (10, 16) histogram of A player positions, rows bottom to top.
"""

from __future__ import annotations

import numpy as np

from .config import TEAM_CONFIG, GameConfig
from .evaluate import run_games
from .tactics import K, TACTICS, team_features

SWARM_R = 8.0
HEAT_BINS = (16, 10)


def team_stats(team_a, team_b, n_games: int = 500, seed: int = 0, config: GameConfig = TEAM_CONFIG) -> dict:
    c = config
    n = n_games
    rows = np.arange(n)
    side = rows % 2
    feat_sum = np.zeros((n, K))
    steps = np.zeros(n)
    deeper0 = np.zeros(n)
    gap_sum = np.zeros(n)
    swarm = np.zeros(n)
    kicks = np.zeros(n)
    passes = np.zeros(n)
    intercepted = np.zeros(n)
    last_kicker = np.full(n, -1)
    heat = np.zeros(HEAT_BINS[::-1])

    def on_step(env, a_idx, info, alive):
        live = alive
        f = team_features(env)[rows, side]
        feat_sum[live] += f[live]
        steps[live] += 1

        pos = env.pos[rows[:, None], a_idx]
        x = np.where(side[:, None] == 0, pos[..., 0], c.field_l - pos[..., 0])
        y = pos[..., 1]
        gap_sum[live] += np.abs(x[:, 0] - x[:, 1])[live]
        deeper0[live] += (x[:, 0] < x[:, 1])[live]
        near = np.linalg.norm(pos - env.ball[:, None, :], axis=-1) < SWARM_R
        swarm[live] += near.all(axis=1)[live]
        h, _, _ = np.histogram2d(
            y[live].ravel(), x[live].ravel(), bins=HEAT_BINS[::-1], range=[[0, c.field_h], [0, c.field_l]]
        )
        heat[:] += h

        touch, kick = info["touch"], info["kick"]
        mine = np.zeros_like(touch)
        mine[rows[:, None], a_idx] = True
        a_touch = (touch & mine).any(axis=1)
        b_touch = (touch & ~mine).any(axis=1)
        pending = last_kicker >= 0
        other_a = (touch & mine).copy()
        other_a[rows[pending], last_kicker[pending]] = False
        passes[live & pending & other_a.any(axis=1)] += 1
        intercepted[live & pending & b_touch] += 1
        a_kick = (kick & mine).any(axis=1)
        kicks[live & a_kick] += 1
        last_kicker[a_touch | b_touch] = -1
        kicker = np.argmax(kick & mine, axis=1)
        last_kicker[a_kick] = kicker[a_kick]

    result = run_games(team_a, team_b, n_games, seed, config, on_step=on_step)
    s = np.maximum(steps, 1)
    tactic = feat_sum / s[:, None]
    persist = np.maximum(deeper0, steps - deeper0) / s
    return {
        "games": n,
        "win": float(np.mean(result == 1)),
        "draw": float(np.mean(result == 0)),
        "loss": float(np.mean(result == -1)),
        "score": float(np.mean((result + 1) / 2)),
        "ep_len": float(steps.mean()),
        "tactic": {k: _mean_ci(tactic[:, i]) for i, k in enumerate(TACTICS)},
        "kicks_per_game": float(kicks.mean()),
        "passes_per_game": float(passes.mean()),
        "intercepted_per_game": float(intercepted.mean()),
        "pass_share_of_kicks": float(passes.sum() / max(kicks.sum(), 1)),
        "depth_gap": _mean_ci(gap_sum / s),
        "role_persistence": _mean_ci(persist),
        "swarm": _mean_ci(swarm / s),
        "heatmap": (heat / max(heat.sum(), 1)).tolist(),
    }


def _mean_ci(v: np.ndarray) -> dict:
    """Mean with a 95% normal-approximation confidence half-width."""
    v = np.asarray(v, dtype=float)
    return {"mean": float(v.mean()), "ci95": float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))}
