"""Tactic vectors: the numeric interface between the LLM coach and the policy.

A tactic ``z`` has one value in [-1, 1] per dimension in `TACTICS`. A
conditioned policy sees ``z`` appended to its observation. During training
``z`` is sampled per episode and each dimension adds a small per-step reward
``TACTIC_W * z_k * f_k``, where ``f_k`` is a team behavior feature in
[-1, 1]. The same features, averaged over a game, are the adherence metrics.
With ``z = 0`` the tactic reward is exactly zero.

All features are in the team's ego frame (attacking towards +x).

- ``press``: mean x of the team (+1 = camped in the opponent half).
- ``flank``: mean y of the team (+1 = top touchline as drawn, which is the
  left wing for the team attacking to the right; -1 = bottom).
- ``spread``: teammate distance, (d - 25) / 25 clipped (+1 = 50 units apart).
- ``possession``: +1 when an own player is nearest the ball, else -1.
"""

from __future__ import annotations

import numpy as np

TACTICS = ("press", "flank", "spread", "possession")
K = len(TACTICS)
TACTIC_W = 0.002
SPREAD_MID = 25.0


def sample_z(rng: np.random.Generator, n: int, neutral_prob: float = 0.3, active_prob: float = 0.5) -> np.ndarray:
    """``(n, K)`` tactics: ``neutral_prob`` of rows are all zero, otherwise each
    dimension is active with ``active_prob`` and then uniform in [-1, 1]."""
    z = rng.uniform(-1.0, 1.0, (n, K)) * (rng.random((n, K)) < active_prob)
    z[rng.random(n) < neutral_prob] = 0.0
    return z.astype(np.float32)


def team_features(core) -> np.ndarray:
    """Behavior features ``(N, 2, K)`` for team 0 and team 1 of a 2v2 `FootballCore`."""
    c = core.cfg
    P = core.P
    hl, hh = 0.5 * c.field_l, 0.5 * c.field_h
    out = np.zeros((core.n, 2, K))
    d_ball = np.linalg.norm(core.pos - core.ball[:, None, :], axis=-1)
    nearest_team = core.team[np.argmin(d_ball, axis=1)]
    for team in (0, 1):
        sl = slice(team * P, (team + 1) * P)
        x = (core.pos[:, sl, 0] - hl) / hl * (1.0 if team == 0 else -1.0)
        y = (core.pos[:, sl, 1] - hh) / hh
        out[:, team, 0] = x.mean(axis=1)
        out[:, team, 1] = y.mean(axis=1)
        if P > 1:
            gap = np.linalg.norm(core.pos[:, sl.start] - core.pos[:, sl.start + 1], axis=-1)
            out[:, team, 2] = np.clip((gap - SPREAD_MID) / SPREAD_MID, -1.0, 1.0)
        out[:, team, 3] = np.where(nearest_team == team, 1.0, -1.0)
    return out
