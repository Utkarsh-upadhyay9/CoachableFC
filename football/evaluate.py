"""Head-to-head evaluation, cross-play round robins and Elo fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DEFAULT_CONFIG, GameConfig
from .core import FootballCore


@dataclass
class MatchResult:
    wins: int
    draws: int
    losses: int

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.games, 1)

    @property
    def score(self) -> float:
        """Win = 1, draw = 0.5, loss = 0."""
        return (self.wins + 0.5 * self.draws) / max(self.games, 1)


def team_slots(n_games: int, side: np.ndarray, P: int) -> np.ndarray:
    """Player indices ``(n_games, P)`` of the team on ``side`` in each game."""
    return side[:, None] * P + np.arange(P)[None]


def team_act(team, obs: np.ndarray) -> np.ndarray:
    """Actions ``(B, P)`` for observations ``(B, P, D)``.

    ``team`` is one player used in every slot (batched in one call) or a list
    with one player per slot.
    """
    b, p, d = obs.shape
    if isinstance(team, (list, tuple)):
        return np.stack([pl(obs[:, k]) for k, pl in enumerate(team)], axis=1)
    return np.asarray(team(obs.reshape(b * p, d))).reshape(b, p)


def run_games(team_a, team_b, n_games: int = 200, seed: int = 0, config: GameConfig = DEFAULT_CONFIG, on_step=None):
    """Play single-goal games of A vs B, A on each side half the time.

    Returns per-game results from A's point of view (+1 / 0 / -1).
    ``on_step(env, a_idx, info, alive)`` is called after every step, for stats.
    """
    env = FootballCore(n_games, config, seed=seed, auto_reset=False)
    P = env.P
    a_side = np.arange(n_games) % 2
    rows = np.arange(n_games)[:, None]
    a_idx = team_slots(n_games, a_side, P)
    b_idx = team_slots(n_games, 1 - a_side, P)
    obs = env.reset()
    result = np.zeros(n_games, dtype=np.int64)
    alive = np.ones(n_games, dtype=bool)
    while alive.any():
        actions = np.zeros((n_games, env.M), dtype=np.int64)
        actions[rows, a_idx] = team_act(team_a, obs[rows, a_idx])
        actions[rows, b_idx] = team_act(team_b, obs[rows, b_idx])
        obs, _, _, done, info = env.step(actions)
        if on_step is not None:
            on_step(env, a_idx, info, alive)
        finished = alive & done
        goal_for_a = np.where(a_side == 0, info["goal"], -info["goal"])
        result[finished] = goal_for_a[finished]
        alive &= ~done
    return result


def play_matches(
    player_a,
    player_b,
    n_games: int = 200,
    seed: int = 0,
    config: GameConfig = DEFAULT_CONFIG,
) -> MatchResult:
    """Play ``n_games`` single-goal games of A vs B, A on each side half the time."""
    result = run_games(player_a, player_b, n_games, seed, config)
    return MatchResult(int((result == 1).sum()), int((result == 0).sum()), int((result == -1).sum()))


def crossplay(players: list, n_games: int = 200, seed: int = 0, config: GameConfig = DEFAULT_CONFIG, verbose=False):
    """Round robin. Returns ``(score, win_rate)`` matrices; entry ``[i, j]`` is i's result vs j."""
    k = len(players)
    score = np.full((k, k), 0.5)
    win = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            r = play_matches(players[i], players[j], n_games, seed=seed + 1000 * i + j, config=config)
            score[i, j], score[j, i] = r.score, 1.0 - r.score
            win[i, j], win[j, i] = r.win_rate, r.losses / r.games
        if verbose:
            print(f"crossplay row {i + 1}/{k} done", flush=True)
    np.fill_diagonal(win, np.nan)
    return score, win


def fit_elo(score: np.ndarray, anchor: int = 0, anchor_rating: float = 1000.0, iters: int = 20000, lr: float = 20.0):
    """Fit Elo ratings so that predicted expected scores match ``score``.

    Minimizes cross-entropy between observed scores and the Elo logistic
    ``1 / (1 + 10 ** ((R_j - R_i) / 400))`` by gradient descent, then shifts
    ratings so ``anchor`` has ``anchor_rating``.
    """
    k = score.shape[0]
    s = np.clip(score, 1e-3, 1 - 1e-3)
    mask = ~np.eye(k, dtype=bool)
    r = np.zeros(k)
    for _ in range(iters):
        diff = (r[:, None] - r[None, :]) / 400.0
        p = 1.0 / (1.0 + 10.0 ** (-diff))
        grad = ((p - s) * mask).sum(axis=1)
        r -= lr * grad
        r -= r.mean()
    return r - r[anchor] + anchor_rating
