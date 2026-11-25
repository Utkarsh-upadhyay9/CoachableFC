"""Vectorized football physics for 1v1 and 2v2.

`FootballCore` simulates `num_envs` independent games at once with numpy so
that PPO can collect experience from many games per step without a Python
loop over environments.

Players are indexed team-major: with ``P = config.players_per_team``, players
``0..P-1`` are team 0 and ``P..2P-1`` are team 1. All players use the same
discrete action space and see an ego-centric observation: team 1's view is
mirrored along x, so from every player's point of view it defends x = -1 and
attacks x = +1. That lets one policy play any slot on either side.

Action encoding (18 actions): ``move = a % 9``, ``kick = a >= 9``.
Move 0 is "stay"; moves 1..8 go counter-clockwise in 45 degree steps,
starting at "forward" (towards the opponent goal).

Observation of player i (``config.obs_dim`` floats, positions scaled to
[-1, 1], velocities by top speed / kick speed, all mirrored for team 1):

- 0:2 own position, 2:4 own velocity
- 4:6 nearest opponent position, 6:8 its velocity
- 8:10 ball position, 10:12 ball velocity, 12:14 ball minus own position
- 14 kick cooldown, 15 time elapsed
- 2v2 only: 16:18 teammate position, 18:20 its velocity,
  20:22 other opponent position, 22:24 its velocity

Opponents are ordered by distance to the observer (ties by index), so the
observation does not depend on which slot an opponent happens to occupy.
"""

from __future__ import annotations

import numpy as np

from .config import DEFAULT_CONFIG, KICKOFF, GameConfig, N_MOVES

_S = 1.0 / np.sqrt(2.0)
MOVE_DIRS = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [_S, _S],
        [0.0, 1.0],
        [-_S, _S],
        [-1.0, 0.0],
        [-_S, -_S],
        [0.0, -1.0],
        [_S, -_S],
    ]
)

BALL_POTENTIAL_W = 1.0
PLAYER_POTENTIAL_W = 0.5


def _norm(v: np.ndarray) -> np.ndarray:
    return np.sqrt((v * v).sum(axis=-1))


class FootballCore:
    """Batch of games with ``P`` players per team (``M = 2P`` players).

    ``step`` returns ``(obs, sparse_reward, shaping_reward, done, info)``:

    - ``obs``: ``(N, M, obs_dim)`` float32, ego-centric per player
    - ``sparse_reward``: ``(N, M)``, +1 for scoring and -1 for conceding
    - ``shaping_reward``: ``(N, M)``, potential-based shaping term
    - ``done``: ``(N,)`` bool, a goal was scored or time ran out
    - ``info["goal"]``: ``(N,)`` int, +1 team 0 scored, -1 team 1 scored
    - ``info["touch"]``: ``(N, M)`` bool, player touched or kicked the ball
    - ``info["kick"]``: ``(N, M)`` bool, player kicked the ball
    """

    def __init__(
        self,
        num_envs: int,
        config: GameConfig = DEFAULT_CONFIG,
        seed: int | None = None,
        auto_reset: bool = True,
    ):
        self.n = num_envs
        self.cfg = config
        self.P = config.players_per_team
        self.M = 2 * self.P
        self.rng = np.random.default_rng(seed)
        self.auto_reset = auto_reset

        self.team = np.arange(self.M) // self.P
        self.pos = np.zeros((num_envs, self.M, 2))
        self.vel = np.zeros((num_envs, self.M, 2))
        self.ball = np.zeros((num_envs, 2))
        self.ball_vel = np.zeros((num_envs, 2))
        self.cooldown = np.zeros((num_envs, self.M), dtype=np.int64)
        self.t = np.zeros(num_envs, dtype=np.int64)
        self._touch = np.zeros((num_envs, self.M), dtype=bool)
        self._kicked = np.zeros((num_envs, self.M), dtype=bool)

        c = config
        spots = np.array(KICKOFF[self.P]) * (c.field_l, c.field_h)
        mirrored = spots.copy()
        mirrored[:, 0] = c.field_l - mirrored[:, 0]
        self.kickoff = np.concatenate([spots, mirrored])
        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self, mask: np.ndarray | None = None) -> np.ndarray:
        c = self.cfg
        if mask is None:
            idx = np.arange(self.n)
        else:
            idx = np.nonzero(mask)[0]
        k = len(idx)
        if k:
            noise = self.rng.uniform(-1.0, 1.0, (k, self.M, 2)) * c.kickoff_noise
            self.pos[idx] = self.kickoff
            self.pos[idx] += noise
            self.vel[idx] = 0.0
            ball_noise = self.rng.uniform(-1.0, 1.0, (k, 2)) * 0.5
            self.ball[idx] = (0.5 * c.field_l, 0.5 * c.field_h)
            self.ball[idx] += ball_noise
            self.ball_vel[idx] = 0.0
            self.cooldown[idx] = 0
            self.t[idx] = 0
        return self.observe()

    # -------------------------------------------------------------- potential
    def potential(self) -> np.ndarray:
        """Shaping potential per player, shape ``(N, M)``."""
        c = self.cfg
        goals = np.array([[c.field_l, 0.5 * c.field_h], [0.0, 0.5 * c.field_h]])[self.team]
        ball_to_goal = _norm(self.ball[:, None, :] - goals[None])
        player_to_ball = _norm(self.pos - self.ball[:, None, :])
        return -(BALL_POTENTIAL_W * ball_to_goal + PLAYER_POTENTIAL_W * player_to_ball) / c.field_l

    # ------------------------------------------------------------------- step
    def step(self, actions: np.ndarray):
        c = self.cfg
        actions = np.asarray(actions, dtype=np.int64).reshape(self.n, self.M)
        moves = actions % N_MOVES
        kicks = actions >= N_MOVES

        phi_before = self.potential()
        self._touch[:] = False

        self._move_players(moves)
        self._collide_players()
        self._ball_contact()
        self._kick(kicks)
        goal = self._move_ball()

        self.t += 1
        truncated = (self.t >= c.max_steps) & (goal == 0)
        terminated = goal != 0
        done = terminated | truncated

        sparse = np.where(self.team == 0, goal[:, None], -goal[:, None]).astype(np.float64)
        shaping = self.potential() - phi_before

        info = {
            "goal": goal,
            "terminated": terminated,
            "truncated": truncated,
            "touch": self._touch.copy(),
            "kick": self._kicked.copy(),
        }
        if self.auto_reset and done.any():
            self.reset(done)
        return self.observe(), sparse, shaping, done, info

    def _move_players(self, moves: np.ndarray) -> None:
        c = self.cfg
        d = MOVE_DIRS[moves].copy()
        d[:, self.P :, 0] *= -1.0
        self.vel = self.vel * c.player_damping + d * c.player_accel
        speed = _norm(self.vel)[..., None]
        self.vel *= np.minimum(1.0, c.player_max_speed / np.maximum(speed, 1e-8))
        self.pos += self.vel

        r = c.player_r
        for axis, hi in ((0, c.field_l - r), (1, c.field_h - r)):
            p = self.pos[..., axis]
            out = (p < r) | (p > hi)
            self.pos[..., axis] = np.clip(p, r, hi)
            self.vel[..., axis][out] = 0.0

    def _collide_players(self) -> None:
        r = self.cfg.player_r
        for i in range(self.M):
            for j in range(i + 1, self.M):
                delta = self.pos[:, j] - self.pos[:, i]
                dist = _norm(delta)
                overlap = 2.0 * r - dist
                if not (overlap > 0).any():
                    continue
                safe = np.maximum(dist, 1e-8)[:, None]
                n = np.where(dist[:, None] > 1e-8, delta / safe, np.array([1.0, 0.0]))
                push = (0.5 * np.maximum(overlap, 0.0))[:, None] * n
                self.pos[:, i] -= push
                self.pos[:, j] += push

    def _ball_contact(self) -> None:
        c = self.cfg
        min_d = c.player_r + c.ball_r
        e = c.ball_player_restitution
        rows = np.arange(self.n)
        # Rotate which player is resolved first so no slot or side is favored.
        for k in range(self.M):
            who = (self.t + k) % self.M
            p = self.pos[rows, who]
            v = self.vel[rows, who]
            rel = self.ball - p
            dist = _norm(rel)
            hit = dist < min_d
            if not hit.any():
                continue
            self._touch[rows[hit], who[hit]] = True
            n = rel / np.maximum(dist, 1e-8)[:, None]
            n[dist < 1e-8] = (1.0, 0.0)
            self.ball[hit] = p[hit] + n[hit] * min_d
            vn = ((self.ball_vel - v) * n).sum(axis=1)
            bounce = hit & (vn < 0)
            self.ball_vel[bounce] -= ((1.0 + e) * vn[bounce])[:, None] * n[bounce]

    def _kick(self, kicks: np.ndarray) -> None:
        c = self.cfg
        np.maximum(self.cooldown - 1, 0, out=self.cooldown)
        rel = self.ball[:, None, :] - self.pos
        dist = _norm(rel)
        can = kicks & (self.cooldown == 0) & (dist <= c.kick_range)
        self._kicked = can
        if not can.any():
            return
        self._touch |= can
        dirn = rel / np.maximum(dist, 1e-8)[..., None]
        impulse = (dirn * c.kick_speed * can[..., None]).sum(axis=1)
        kicked = can.any(axis=1)
        self.ball_vel[kicked] = impulse[kicked]
        self.cooldown[can] = c.kick_cooldown

    def _move_ball(self) -> np.ndarray:
        c = self.cfg
        self.ball += self.ball_vel
        self.ball_vel *= c.ball_friction

        r = c.ball_r
        rest = c.ball_wall_restitution
        by = self.ball[:, 1]
        low = by < r
        high = by > c.field_h - r
        self.ball[low, 1] = 2 * r - by[low]
        self.ball[high, 1] = 2 * (c.field_h - r) - by[high]
        self.ball_vel[low | high, 1] *= -rest

        in_mouth = np.abs(self.ball[:, 1] - 0.5 * c.field_h) < 0.5 * c.goal_w
        bx = self.ball[:, 0]
        left = (bx < r) & ~in_mouth
        right = (bx > c.field_l - r) & ~in_mouth
        self.ball[left, 0] = 2 * r - bx[left]
        self.ball[right, 0] = 2 * (c.field_l - r) - bx[right]
        self.ball_vel[left | right, 0] *= -rest

        goal = np.zeros(self.n, dtype=np.int64)
        goal[(self.ball[:, 0] > c.field_l) & in_mouth] = 1
        goal[(self.ball[:, 0] < 0.0) & in_mouth] = -1
        return goal

    # ---------------------------------------------------------------- observe
    def observe(self) -> np.ndarray:
        c = self.cfg
        P, M = self.P, self.M
        hl, hh = 0.5 * c.field_l, 0.5 * c.field_h
        obs = np.empty((self.n, M, c.obs_dim), dtype=np.float32)
        ball_p = np.stack([(self.ball[:, 0] - hl) / hl, (self.ball[:, 1] - hh) / hh], axis=1)
        ball_v = self.ball_vel / c.kick_speed
        pos_n = (self.pos - (hl, hh)) / (hl, hh)
        vel_n = self.vel / c.player_max_speed
        rows = np.arange(self.n)[:, None]
        for i in range(M):
            team = i // P
            mx = np.array([1.0 if team == 0 else -1.0, 1.0])
            opp = np.arange(M)[self.team != team]
            mates = [j for j in range(team * P, team * P + P) if j != i]
            d = _norm(self.pos[:, opp] - self.pos[:, i : i + 1])
            opp_sorted = opp[np.argsort(d, axis=1, kind="stable")]
            near = opp_sorted[:, 0]
            me_p = pos_n[:, i]
            o = obs[:, i]
            o[:, 0:2] = me_p * mx
            o[:, 2:4] = vel_n[:, i] * mx
            o[:, 4:6] = pos_n[rows[:, 0], near] * mx
            o[:, 6:8] = vel_n[rows[:, 0], near] * mx
            o[:, 8:10] = ball_p * mx
            o[:, 10:12] = ball_v * mx
            o[:, 12:14] = (ball_p - me_p) * mx
            o[:, 14] = self.cooldown[:, i] / c.kick_cooldown
            o[:, 15] = self.t / c.max_steps
            k = 16
            others = [np.full(self.n, j) for j in mates] + [opp_sorted[:, q] for q in range(1, P)]
            for j in others:
                o[:, k : k + 2] = pos_n[rows[:, 0], j] * mx
                o[:, k + 2 : k + 4] = vel_n[rows[:, 0], j] * mx
                k += 4
        return obs
