"""Self-play PPO trainer for 1v1 and 2v2 (parameter sharing).

The learner plays ``n_envs`` games in parallel, on team 0 in even-indexed
games and team 1 in odd ones. In 2v2 one shared network controls both
players of the learner's team, so each env-step yields two samples. Games are
split into ``n_groups`` blocks and each block gets one frozen opponent team
per rollout:

- ``opponent_mode="latest"``: every block plays a frozen copy of the current
  learner (classic naive self-play).
- ``opponent_mode="pool"``: with probability ``pool_latest_prob`` a block
  plays the latest copy, otherwise a snapshot sampled uniformly from all past
  snapshots. This is meant to reduce strategy cycling and forgetting.

Reward is the shared team goal reward plus ``coef * shaping`` where ``coef``
decays linearly from ``shaping_start`` to 0 over the first
``shaping_end_frac`` of training (or is always 0 with ``shaping=False``).

With ``tactics=True`` every team gets a tactic vector per episode (see
`football.tactics`), appended to its observations, plus the tactic reward.
Opponents get their own independently sampled tactic.
"""

from __future__ import annotations

import copy
import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from .bots import ChaseBot, RandomBot, TeamChaseBot
from .config import GameConfig
from .core import FootballCore
from .evaluate import play_matches, team_slots
from .policy import ActorCritic, PolicyPlayer, save_checkpoint
from .ppo import PPO, PPOConfig, RolloutBuffer
from .tactics import K, TACTIC_W, sample_z, team_features


@dataclass
class TrainConfig:
    name: str = "main"
    total_steps: int = 20_000_000
    n_envs: int = 64
    n_steps: int = 128
    seed: int = 0
    hidden: int = 128
    players_per_team: int = 1
    tactics: bool = False
    tactic_w: float = TACTIC_W

    opponent_mode: str = "pool"
    pool_latest_prob: float = 0.5
    snapshot_every: int = 10
    n_groups: int = 8

    shaping: bool = True
    shaping_start: float = 1.0
    shaping_end_frac: float = 0.5
    draw_penalty: float = 0.0

    ckpt_every: int = 25
    eval_every: int = 25
    eval_games: int = 200
    out_dir: str = "runs"
    torch_threads: int = 4

    ppo: PPOConfig = field(default_factory=PPOConfig)

    def shaping_coef(self, progress: float) -> float:
        if not self.shaping:
            return 0.0
        return self.shaping_start * max(0.0, 1.0 - progress / self.shaping_end_frac)


@torch.no_grad()
def _sample(net: ActorCritic, obs: np.ndarray) -> np.ndarray:
    logits = net.actor(torch.from_numpy(obs))
    return Categorical(logits=logits).sample().numpy()


class SelfPlayTrainer:
    def __init__(self, cfg: TrainConfig):
        assert cfg.opponent_mode in ("pool", "latest")
        assert cfg.n_envs % cfg.n_groups == 0
        assert not cfg.tactics or cfg.players_per_team == 2
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        torch.set_num_threads(cfg.torch_threads)
        self.rng = np.random.default_rng(cfg.seed)
        self.game = GameConfig(players_per_team=cfg.players_per_team)
        self.P = cfg.players_per_team
        self.obs_dim = self.game.obs_dim + (K if cfg.tactics else 0)

        self.run_dir = Path(cfg.out_dir) / cfg.name
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        with (self.run_dir / "config.json").open("w") as f:
            json.dump({"train": asdict(cfg), "game": self.game.to_dict()}, f, indent=2)

        self.core = FootballCore(cfg.n_envs, self.game, seed=cfg.seed)
        self.net = ActorCritic(obs_dim=self.obs_dim, hidden=cfg.hidden)
        self.ppo = PPO(self.net, cfg.ppo)
        self.buf = RolloutBuffer(cfg.n_steps, cfg.n_envs * self.P, self.obs_dim)

        self.latest = copy.deepcopy(self.net)
        self.pool: list[ActorCritic] = []

        n = cfg.n_envs
        self.rows = np.arange(n)[:, None]
        self.side = np.arange(n) % 2
        self.me = team_slots(n, self.side, self.P)
        self.them = team_slots(n, 1 - self.side, self.P)
        block = n // cfg.n_groups
        self.groups = [np.arange(g * block, (g + 1) * block) for g in range(cfg.n_groups)]

        self.z = np.zeros((n, 2, K), dtype=np.float32)
        if cfg.tactics:
            self.z = sample_z(self.rng, 2 * n).reshape(n, 2, K)

        self.random_bot = RandomBot(seed=cfg.seed + 1)
        self.chase_bot = ChaseBot(self.game) if self.P == 1 else TeamChaseBot(self.game)

    def _pick_opponents(self) -> list[ActorCritic]:
        opps = []
        for _ in self.groups:
            use_pool = self.cfg.opponent_mode == "pool" and self.pool and self.rng.random() >= self.cfg.pool_latest_prob
            opps.append(self.pool[self.rng.integers(len(self.pool))] if use_pool else self.latest)
        return opps

    def _team_obs(self, obs_all: np.ndarray, idx: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Flattened ``(n * P, obs_dim)`` observations of the players in ``idx``."""
        o = obs_all[self.rows, idx]
        if self.cfg.tactics:
            o = np.concatenate([o, np.broadcast_to(z[:, None, :], (*o.shape[:2], K))], axis=2)
        return np.ascontiguousarray(o.reshape(-1, self.obs_dim), dtype=np.float32)

    def train(self) -> Path:
        cfg = self.cfg
        P = self.P
        n = cfg.n_envs
        steps_per_update = n * cfg.n_steps
        n_updates = cfg.total_steps // steps_per_update
        side = self.side
        z_me, z_them = self.z[:, 0], self.z[:, 1]

        obs_all = self.core.reset()
        ep_len = np.zeros(n, dtype=np.int64)
        metrics_path = self.run_dir / "metrics.csv"
        writer = None
        f = metrics_path.open("w", newline="")
        t_start = time.time()

        for update in range(1, n_updates + 1):
            progress = (update - 1) / n_updates
            coef = cfg.shaping_coef(progress)
            self.ppo.set_lr(cfg.ppo.lr * (1.0 - progress))

            opps = self._pick_opponents()
            # Batch together groups that face the same opponent network.
            by_opp: dict[int, tuple[ActorCritic, np.ndarray]] = {}
            for g, net in zip(self.groups, opps):
                key = id(net)
                by_opp[key] = (net, g if key not in by_opp else np.concatenate([by_opp[key][1], g]))

            wins = draws = losses = 0
            lengths = []
            t_roll = time.time()
            for t in range(cfg.n_steps):
                lo = self._team_obs(obs_all, self.me, z_me)
                oo = self._team_obs(obs_all, self.them, z_them).reshape(n, P, -1)
                la, logp, v = self.net.act(torch.from_numpy(lo))
                la = la.numpy()
                oa = np.empty((n, P), dtype=np.int64)
                for net, idx in by_opp.values():
                    oa[idx] = _sample(net, oo[idx].reshape(-1, self.obs_dim)).reshape(-1, P)

                actions = np.empty((n, 2 * P), dtype=np.int64)
                actions[self.rows, self.me] = la.reshape(n, P)
                actions[self.rows, self.them] = oa
                if cfg.tactics:
                    feats = team_features(self.core)
                obs_all, sparse, shaped, done, info = self.core.step(actions)

                reward = sparse[self.rows, self.me] + coef * shaped[self.rows, self.me]
                reward -= cfg.draw_penalty * info["truncated"][:, None]
                if cfg.tactics:
                    mine = feats[np.arange(n), side]
                    reward += cfg.tactic_w * (z_me * mine).sum(axis=1)[:, None]
                self.buf.add(t, lo, la, logp.numpy(), v.numpy(), reward.reshape(-1), np.repeat(done, P))

                ep_len += 1
                if done.any():
                    g = np.where(side == 0, info["goal"], -info["goal"])[done]
                    wins += int((g == 1).sum())
                    draws += int((g == 0).sum())
                    losses += int((g == -1).sum())
                    lengths.extend(ep_len[done].tolist())
                    ep_len[done] = 0
                    if cfg.tactics:
                        k = int(done.sum())
                        z_me[done] = sample_z(self.rng, k)
                        z_them[done] = sample_z(self.rng, k)
            rollout_time = time.time() - t_roll

            last_v = self.net.value(torch.from_numpy(self._team_obs(obs_all, self.me, z_me))).detach().numpy()
            stats = self.ppo.update(self.buf, last_v)

            self.latest.load_state_dict(self.net.state_dict())
            if update % cfg.snapshot_every == 0:
                self.pool.append(copy.deepcopy(self.latest))

            env_steps = update * steps_per_update
            games = max(wins + draws + losses, 1)
            row = {
                "update": update,
                "env_steps": env_steps,
                "time_s": round(time.time() - t_start, 1),
                "sps": round(steps_per_update / (time.time() - t_roll), 0),
                "rollout_s": round(rollout_time, 2),
                "shaping_coef": round(coef, 4),
                "lr": self.ppo.opt.param_groups[0]["lr"],
                "pool_size": len(self.pool),
                "train_win": wins / games,
                "train_draw": draws / games,
                "train_loss": losses / games,
                "ep_len": float(np.mean(lengths)) if lengths else float("nan"),
                **stats,
                "win_vs_random": "",
                "win_vs_chase": "",
                "loss_vs_chase": "",
            }

            if update % cfg.eval_every == 0 or update == n_updates:
                player = PolicyPlayer(self.net)
                r_rand = play_matches(player, self.random_bot, cfg.eval_games, seed=update, config=self.game)
                r_chase = play_matches(player, self.chase_bot, cfg.eval_games, seed=update, config=self.game)
                row["win_vs_random"] = r_rand.win_rate
                row["win_vs_chase"] = r_chase.win_rate
                row["loss_vs_chase"] = r_chase.losses / r_chase.games
                print(
                    f"[{cfg.name}] upd {update}/{n_updates} steps {env_steps / 1e6:.2f}M "
                    f"sps {row['sps']:.0f} shape {coef:.2f} train W/D/L {row['train_win']:.2f}/"
                    f"{row['train_draw']:.2f}/{row['train_loss']:.2f} len {row['ep_len']:.0f} "
                    f"ent {stats['entropy']:.2f} | vs random {r_rand.win_rate:.2f} "
                    f"vs chase {r_chase.win_rate:.2f}/{row['loss_vs_chase']:.2f}",
                    flush=True,
                )

            meta = {"players_per_team": P, "tactics": cfg.tactics}
            if update % cfg.ckpt_every == 0 or update == n_updates:
                save_checkpoint(self.ckpt_dir / f"ckpt_{update:05d}.pt", self.net, update=update, env_steps=env_steps, **meta)

            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            f.flush()

        f.close()
        final = self.run_dir / "final.pt"
        save_checkpoint(final, self.net, update=n_updates, env_steps=n_updates * steps_per_update, **meta)
        return final
