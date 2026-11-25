"""Build a player from a short spec string: ``random``, ``chase`` or a checkpoint path."""

from __future__ import annotations

from pathlib import Path

from .bots import ChaseBot, RandomBot, TeamChaseBot
from .config import DEFAULT_CONFIG, TEAM_CONFIG, GameConfig
from .policy import load_player


def game_config(net) -> GameConfig:
    """1v1 networks see 16 numbers; anything larger is a 2v2 network."""
    return DEFAULT_CONFIG if net.obs_dim == DEFAULT_CONFIG.obs_dim else TEAM_CONFIG


def chase_for(config: GameConfig):
    return ChaseBot(config) if config.players_per_team == 1 else TeamChaseBot(config)


def make_player(spec: str, greedy: bool = False, seed: int | None = None, config: GameConfig = DEFAULT_CONFIG):
    if spec == "random":
        return RandomBot(seed=seed)
    if spec == "chase":
        return chase_for(config)
    path = Path(spec)
    if not path.exists():
        raise FileNotFoundError(f"unknown player spec {spec!r} (expected random, chase or a .pt file)")
    return load_player(path, greedy=greedy)
