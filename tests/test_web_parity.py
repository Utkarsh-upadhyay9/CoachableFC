"""The browser physics (docs/js/game.js) must match football/core.py."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from football.bots import ChaseBot, TeamChaseBot
from football.config import GameConfig
from football.core import FootballCore

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node not installed")
@pytest.mark.parametrize("players", [1, 2])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_js_physics_matches_python(tmp_path, seed, players):
    cfg = GameConfig(players_per_team=players)
    rng = np.random.default_rng(seed)
    env = FootballCore(1, cfg, seed=seed, auto_reset=False)
    m = 2 * players
    actions = rng.integers(0, 18, (300, m))
    # Bias towards forward moves + kicks so goals and contacts actually happen.
    actions[::3, 0] = 10
    actions[1::4, m - 1] = 14

    state = {
        "players_per_team": players,
        "pos": env.pos[0].tolist(),
        "vel": env.vel[0].tolist(),
        "ball": env.ball[0].tolist(),
        "ball_vel": env.ball_vel[0].tolist(),
        "cooldown": env.cooldown[0].tolist(),
        "actions": actions.tolist(),
    }
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps(state))
    out = subprocess.run([NODE, str(ROOT / "tests" / "parity_trace.mjs"), str(inp)], capture_output=True, text=True, check=True)
    trace = json.loads(out.stdout)

    bot = ChaseBot() if players == 1 else TeamChaseBot()
    for k, step in enumerate(trace):
        chase = int(bot(env.observe()[:, 0])[0])
        _, _, _, done, info = env.step(actions[k][None])
        assert chase == step["chase"], f"chase bot differs at step {k}"
        np.testing.assert_allclose(env.pos[0], step["pos"], atol=1e-9, err_msg=f"step {k}")
        np.testing.assert_allclose(env.ball[0], step["ball"], atol=1e-9, err_msg=f"step {k}")
        np.testing.assert_allclose(env.observe()[0], step["obs"], atol=1e-5)
        assert info["touch"][0].tolist() == step["touch"]
        assert int(info["goal"][0]) == step["goal"]
        if done[0]:
            break
    assert len(trace) == k + 1
