import numpy as np

from football.bots import TeamChaseBot
from football.config import N_MOVES, TEAM_CONFIG
from football.core import FootballCore
from football.evaluate import play_matches
from football.policy import ActorCritic, PolicyPlayer
from football.tactics import K, sample_z, team_features


def test_2v2_shapes():
    env = FootballCore(8, TEAM_CONFIG, seed=0)
    obs = env.reset()
    assert obs.shape == (8, 4, 24)
    obs, sparse, shaping, done, info = env.step(np.zeros((8, 4), dtype=np.int64))
    assert sparse.shape == shaping.shape == info["touch"].shape == (8, 4)


def test_2v2_mirror_symmetry():
    env = FootballCore(1, TEAM_CONFIG, seed=0)
    c = TEAM_CONFIG
    team0 = np.array([[30.0, 20.0], [12.0, 45.0]])
    team1 = team0.copy()
    team1[:, 0] = c.field_l - team1[:, 0]
    env.pos[0] = np.concatenate([team0, team1])
    env.ball[0] = [50.0, 32.0]
    obs = env.observe()[0]
    np.testing.assert_allclose(obs[0], obs[2], atol=1e-6)
    np.testing.assert_allclose(obs[1], obs[3], atol=1e-6)


def test_2v2_opponents_sorted_by_distance():
    env = FootballCore(1, TEAM_CONFIG, seed=0)
    env.pos[0] = [[20.0, 32.0], [10.0, 10.0], [90.0, 50.0], [25.0, 32.0]]
    obs = env.observe()[0, 0]
    hl, hh = 50.0, 32.0
    np.testing.assert_allclose(obs[4:6], [(25 - hl) / hl, 0.0], atol=1e-6)
    np.testing.assert_allclose(obs[20:22], [(90 - hl) / hl, (50 - hh) / hh], atol=1e-6)
    np.testing.assert_allclose(obs[16:18], [(10 - hl) / hl, (10 - hh) / hh], atol=1e-6)


def test_2v2_team_reward_sign():
    env = FootballCore(1, TEAM_CONFIG, seed=0, auto_reset=False)
    c = TEAM_CONFIG
    env.pos[0] = [[80.0, 32.0], [40.0, 10.0], [10.0, 5.0], [10.0, 60.0]]
    env.ball[0] = [80.0 + c.player_r + c.ball_r + 0.5, 32.0]
    for _ in range(60):
        _, sparse, _, done, info = env.step(np.array([[1 + N_MOVES, 0, 0, 0]]))
        if done[0]:
            break
    assert info["goal"][0] == 1
    assert sparse[0].tolist() == [1, 1, -1, -1]


def test_2v2_sides_are_balanced():
    r = play_matches(TeamChaseBot(), TeamChaseBot(), n_games=2000, seed=1, config=TEAM_CONFIG)
    assert abs(r.wins - r.losses) / r.games < 0.05


def test_tactic_features_bounded_and_neutral_sampling():
    env = FootballCore(64, TEAM_CONFIG, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(50):
        env.step(rng.integers(0, 18, (64, 4)))
        f = team_features(env)
        assert f.shape == (64, 2, K)
        assert (np.abs(f) <= 1.0 + 1e-9).all()
    z = sample_z(rng, 10_000)
    assert 0.25 < (np.abs(z).sum(axis=1) == 0).mean() < 0.5
    assert (np.abs(z) <= 1).all()


def test_conditioned_player_defaults_to_neutral_tactic():
    net = ActorCritic(obs_dim=24 + K)
    obs = np.random.default_rng(0).normal(size=(5, 24)).astype(np.float32)
    zero = PolicyPlayer(net, greedy=True)
    explicit = PolicyPlayer(net, greedy=True, z=np.zeros(K))
    assert (zero(obs) == explicit(obs)).all()
