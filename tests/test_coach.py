import json

import football.coach as coach_mod
from football.coach import Coach, _clamp, keyword_tactic


def test_falls_back_to_keywords_when_ollama_is_down(monkeypatch, tmp_path):
    def down(*a, **k):
        raise ConnectionRefusedError("no server")

    monkeypatch.setattr(coach_mod, "ask_ollama", down)
    log = tmp_path / "log.jsonl"
    out = Coach(log_path=log, keywords_first=False)("press high on the left")
    assert out["source"] == "keywords"
    assert out["z"]["press"] == 1.0 and out["z"]["flank"] == 1.0
    entry = json.loads(log.read_text().splitlines()[0])
    assert entry["error"].startswith("ConnectionRefusedError")


def test_llm_only_asked_when_no_keyword_matches(monkeypatch):
    monkeypatch.setattr(coach_mod, "ask_ollama", lambda *a, **k: {"press": 0, "flank": 0, "spread": 0, "possession": 1})
    assert Coach()("press high")["source"] == "keywords"
    out = Coach()("dominate the ball")
    assert out["source"] == "llm" and out["z"]["possession"] == 1.0


def test_llm_output_is_clamped_and_validated(monkeypatch):
    monkeypatch.setattr(coach_mod, "ask_ollama", lambda *a, **k: {"press": 7, "flank": "left", "spread": float("nan")})
    z = Coach()("anything")["z"]
    assert z == {"press": 1.0, "flank": 0.0, "spread": 0.0, "possession": 0.0}
    assert _clamp({"press": -3.333}) ["press"] == -1.0


def test_keyword_negation_and_mild_wording():
    assert keyword_tactic("don't press")["press"] == -1.0
    assert keyword_tactic("a bit more aggressive")["press"] == 0.5
    assert keyword_tactic("make me a sandwich") == {"press": 0.0, "flank": 0.0, "spread": 0.0, "possession": 0.0}
