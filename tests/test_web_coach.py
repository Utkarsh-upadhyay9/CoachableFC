"""The browser coach (docs/js/coach.js) must parse tactics like football/coach.py."""

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from football.coach import WEB_DATA, _clamp, keyword_tactic, web_data

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")

spec = importlib.util.spec_from_file_location("coach_bench", ROOT / "scripts" / "coach_bench.py")
coach_bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coach_bench)

EXTRA = ["DON'T PRESS", "never go wide", "a bit more on the left", "stop pressing, stay compact",
         "slightly more possession on the bottom", "attack", "", "push up", "less attack"]
RAW = [{"press": 7, "flank": "left"}, {"press": -3.333}, {"press": "0.5", "possession": True},
       {}, {"spread": None, "flank": [1]}, {"press": 0.333333, "flank": -0.999}]


def test_coach_json_matches_python():
    assert json.loads(WEB_DATA.read_text()) == web_data(), "run: python -m football.coach"


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_keywords_and_clamp_match_python(tmp_path):
    phrases = [t for t, _ in coach_bench.BENCH + coach_bench.TEST] + EXTRA
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps({"phrases": phrases, "raw": RAW}))
    out = subprocess.run([NODE, str(ROOT / "tests" / "coach_parity.mjs"), str(inp)],
                         capture_output=True, text=True, check=True)
    js = json.loads(out.stdout)
    assert len(js["keywords"]) == len(phrases) and len(js["clamp"]) == len(RAW)
    for text, z in zip(phrases, js["keywords"]):
        assert z == keyword_tactic(text), text
    for raw, z in zip(RAW, js["clamp"]):
        assert z == _clamp(raw), raw
