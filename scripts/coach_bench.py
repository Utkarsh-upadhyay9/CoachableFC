"""Benchmark the LLM coach against the keyword fallback.

Each phrase lists the expected sign of the dimensions it talks about; every
other dimension should come back near 0. A value counts as positive above
0.25 and negative below -0.25.

Example:
    python scripts/coach_bench.py --models qwen2.5:3b-instruct llama3.2:3b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from football.coach import Coach  # noqa: E402
from football.tactics import TACTICS  # noqa: E402

BENCH = [
    ("press high", {"press": 1}),
    ("push up and attack", {"press": 1}),
    ("get forward and hunt the ball in their half", {"press": 1, "possession": 1}),
    ("park the bus", {"press": -1}),
    ("sit deep and protect our goal", {"press": -1}),
    ("defend, we're winning", {"press": -1}),
    ("don't press, stay back", {"press": -1}),
    ("attack down the left", {"press": 1, "flank": 1}),
    ("play on the right wing", {"flank": -1}),
    ("stay on the top side of the pitch", {"flank": 1}),
    ("use the bottom flank", {"flank": -1}),
    ("go through the middle", {}),
    ("spread out", {"spread": 1}),
    ("stay wide apart so they can't mark both of us", {"spread": 1}),
    ("stick together", {"spread": -1}),
    ("stay close and support each other", {"spread": -1}),
    ("keep the ball", {"possession": 1}),
    ("hold possession", {"possession": 1}),
    ("be first to every loose ball", {"possession": 1}),
    ("let them have the ball and hit them on the counter", {"possession": -1}),
    ("sit back, absorb pressure and counter", {"press": -1, "possession": -1}),
    ("press high and spread out", {"press": 1, "spread": 1}),
    ("defend compactly on the right", {"press": -1, "spread": -1, "flank": -1}),
    ("keep possession on the left side", {"possession": 1, "flank": 1}),
    ("be a bit more aggressive", {"press": 1}),
    ("slightly deeper please", {"press": -1}),
    ("play normally", {}),
    ("whatever you think is best", {}),
    ("make me a sandwich", {}),
    ("asdfgh", {}),
    ("all out attack, both of you go forward together", {"press": 1, "spread": -1}),
    ("one of you stay back, the other go wide and forward", {"spread": 1}),
]

# Held out: written before any prompt or keyword tuning, never used to tune.
TEST = [
    ("high pressing game", {"press": 1}),
    ("take the game to them", {"press": 1}),
    ("everyone back, protect the lead", {"press": -1}),
    ("drop off and stay compact", {"press": -1, "spread": -1}),
    ("play down the left touchline", {"flank": 1}),
    ("overload the right side", {"flank": -1}),
    ("focus our play up top", {"flank": 1}),
    ("keep it central", {}),
    ("give each other more space", {"spread": 1}),
    ("cover the whole width of the pitch", {"spread": 1}),
    ("stay tight as a pair", {"spread": -1}),
    ("don't drift apart", {"spread": -1}),
    ("dominate the ball", {"possession": 1}),
    ("win every 50/50", {"possession": 1}),
    ("let them come at us and break fast", {"possession": -1, "press": -1}),
    ("hang back and wait for their mistake", {"press": -1, "possession": -1}),
    ("press them on the left and stay together", {"press": 1, "flank": 1, "spread": -1}),
    ("spread wide and keep the ball", {"spread": 1, "possession": 1}),
    ("a little more attacking", {"press": 1}),
    ("calm down, stop pressing", {"press": -1}),
    ("just have fun", {}),
    ("what's the weather like", {}),
    ("defend deep on the bottom side", {"press": -1, "flank": -1}),
    ("go for the win, all forward", {"press": 1}),
]

SETS = {"dev": BENCH, "test": TEST}


def sign(v: float) -> int:
    return 1 if v > 0.25 else (-1 if v < -0.25 else 0)


def grade(coach: Coach, bench: list) -> dict:
    dims_ok, phrases_ok, lat, rows = 0, 0, [], []
    for text, expect in bench:
        out = coach(text)
        got = {k: sign(out["z"][k]) for k in TACTICS}
        want = {k: expect.get(k, 0) for k in TACTICS}
        ok = sum(got[k] == want[k] for k in TACTICS)
        dims_ok += ok
        phrases_ok += ok == len(TACTICS)
        lat.append(out["latency_s"])
        rows.append({"text": text, "z": out["z"], "source": out["source"], "correct_dims": ok})
    return {
        "dim_accuracy": dims_ok / (len(bench) * len(TACTICS)),
        "phrase_accuracy": phrases_ok / len(bench),
        "median_latency_s": float(np.median(lat)),
        "llm_calls": sum(r["source"] == "llm" for r in rows),
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=["qwen2.5:3b-instruct", "llama3.2:3b"])
    ap.add_argument("--out", default="results/2v2/coach_bench.json")
    args = ap.parse_args()

    coaches = {"keywords": Coach(use_llm=False)}
    for m in args.models:
        coaches[m] = Coach(model=m, log_path="results/2v2/coach_log.jsonl", keywords_first=False)
        coaches[m]("warm up")
        coaches[f"keywords+{m}"] = Coach(model=m, log_path="results/2v2/coach_log.jsonl")
    results = {s: {name: grade(c, bench) for name, c in coaches.items()} for s, bench in SETS.items()}
    for s, res in results.items():
        print(f"--- {s} set ({len(SETS[s])} phrases)")
        for name, r in res.items():
            print(f"{name:>32s}  dims {r['dim_accuracy']:.1%}  phrases {r['phrase_accuracy']:.1%}  "
                  f"median latency {r['median_latency_s']:.2f}s  llm calls {r['llm_calls']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"n_phrases": {s: len(b) for s, b in SETS.items()}, "results": results}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
