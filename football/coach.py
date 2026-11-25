"""LLM coach: a plain-English tactic in, a tactic vector ``z`` out.

The coach asks a small local model (served by Ollama) for one number per
dimension in `football.tactics.TACTICS`, using JSON-schema structured output
at temperature 0. If Ollama is not running or answers badly, a keyword parser
takes over, so the demo and the tests never depend on the model. Every call
can be logged to a JSONL file. The coach is only called when the tactic
changes, never inside the game loop.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path

from .tactics import TACTICS

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:3b-instruct"

SYSTEM = """You turn a football coach's instruction for a 2v2 game into four numbers, each from -1 to 1.
Our team attacks to the right. "Left" means the top of the screen, "right" the bottom.

press: where the team plays. 1 = push high into the opponent half, press, attack. -1 = sit deep near our own goal, defend, park the bus. 0 = no preference.
flank: which side of the pitch. 1 = left wing / top. -1 = right wing / bottom. 0 = middle or no preference.
spread: distance between the two teammates. 1 = spread out, stay wide apart, split up. -1 = stay close together, compact, support each other. 0 = no preference.
possession: who has the ball. 1 = keep the ball, be first to it, control the game. -1 = let the opponent have the ball, wait, counter. 0 = no preference.

The four numbers are independent. Only set a number when the instruction clearly talks about that thing; otherwise it is 0.
Attacking or pressing says nothing about possession. Defending or sitting deep says nothing about possession either.
Use about 0.5 for mild wording ("a bit", "slightly") and 1 for strong wording. If the instruction is not about football tactics, return all zeros.

Examples:
"camp in their half" -> press 1, flank 0, spread 0, possession 0
"guard our box" -> press -1, flank 0, spread 0, possession 0
"hug the left touchline" -> press 0, flank 1, spread 0, possession 0
"stay side by side" -> press 0, flank 0, spread -1, possession 0
"pass it around patiently" -> press 0, flank 0, spread 0, possession 1
"attack fast on the right and stay apart" -> press 1, flank -1, spread 1, possession 0
"let them keep it, we wait" -> press 0, flank 0, spread 0, possession -1
"sing a song" -> press 0, flank 0, spread 0, possession 0"""

SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "number"} for k in TACTICS},
    "required": list(TACTICS),
}

# (pattern, dimension, value); the first match per dimension wins.
KEYWORDS = [
    (r"park the bus|sit (back|deep)|defend|low block|stay back|drop deep|deep", "press", -1.0),
    (r"press|push (up|high)|high line|attack|go forward|aggressive|high", "press", 1.0),
    (r"\bleft\b|\btop\b", "flank", 1.0),
    (r"\bright\b|\bbottom\b", "flank", -1.0),
    (r"spread|wide|width|stretch|split up|apart", "spread", 1.0),
    (r"together|compact|close to each other|tight|stay near|support each other", "spread", -1.0),
    (r"let them have|counter|give them the ball|absorb", "possession", -1.0),
    (r"possession|keep the ball|hold the ball|control|patient|first to the ball", "possession", 1.0),
]
NEGATION = r"(don't|do not|dont|stop|no|never|less)\s+(\w+\s+)?"
MILD = r"a bit|slightly|a little|somewhat|bit more"


def keyword_tactic(text: str) -> dict:
    t = text.lower()
    z = {k: 0.0 for k in TACTICS}
    done = set()
    for pat, dim, val in KEYWORDS:
        if dim in done:
            continue
        m = re.search(pat, t)
        if not m:
            continue
        if re.search(NEGATION + "(" + pat + ")", t):
            val = -val
        z[dim] = val * (0.5 if re.search(MILD, t) else 1.0)
        done.add(dim)
    return z


def _clamp(z: dict) -> dict:
    out = {}
    for k in TACTICS:
        try:
            v = float(z.get(k, 0.0))
        except (TypeError, ValueError):
            v = 0.0
        out[k] = round(max(-1.0, min(1.0, v if v == v else 0.0)), 2)
    return out


def ask_ollama(text: str, model: str = DEFAULT_MODEL, timeout: float = 30.0, url: str = OLLAMA_URL) -> dict:
    body = {
        "model": model,
        "stream": False,
        "format": SCHEMA,
        "options": {"temperature": 0, "seed": 0},
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        content = json.loads(r.read())["message"]["content"]
    return json.loads(content)


class Coach:
    def __init__(self, model: str = DEFAULT_MODEL, log_path: str | Path | None = None, use_llm: bool = True,
                 keywords_first: bool = True):
        self.model = model
        self.log_path = Path(log_path) if log_path else None
        self.use_llm = use_llm
        self.keywords_first = keywords_first  # ask the LLM only when no keyword matches

    def __call__(self, text: str) -> dict:
        """Returns ``{"z": {dim: value}, "source": "llm" | "keywords", "latency_s": float}``."""
        text = text.strip()[:300]
        t0 = time.time()
        raw, source, error = None, "keywords", None
        kw = keyword_tactic(text)
        if self.use_llm and not (self.keywords_first and any(kw.values())):
            try:
                raw = ask_ollama(text, self.model)
                source = "llm"
            except Exception as e:  # noqa: BLE001 - any failure falls back to keywords
                error = f"{type(e).__name__}: {e}"
        z = _clamp(raw) if source == "llm" else kw
        out = {"z": z, "source": source, "latency_s": round(time.time() - t0, 3)}
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {"time": time.time(), "text": text, "model": self.model if source == "llm" else None,
                     "raw": raw, "error": error, **out}
            with self.log_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
        return out


def z_vector(z: dict) -> list[float]:
    return [float(z[k]) for k in TACTICS]
