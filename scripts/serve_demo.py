"""Serve the browser demo with the LLM coach on this machine.

Static files come from docs/. ``POST /coach`` with ``{"text": "..."}`` returns
the tactic from `football.coach.Coach` (Ollama, falling back to keywords);
``GET /coach/health`` tells the page that typed tactics are available.

Example:
    ollama serve &            # once, if it is not already running
    python scripts/serve_demo.py --model qwen2.5:3b-instruct
    # then open http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from football.coach import Coach  # noqa: E402


class Handler(SimpleHTTPRequestHandler):
    coach: Coach

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/coach/health":
            return self._json(200, {"model": self.coach.model})
        return super().do_GET()

    def do_POST(self):
        if self.path.rstrip("/") != "/coach":
            return self._json(404, {"error": "not found"})
        try:
            n = min(int(self.headers.get("Content-Length", 0)), 4096)
            text = str(json.loads(self.rfile.read(n) or b"{}").get("text", ""))
        except (ValueError, AttributeError):
            return self._json(400, {"error": "expected JSON {\"text\": ...}"})
        if not text.strip():
            return self._json(400, {"error": "empty tactic"})
        out = self.coach(text)
        return self._json(200, {**out, "model": self.coach.model})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="qwen2.5:3b-instruct")
    ap.add_argument("--log", default="results/2v2/coach_demo_log.jsonl")
    args = ap.parse_args()
    Handler.coach = Coach(model=args.model, log_path=ROOT / args.log)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(ROOT / "docs")))
    print(f"demo on http://localhost:{args.port} (coach: {args.model})")
    server.serve_forever()


if __name__ == "__main__":
    main()
