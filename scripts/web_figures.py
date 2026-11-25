"""Copy the 2v2 figures and match GIFs into docs/figures/ for the web page.

Also draws two figures that only the page uses: the coach benchmark chart
(from coach_bench.json and coach_bench_browser.json) and a 2x2 grid of the
cross-play heatmaps of the four runs other than 2v2_cond.

Example:
    python scripts/web_figures.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "2v2"
OUT = ROOT / "docs" / "figures"

COPIES = {
    "2v2_learning_curves.png": RES / "learning_curves.png",
    "2v2_elo_compare.png": RES / "elo_compare.png",
    "2v2_baselines.png": RES / "baselines.png",
    "2v2_crossplay_cond.png": RES / "cond" / "crossplay_heatmap.png",
    "2v2_tactic_sweep.png": RES / "tactics_cond" / "tactic_sweep.png",
    "2v2_tactic_heatmaps.png": RES / "tactics_cond" / "tactic_heatmaps.png",
    "2v2_w10_tactic_sweep.png": RES / "tactics_cond_w10" / "tactic_sweep.png",
    "2v2_w10_tactic_heatmaps.png": RES / "tactics_cond_w10" / "tactic_heatmaps.png",
    "2v2_press_high.gif": ROOT / "media" / "2v2_press_high.gif",
    "2v2_park_the_bus.gif": ROOT / "media" / "2v2_park_the_bus.gif",
    "2v2_pro_vs_chase.gif": ROOT / "media" / "2v2_pro_vs_chase.gif",
    "2v2_w10_left_spread.gif": ROOT / "media" / "2v2_w10_left_spread.gif",
}
GRID_RUNS = ["pool", "pool_seed1", "latest", "cond_w10"]


def heatmap_grid(out: Path, width: int = 1400) -> None:
    tiles = [Image.open(RES / r / "crossplay_heatmap.png").convert("RGB") for r in GRID_RUNS]
    w, h = tiles[0].size
    grid = Image.new("RGB", (2 * w, 2 * h), "white")
    for i, t in enumerate(tiles):
        grid.paste(t.resize((w, h)), ((i % 2) * w, (i // 2) * h))
    grid = grid.resize((width, round(width * h / w)), Image.LANCZOS)
    grid.save(out, optimize=True)


def coach_chart(out: Path) -> None:
    test = json.loads((RES / "coach_bench.json").read_text())["results"]["test"]
    browser = json.loads((RES / "coach_bench_browser.json").read_text())
    rows = [
        ("keywords", test["keywords"]),
        ("qwen2.5 3B", test["qwen2.5:3b-instruct"]),
        ("llama3.2 3B", test["llama3.2:3b"]),
        ("qwen2.5 7B", test["qwen2.5:7b-instruct"]),
        ("keywords, then qwen2.5 3B", test["keywords+qwen2.5:3b-instruct"]),
        ("keywords, then qwen2.5 7B", test["keywords+qwen2.5:7b-instruct"]),
        ("keywords, then Qwen2.5 1.5B\nin the browser", browser),
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    y = range(len(rows))
    vals = [100 * r["dim_accuracy"] for _, r in rows]
    colors = ["#566860"] + ["#1d5bd6"] * 3 + ["#d6382a"] * 3
    ax.barh(list(y), vals, color=colors)
    for i, v in zip(y, vals):
        ax.text(v + 0.8, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_yticks(list(y), [name for name, _ in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("tactic values with the right sign, 24 held-out phrases (%)")
    ax.set_title("Coach accuracy on the test phrases")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, src in COPIES.items():
        shutil.copyfile(src, OUT / name)
    heatmap_grid(OUT / "2v2_crossplay_others.png")
    coach_chart(OUT / "2v2_coach_bench.png")
    print(f"wrote {len(COPIES) + 2} files to {OUT}")


if __name__ == "__main__":
    main()
