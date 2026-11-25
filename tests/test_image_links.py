"""Every image the web page, README and final report show must exist."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = {
    "docs/index.html": r'<img[^>]+src="([^"]+)"',
    "README.md": r"!\[[^\]]*\]\(([^)\s]+)\)",
    "report/FINAL_REPORT.md": r"!\[[^\]]*\]\(([^)\s]+)\)",
}


@pytest.mark.parametrize("page", PAGES)
def test_images_exist(page):
    path = ROOT / page
    srcs = [s for s in re.findall(PAGES[page], path.read_text()) if not s.startswith(("http", "data:"))]
    assert srcs, f"no images found in {page}"
    missing = [s for s in srcs if not (path.parent / s).is_file()]
    assert not missing, f"{page} points to missing images: {missing}"
