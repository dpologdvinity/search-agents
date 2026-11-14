"""Every game page's background scene exists and sits outside .proj-page, so the vignette and scanlines paint over it.
A missing file would fail silently in the browser (the page just goes plain dark)."""

import re
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "web"
PAGES = sorted(p for p in WEB.glob("*.html") if 'class="game-bg"' in p.read_text())


def test_every_listed_game_has_a_background():
    games = re.findall(r"\{ href: '([\w-]+)\.html'", (WEB / "js/nav.js").read_text().split("const HIDDEN")[0])
    assert sorted(p.stem for p in PAGES) == sorted(games)


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_background_file_exists_and_sits_outside_the_page_layer(page):
    html = page.read_text()
    src = re.search(r"<div class=\"game-bg\"[^>]*url\('([^']+)'\)", html).group(1)
    assert (WEB / src).is_file()
    assert html.index('class="game-bg"') < html.index('class="proj-page')
