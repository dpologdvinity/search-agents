"""Check responsive play layouts against a running app with Playwright installed.

Run: .venv/bin/python tests/web/play_layout.py http://127.0.0.1:8765
"""

import sys

from playwright.sync_api import sync_playwright

GAMES = {
    "routes": "#map",
    "checkers": "#ck",
    "blackjack": "#table",
    "poker": "#table",
    "cartpole": "#stage",
    "nonogram": "#nono-wrap",
    "localize": "#lr-map",
    "minesweeper": "#ms",
    "queens": "#play-board",
    "rover": "#rv-map",
    "ghosthunt": "#gh-map",
    "tetris": "#tt-board",
    "npuzzle": "#board",
}


def check_play_area(page, board_selector):
    """Assert the board, actions and agent panel share the available screen.

    On phones the board takes the column's full width instead, capped by the screen height for phones held
    sideways, and the page scrolls to the actions: fitted to the height left under the header, boards shrank to a
    few pixels per cell.
    """
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_function(
        """selector => {
      const board = document.querySelector(selector);
      const main = board.closest('.proj-game-col');
      const buttons = [...main.querySelectorAll('button')].filter(b =>
        b.checkVisibility() && !b.closest('details') && !b.closest('.tt-over'));
      if (innerWidth <= 760) return board.getBoundingClientRect().width > 40;
      return board.getBoundingClientRect().width > 40 &&
        board.getBoundingClientRect().bottom <= innerHeight + 1 &&
        buttons.every(b => b.getBoundingClientRect().bottom <= innerHeight + 1);
    }""",
        arg=board_selector,
        timeout=5000,
    )
    metrics = page.evaluate(
        """(selector) => {
      const element = document.querySelector(selector);
      const board = element.getBoundingClientRect();
      const main = element.closest('.proj-game-col');
      const buttons = [...main.querySelectorAll('button')].filter(b =>
        b.checkVisibility() && !b.closest('details') && !b.closest('.tt-over'));
      const activity = main.parentElement.querySelector('.play-inspector').getBoundingClientRect();
      return {
        board: {x: board.x, y: board.y, right: board.right, bottom: board.bottom,
          width: board.width, height: board.height},
        actionsBottom: Math.max(...buttons.map(b => b.getBoundingClientRect().bottom)),
        activity: {x: activity.x, y: activity.y, bottom: activity.bottom},
        width: innerWidth, height: innerHeight, pageWidth: document.documentElement.scrollWidth,
        // set inline by fitPlayLayouts only for fitted boards; the card tables size themselves
        boardSize: parseFloat(main.closest('.play-layout').style.getPropertyValue('--play-board-size')) || null,
        columnWidth: Math.floor(main.clientWidth - parseFloat(getComputedStyle(main).paddingLeft)
          - parseFloat(getComputedStyle(main).paddingRight))
      };
    }""",
        board_selector,
    )
    assert metrics["pageWidth"] <= metrics["width"], metrics
    board = metrics["board"]
    assert board["width"] > 40 and board["height"] > 40, metrics
    assert board["x"] >= 0 and board["right"] <= metrics["width"] + 1, metrics
    if metrics["width"] <= 760:
        if metrics["boardSize"] is not None:
            fitted = max(140, min(600, metrics["columnWidth"], metrics["height"] - 24))
            assert metrics["boardSize"] == fitted, metrics
        return
    assert board["bottom"] <= metrics["height"] + 1, metrics
    assert metrics["actionsBottom"] <= metrics["height"] + 1, metrics
    if metrics["width"] > 760:
        assert metrics["activity"]["x"] >= board["right"] - 1, metrics
        assert metrics["activity"]["y"] < board["bottom"], metrics
        assert metrics["activity"]["bottom"] <= metrics["height"] + 1, metrics


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        for width, height in [(1366, 768), (1024, 768), (390, 844), (740, 360), (1280, 600)]:
            for game, board in GAMES.items():
                page = browser.new_page(viewport={"width": width, "height": height}, reduced_motion="reduce")
                page.route("https://fonts.googleapis.com/**", lambda route: route.abort())
                errors = []
                page.on("pageerror", lambda error, captured=errors: captured.append(str(error)))
                # networkidle: pages that fetch content (Localize's presets) grow and refit once it arrives
                page.goto(f"{base_url}/{game}.html", wait_until="networkidle")
                if game == "nonogram":
                    page.locator(".nono-cell").first.wait_for()
                if game == "minesweeper":
                    page.locator(".ms-cell").first.wait_for()
                check_play_area(page, board)
                if game in {"blackjack", "poker"}:
                    page.click("#btn-deal" if game == "blackjack" else "#mode-watch")
                    if game == "blackjack":
                        page.locator(".bj-hand").first.wait_for()
                    check_play_area(page, board)
                if game in {"checkers", "minesweeper", "nonogram"}:
                    page.locator(".play-settings").evaluate("""e => {
                      e.open = true;
                      for (const details of e.querySelectorAll('details')) details.open = true;
                    }""")
                    selector, value = {
                        "checkers": ("#size", "12"),
                        "minesweeper": ("#preset", "expert"),
                        "nonogram": ("#puzzle", "cat"),
                    }[game]
                    page.select_option(selector, value)
                    page.locator(".play-settings").evaluate("e => e.open = false")
                    if game == "nonogram":
                        page.wait_for_function("document.querySelectorAll('.nono-cell').length === 360")
                    check_play_area(page, board)
                assert not errors, errors
                print(f"PASS {game} at {width}x{height}", flush=True)
                page.close()
        browser.close()


if __name__ == "__main__":
    main()
