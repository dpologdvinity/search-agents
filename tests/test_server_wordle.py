import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server import wordle_api
from server.app import app
from server.limits import RateLimiter, SearchSlots


def make_app(rate: int = 30, moves: int = 120) -> FastAPI:
    """A minimal app with just this router, set up the way server.app's lifespan sets it up.

    The router is not registered in server.app here, so the tests mount it directly. Each test gets its own
    limiters, so one test's requests cannot throttle another.
    """
    app = FastAPI()
    app.include_router(wordle_api.router)
    app.state.rate = RateLimiter(rate, per=60.0)
    app.state.move_rate = RateLimiter(moves, per=60.0)
    app.state.slots = SearchSlots(2, wait_seconds=5.0)
    return app


@pytest.fixture
def client():
    with TestClient(make_app()) as c:
        yield c


def test_app_starts_while_the_lexicon_warms():
    # The lifespan starts the table build in a thread. The app must start and answer whether or not that build
    # has finished, because a request waits on the same lock.
    with TestClient(app) as c:
        assert c.get("/api/wordle/meta").status_code == 200


def new_game(c) -> str:
    return c.post("/api/wordle/new").json()["game_id"]


def test_meta_describes_the_lists(client):
    body = client.get("/api/wordle/meta").json()
    assert body["word_length"] == 5 and body["max_guesses"] == 6
    assert body["answers"] > 1000 and body["guesses"] >= body["answers"]
    assert body["strategies"] == ["entropy", "minimax", "random"]


def test_opening_hint_has_the_whole_answer_list(client):
    gid = new_game(client)  # a new game has every answer as a candidate: the opening
    body = client.get("/api/wordle/hint", params={"game_id": gid, "top": 5}).json()
    assert body["remaining"] == client.get("/api/wordle/meta").json()["answers"]
    assert len(body["options"]) == 5 and body["words"] is None
    bits = [o["bits"] for o in body["options"]]
    assert bits == sorted(bits, reverse=True) and bits[0] > 6.0


def test_guess_scores_and_reports_the_bits(client):
    gid = new_game(client)
    body = client.post("/api/wordle/guess", json={"game_id": gid, "word": "TARES"}).json()
    assert body["word"] == "tares" and len(body["feedback"]) == 5 and set(body["feedback"]) <= set("gy.")
    assert body["guess_no"] == 1 and body["solved"] is False and body["over"] is False
    assert body["answer"] is None  # the answer stays hidden until the game is over
    assert body["expected_bits"] > 6.0 and 0 < body["remaining"] < client.get("/api/wordle/meta").json()["answers"]
    assert body["bits_gained"] > 0


def test_guess_rejects_words_outside_the_list(client):
    gid = new_game(client)
    assert client.post("/api/wordle/guess", json={"game_id": gid, "word": "zzzzz"}).status_code == 400
    assert client.post("/api/wordle/guess", json={"game_id": gid, "word": "tar"}).status_code == 422


def test_unknown_game_is_404(client):
    assert client.post("/api/wordle/guess", json={"game_id": "nope", "word": "crane"}).status_code == 404
    assert client.get("/api/wordle/hint", params={"game_id": "nope"}).status_code == 404


def test_solver_plays_a_whole_game_over_the_api(client):
    """The page's AI PLAYS loop: ask for the top-ranked word, then guess it. It must finish within six."""
    gid = new_game(client)
    body = {}
    for _ in range(6):
        word = client.get("/api/wordle/hint", params={"game_id": gid, "top": 1}).json()["options"][0]["word"]
        body = client.post("/api/wordle/guess", json={"game_id": gid, "word": word}).json()
        if body["over"]:
            break
    assert body["over"] is True and body["answer"] is not None
    assert body["solved"] or body["guess_no"] == 6
    # A finished game refuses more guesses.
    assert client.post("/api/wordle/guess", json={"game_id": gid, "word": "crane"}).status_code == 409


def test_hint_lists_words_once_few_are_left(client):
    gid = new_game(client)
    for word in ("tares", "clung"):
        body = client.post("/api/wordle/guess", json={"game_id": gid, "word": word}).json()
        if body["over"]:
            return  # a lucky finish: nothing left to list
    hint = client.get("/api/wordle/hint", params={"game_id": gid, "top": 3}).json()
    assert hint["remaining"] >= 1
    if hint["remaining"] <= wordle_api.LIST_UP_TO:
        assert len(hint["words"]) == hint["remaining"]
    else:
        assert hint["words"] is None


def test_bad_strategy_and_top_are_rejected(client):
    assert client.get("/api/wordle/hint", params={"strategy": "vibes"}).status_code == 422
    assert client.get("/api/wordle/hint", params={"top": 0}).status_code == 422


def test_new_games_are_rate_limited():
    with TestClient(make_app(moves=2)) as c:
        codes = [c.post("/api/wordle/new").status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_concurrent_guesses_cannot_pass_the_six_guess_limit(monkeypatch):
    """Eight guesses sent at once to one game: at most six are scored, numbered 1 to 6 with no gaps."""
    import asyncio

    import httpx

    from wordle.lexicon import get_lexicon

    words = get_lexicon().guesses[:8]
    real_lexicon = wordle_api._lexicon

    async def yielding_lexicon():
        await asyncio.sleep(0)  # a real lookup suspends here, which is where the requests interleaved
        return await real_lexicon()

    monkeypatch.setattr(wordle_api, "_lexicon", yielding_lexicon)
    app_ = make_app(rate=1000, moves=1000)
    with TestClient(app_) as c:
        game_id = new_game(c)

    async def race():
        transport = httpx.ASGITransport(app=app_)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            return await asyncio.gather(*(ac.post("/api/wordle/guess", json={"game_id": game_id, "word": w})
                                          for w in words))

    responses = asyncio.run(race())
    assert all(r.status_code in (200, 409) for r in responses)
    scored = sorted(r.json()["guess_no"] for r in responses if r.status_code == 200)
    assert len(scored) <= 6 and scored == list(range(1, len(scored) + 1))
