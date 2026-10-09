"""Unknown pages get the styled 404 page with status 404; unknown API routes keep their JSON 404."""

import pytest
from fastapi.testclient import TestClient

from server.app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_unknown_page_returns_the_404_game_page(client):
    r = client.get("/no-such-page.html")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("text/html")
    assert "404 // PATH NOT FOUND" in r.text
    assert 'id="ttt-board"' in r.text
    assert 'src="js/notfound.js"' in r.text


def test_unknown_nested_path_returns_the_404_page(client):
    r = client.get("/some/deep/missing/path")
    assert r.status_code == 404
    assert "PATH NOT FOUND" in r.text


@pytest.mark.parametrize("path", ["/api/no-such-route", "/api/npuzzle/nope", "/api"])
def test_unknown_api_route_keeps_the_json_404(client, path):
    r = client.get(path)
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/json")
    assert r.json() == {"detail": "Not Found"}


def test_non_get_requests_do_not_get_the_html_page(client):
    # The HTML page is for GET only; other methods keep the framework's own answer.
    r = client.post("/no-such-page.html")
    assert r.status_code != 200
    assert "PATH NOT FOUND" not in r.text


@pytest.mark.parametrize("path", ["/", "/index.html", "/2048.html", "/nonogram.html", "/js/nav.js"])
def test_existing_pages_still_serve(client, path):
    assert client.get(path).status_code == 200


def test_404_page_asked_for_by_name_is_still_a_404(client):
    # StaticFiles would serve the file with status 200 here; the page is only ever a 404.
    r = client.get("/404.html")
    assert r.status_code == 404
    assert "404 // PATH NOT FOUND" in r.text


@pytest.mark.parametrize("path", ["/a%00b", "/js/%00nav.js", "/%00"])
def test_nul_byte_path_gets_the_html_404_page(client, path):
    r = client.get(path)
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("text/html")
    assert "PATH NOT FOUND" in r.text


def test_health_still_json(client):
    assert client.get("/api/health").json() == {"ok": True}
