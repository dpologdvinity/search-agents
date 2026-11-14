"""Frontend files tell browsers to revalidate, so a deploy shows up on the next load instead of after a stale
cached copy (a browser kept serving an old js/nav.js, with menu entries the site had since removed)."""

import pytest
from fastapi.testclient import TestClient

from server.app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("path", ["/", "/sudoku.html", "/js/nav.js", "/app.css", "/img/bg/sudoku.svg"])
def test_frontend_files_must_be_revalidated(client, path):
    r = client.get(path)
    assert r.status_code == 200
    assert r.headers["cache-control"] == "no-cache"
    assert "etag" in r.headers  # revalidation stays cheap: an unchanged file comes back as a 304


def test_unchanged_file_revalidates_as_304(client):
    etag = client.get("/js/nav.js").headers["etag"]
    assert client.get("/js/nav.js", headers={"if-none-match": etag}).status_code == 304


def test_background_svgs_are_gzipped(client):
    r = client.get("/img/bg/sudoku.svg", headers={"accept-encoding": "gzip"})
    assert r.status_code == 200
    assert r.headers["content-encoding"] == "gzip"
