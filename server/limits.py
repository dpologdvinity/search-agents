"""Request guards for the public demo: client keys, body size and JSON checks, rate limit, search cap.

Every router takes its client key, body limit and JSON parsing from here, so the rules are defined once.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import math
import time
from contextlib import asynccontextmanager

from starlette.requests import HTTPConnection
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Largest request body, and largest WebSocket text message, the server reads. The biggest real request is a few KB.
MAX_BODY_BYTES = 256 * 1024

# Fly's edge proxies reach the app from its private 6PN network (fdaa::/16). 172.16.0.0/12 is the private IPv4 range
# used by Docker bridges. Only a peer in one of these ranges may vouch for a Fly-Client-IP header.
FLY_PROXY_NETWORKS = (ipaddress.ip_network("172.16.0.0/12"), ipaddress.ip_network("fdaa::/16"))


def client_key(conn: HTTPConnection) -> str:
    """The address a rate limit counts this request against, for an HTTP request or a WebSocket.

    Any client can send a Fly-Client-IP header, so it is believed only when the socket peer is Fly's proxy. Otherwise
    a direct client could rotate the header to get a fresh rate-limit bucket on every request. The Dockerfile runs
    uvicorn with --proxy-headers and --forwarded-allow-ips set to the same Fly ranges, so when X-Forwarded-For comes
    from Fly, uvicorn has already replaced conn.client with the real client and the header is not consulted.
    """
    peer = conn.client.host if conn.client else "unknown"
    if _is_fly_proxy(peer):
        return conn.headers.get("fly-client-ip") or peer
    return peer


def _is_fly_proxy(host: str) -> bool:
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:  # "unknown" or "testclient": not an address
        return False
    return any(addr in net for net in FLY_PROXY_NETWORKS)


class NonFiniteJSON(ValueError):
    """The JSON held NaN, Infinity, or a number too large for a float."""


def loads_finite(text: str | bytes):
    """json.loads that refuses NaN, Infinity, and float literals that overflow to inf (such as 1e400).

    Python's json accepts all of those and passes them to the models as floats. A JSON response cannot carry them, so
    a route that echoes or returns one fails with a 500. Invalid JSON still raises json.JSONDecodeError, a ValueError.
    """

    def reject_constant(token: str):
        raise NonFiniteJSON("NaN and Infinity are not allowed in JSON numbers")

    def finite_float(literal: str) -> float:
        value = float(literal)
        if not math.isfinite(value):
            raise NonFiniteJSON("numbers must be finite")
        return value

    return json.loads(text, parse_constant=reject_constant, parse_float=finite_float)


class BodyLimit:
    """ASGI middleware: 413 for a request body over MAX_BODY_BYTES, 422 for JSON holding NaN or Infinity.

    It sits in front of routing, so an oversized body is refused before any route parses it. The limit counts streamed
    bytes, not only the Content-Length header, so a chunked upload is capped too. A body under the limit is read here
    once, checked, and then handed to the app from memory.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope["headers"]).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            await _reply(send, 413, "request body is over the 256 KB limit")
            return

        chunks: list[bytes] = []
        total = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":  # the client disconnected while sending
                return
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > MAX_BODY_BYTES:
                await _reply(send, 413, "request body is over the 256 KB limit")
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)

        if body:
            try:
                loads_finite(body)
            except NonFiniteJSON as e:
                # Same shape as FastAPI's own 422, so the page's error display works unchanged.
                detail = [{"type": "value_error", "loc": ["body"], "msg": str(e)}]
                await _reply(send, 422, detail)
                return
            except ValueError:
                pass  # not JSON: the route's own parser answers

        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()  # afterwards, the client disconnect, as usual

        await self.app(scope, replay, send)


async def _reply(send: Send, status: int, detail) -> None:
    body = json.dumps({"detail": detail}).encode()
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


class RateLimiter:
    """Token bucket per client key: `rate` requests per `per` seconds, bursting to `rate`."""

    def __init__(self, rate: int, per: float):
        self.rate = rate
        self.per = per
        self._buckets: dict[str, tuple[float, float]] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        tokens, last = self._buckets.get(key, (float(self.rate), now))
        tokens = min(self.rate, tokens + (now - last) * self.rate / self.per)
        allowed = tokens >= 1
        self._buckets[key] = (tokens - 1 if allowed else tokens, now)
        if len(self._buckets) > 10_000:  # drop idle clients so memory stays bounded
            cutoff = now - self.per
            self._buckets = {k: v for k, v in self._buckets.items() if v[1] > cutoff}
        return allowed


class Busy(Exception):
    pass


class SearchSlots:
    """Limits how many searches run at once; Python search is CPU-bound."""

    def __init__(self, slots: int, wait_seconds: float):
        self._sem = asyncio.Semaphore(slots)
        self._wait = wait_seconds

    @asynccontextmanager
    async def acquire(self):
        try:
            await asyncio.wait_for(self._sem.acquire(), self._wait)
        except TimeoutError:
            raise Busy from None
        try:
            yield
        finally:
            self._sem.release()
