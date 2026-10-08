"""Abuse guards for the public demo: per-client rate limit and a cap on concurrent searches."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager


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
