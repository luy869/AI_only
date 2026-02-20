"""
Athena — Rate Limiter

Token-bucket rate limiter scoped per Discord user.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from bot.config import settings


@dataclass
class _Bucket:
    """Per-user token bucket."""

    tokens: float
    last_refill: float


class RateLimiter:
    """
    Token-bucket rate limiter.

    - Each user starts with *burst* tokens.
    - Tokens refill at *rate_per_minute / 60* tokens per second.
    - A request consumes 1 token; if none remain, the user is rate-limited.
    """

    def __init__(
        self,
        rate_per_minute: int | None = None,
        burst: int | None = None,
    ) -> None:
        self._rate = (rate_per_minute or settings.rate_limit_per_minute) / 60.0
        self._burst = burst or settings.rate_limit_burst
        self._buckets: dict[int, _Bucket] = {}

    def _get_bucket(self, user_id: int) -> _Bucket:
        now = time.monotonic()
        if user_id not in self._buckets:
            self._buckets[user_id] = _Bucket(tokens=float(self._burst), last_refill=now)
        bucket = self._buckets[user_id]

        # Refill
        elapsed = now - bucket.last_refill
        bucket.tokens = min(float(self._burst), bucket.tokens + elapsed * self._rate)
        bucket.last_refill = now
        return bucket

    def try_acquire(self, user_id: int) -> bool:
        """
        Attempt to consume one token.

        Returns ``True`` if allowed, ``False`` if rate-limited.
        """
        bucket = self._get_bucket(user_id)
        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return True
        return False

    def retry_after(self, user_id: int) -> float:
        """Seconds until the user can make another request."""
        bucket = self._get_bucket(user_id)
        if bucket.tokens >= 1.0:
            return 0.0
        deficit = 1.0 - bucket.tokens
        return deficit / self._rate

    def cleanup(self, max_age_seconds: float = 600.0) -> int:
        """Remove stale buckets.  Returns count of removed entries."""
        now = time.monotonic()
        stale = [
            uid
            for uid, b in self._buckets.items()
            if now - b.last_refill > max_age_seconds
        ]
        for uid in stale:
            del self._buckets[uid]
        return len(stale)
