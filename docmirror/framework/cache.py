# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Parsed Result Cache (Redis-backed)
===================================

File-level Redis cache keyed by SHA256. Avoids redundant parsing of the same file.

Features:
    - Lazy connection with socket timeout and connection pool limits
    - Circuit breaker: after 3 consecutive failures, skip Redis for 60s
    - Retry on timeout for transient network issues

Usage::

    from docmirror.framework.cache import parse_cache

    # Lookup cache
    cached = await parse_cache.get(checksum, doc_type)
    if cached:
        return cached  # PerceptionResult JSON string

    # Write cache
    await parse_cache.set(checksum, doc_type, result_json)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Default TTL: 24 hours (in seconds)
_DEFAULT_TTL = 86400

# ── Circuit breaker state ──
_circuit_open_until: float = 0.0
_CIRCUIT_COOLDOWN_SECONDS = 60
_consecutive_failures: int = 0
_FAILURE_THRESHOLD = 3


class ParseCache:
    """Async Redis-based parse result cache with circuit breaker."""

    def __init__(self):
        self._redis = None

    async def _get_redis(self):
        """Lazily connect to Redis using the REDIS_URL environment variable.

        Includes socket timeout, connection pool limits, and circuit breaker
        to prevent per-request latency spikes when Redis is unreachable.
        """
        global _circuit_open_until, _consecutive_failures

        # Circuit breaker: skip if recently failed
        if time.time() < _circuit_open_until:
            return None

        if self._redis is not None:
            return self._redis

        url = os.environ.get("REDIS_URL")
        if not url:
            return None

        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(
                url,
                socket_timeout=3.0,
                socket_connect_timeout=3.0,
                max_connections=10,
                retry_on_timeout=True,
                decode_responses=True,
            )
            # Verify connection
            await self._redis.ping()
            _consecutive_failures = 0
            logger.info(f"[ParseCache] Connected to Redis: {url}")
            return self._redis
        except Exception as e:
            logger.warning(f"[ParseCache] Redis unavailable: {e}")
            self._redis = None
            _consecutive_failures += 1
            if _consecutive_failures >= _FAILURE_THRESHOLD:
                _circuit_open_until = time.time() + _CIRCUIT_COOLDOWN_SECONDS
                logger.warning(f"[ParseCache] Circuit breaker OPEN — skipping Redis for {_CIRCUIT_COOLDOWN_SECONDS}s")
            return None

    @staticmethod
    def _key(checksum: str, doc_type: str = "") -> str:
        """Generate the Redis cache key."""
        suffix = f":{doc_type}" if doc_type else ""
        return f"parse:{checksum}{suffix}"

    async def get(self, checksum: str, doc_type: str = "") -> str | None:
        """Lookup cache. Returns the cached JSON string or None on miss."""
        r = await self._get_redis()
        if not r:
            return None
        try:
            key = self._key(checksum, doc_type)
            cached = await r.get(key)
            if cached:
                logger.info(f"[ParseCache] ✅ HIT {key[:30]}...")
            return cached
        except Exception as e:
            logger.debug(f"[ParseCache] GET error: {e}")
            return None

    async def set(self, checksum: str, doc_type: str, json_str: str, ttl: int = _DEFAULT_TTL) -> bool:
        """Write a JSON string to the cache with an expiration."""
        r = await self._get_redis()
        if not r:
            return False
        try:
            key = self._key(checksum, doc_type)
            await r.setex(key, ttl, json_str)
            logger.info(f"[ParseCache] 📝 SET {key[:30]}... (TTL={ttl}s)")
            return True
        except Exception as e:
            logger.debug(f"[ParseCache] SET error: {e}")
            return False


# Global singleton cache instance
parse_cache = ParseCache()
