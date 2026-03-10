"""
Parse result cache — file-level Redis cache keyed by SHA256.

Avoids redundant parsing of the same file. Cache key format: parse:{sha256}:{doc_type}
Default TTL: 24 hours.

Usage::

    from docmirror.framework.cache import parse_cache

    # Lookup cache
    cached = await parse_cache.get(checksum, doc_type)
    if cached:
        return cached  # PerceptionResult JSON str

    # Write cache
    await parse_cache.set(checksum, doc_type, result_json)
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Default TTL: 24 hours
_DEFAULT_TTL = 86400


class ParseCache:
    """Async Redis parse result cache."""

    def __init__(self):
        self._redis = None

    async def _get_redis(self):
        """Lazily connect to Redis using REDIS_URL env var."""
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
                self._redis = aioredis.from_url(url, decode_responses=True)
                logger.info(f"[ParseCache] Connected to Redis: {url}")
            except Exception as e:
                logger.warning(f"[ParseCache] Redis unavailable: {e}")
                return None
        return self._redis

    @staticmethod
    def _key(checksum: str, doc_type: str = "") -> str:
        """Generate cache key."""
        suffix = f":{doc_type}" if doc_type else ""
        return f"parse:{checksum}{suffix}"

    async def get(self, checksum: str, doc_type: str = "") -> Optional[str]:
        """Lookup cache, returns JSON string or None."""
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

    async def set(
        self, checksum: str, doc_type: str, json_str: str, ttl: int = _DEFAULT_TTL
    ) -> bool:
        """Write to cache."""
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


# Global singleton
parse_cache = ParseCache()
