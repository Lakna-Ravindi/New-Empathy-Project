"""Multi-worker rate limiting for Flask endpoints backed by MongoDB with in-memory fallback."""

import os
import time
import logging
from datetime import datetime, timedelta, timezone
from functools import wraps
from flask import request, jsonify
from pymongo import MongoClient, ReturnDocument
from pymongo.errors import PyMongoError
from mongo_config import MONGODB_URI, MONGODB_DATABASE

logger = logging.getLogger(__name__)


class MultiWorkerRateLimiter:
    """Rate limiter that works across multiple WSGI worker processes using MongoDB atomic increments."""

    def __init__(self, uri=None, database_name=None):
        self.uri = uri or MONGODB_URI
        self.database_name = database_name or MONGODB_DATABASE
        self.client = None
        self.db = None
        self.collection = None
        self.is_connected = False
        self._memory_store = {}
        self._connect()

    def _connect(self):
        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=2000)
            self.client.admin.command("ping")
            self.db = self.client[self.database_name]
            self.collection = self.db["rate_limits"]
            # Ensure TTL index so old rate limit buckets auto-expire
            indexes = self.collection.index_information()
            if not any(tuple(idx.get("key", [])) == (("expires_at", 1),) for idx in indexes.values()):
                self.collection.create_index([("expires_at", 1)], expireAfterSeconds=0, name="idx_rate_limits_ttl")
            if not any(tuple(idx.get("key", [])) == (("key", 1),) for idx in indexes.values()):
                self.collection.create_index([("key", 1)], unique=True, name="uq_rate_limit_key")
            self.is_connected = True
        except Exception as e:
            self.is_connected = False
            logger.warning("MongoDB unavailable for rate limiting, using memory fallback: %s", e)

    def is_rate_limiting_enabled(self) -> bool:
        explicit = os.getenv("RATELIMIT_ENABLED")
        if explicit is not None:
            return explicit.lower() not in {"false", "0", "no", "off"}
        # If running inside automated pytest, default to False to avoid throttling tests
        if os.getenv("PYTEST_CURRENT_TEST"):
            return False
        return True

    def get_client_ip(self) -> str:
        """Resolve the client IP address from proxy headers or remote address."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # First IP in X-Forwarded-For is the original client
            client_ip = forwarded.split(",")[0].strip()
            if client_ip:
                return client_ip
        real_ip = request.headers.get("X-Real-IP")
        if real_ip and real_ip.strip():
            return real_ip.strip()
        return request.remote_addr or "127.0.0.1"

    def check_rate_limit(self, key_prefix: str, limit: int, window_seconds: int = 60) -> tuple[bool, int, int]:
        """
        Check and increment the rate limit counter for a client.
        Returns: (is_allowed, current_count, retry_after_seconds)
        """
        if not self.is_rate_limiting_enabled():
            return True, 0, 0

        client_ip = self.get_client_ip()
        now_ts = int(time.time())
        window_start = now_ts - (now_ts % window_seconds)
        window_key = f"{key_prefix}:{client_ip}:{window_start}"
        retry_after = max(1, (window_start + window_seconds) - now_ts)

        # Try MongoDB atomic update if connected
        if not self.is_connected:
            self._connect()

        if self.is_connected and self.collection is not None:
            try:
                expires_at = datetime.now(timezone.utc) + timedelta(seconds=window_seconds * 2 + 10)
                doc = self.collection.find_one_and_update(
                    {"key": window_key},
                    {
                        "$inc": {"count": 1},
                        "$setOnInsert": {
                            "key": window_key,
                            "prefix": key_prefix,
                            "ip": client_ip,
                            "expires_at": expires_at,
                        },
                    },
                    upsert=True,
                    return_document=ReturnDocument.AFTER,
                )
                count = doc.get("count", 1)
                return (count <= limit), count, retry_after
            except PyMongoError as e:
                logger.warning("MongoDB rate limit check failed, using fallback: %s", e)

        # In-memory fallback
        curr_entry = self._memory_store.get(window_key)
        if curr_entry and curr_entry["window"] == window_start:
            curr_entry["count"] += 1
            count = curr_entry["count"]
        else:
            self._memory_store[window_key] = {"count": 1, "window": window_start}
            count = 1

        # Prune old in-memory entries occasionally
        if len(self._memory_store) > 1000:
            threshold = now_ts - (window_seconds * 2)
            self._memory_store = {
                k: v for k, v in self._memory_store.items()
                if v.get("window", 0) >= threshold
            }

        return (count <= limit), count, retry_after


_rate_limiter = None


def get_rate_limiter() -> MultiWorkerRateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = MultiWorkerRateLimiter()
    return _rate_limiter


def rate_limit(key_prefix: str, default_limit: int, window_seconds: int = 60, env_var: str | None = None):
    """
    Decorator for Flask view functions to enforce configurable rate limits.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            limiter = get_rate_limiter()
            limit = default_limit
            if env_var:
                val = os.getenv(env_var)
                if val:
                    try:
                        limit = int(val)
                    except ValueError:
                        pass

            allowed, count, retry_after = limiter.check_rate_limit(
                key_prefix=key_prefix,
                limit=limit,
                window_seconds=window_seconds,
            )

            if not allowed:
                response = jsonify({
                    "error": "Too many requests. Please try again later.",
                    "retry_after": retry_after,
                })
                response.status_code = 429
                response.headers["Retry-After"] = str(retry_after)
                return response

            return fn(*args, **kwargs)

        return wrapper

    return decorator
