"""Persistent disk cache for external API calls (URDB, PVWatts, geocoding).

Layered under Streamlit's in-memory ``@st.cache_data`` so both interactive
sessions and batch (sensitivity / test) runs benefit, and so cached values
survive process restarts.

Cache directory lives at ``<repo_root>/.cache/<namespace>`` and is git-ignored.
Bump ``CACHE_VERSION`` to invalidate every namespace at once; per-namespace
invalidation is possible via :func:`clear_namespace`.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import diskcache

CACHE_VERSION = "v1"

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CACHE_ROOT = Path(os.environ.get("PV_SIM_CACHE_DIR", _REPO_ROOT / ".cache"))

_caches: dict[str, diskcache.Cache] = {}


def get_cache(namespace: str) -> diskcache.Cache:
    """Return a process-wide shared cache for ``namespace``."""
    if namespace not in _caches:
        path = _CACHE_ROOT / namespace
        path.mkdir(parents=True, exist_ok=True)
        _caches[namespace] = diskcache.Cache(str(path), size_limit=1_000_000_000)
    return _caches[namespace]


def clear_namespace(namespace: str) -> None:
    cache = get_cache(namespace)
    cache.clear()


def _default_key(args: tuple, kwargs: dict) -> str:
    """Stable hash for positional + keyword args. Non-JSON-serializable values
    fall back to ``repr()`` — callers with unhashable inputs (e.g. large
    DataFrames) should pass an explicit ``key_fn``.
    """
    payload = {
        "args": [_coerce(a) for a in args],
        "kwargs": {k: _coerce(v) for k, v in sorted(kwargs.items())},
        "v": CACHE_VERSION,
    }
    blob = json.dumps(payload, sort_keys=True, default=repr).encode()
    return hashlib.sha256(blob).hexdigest()


def _coerce(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def disk_cached(
    namespace: str,
    ttl: float | None = None,
    key_fn: Callable[..., str] | None = None,
) -> Callable:
    """Decorator: persist the wrapped function's return value on disk.

    Args:
        namespace: subfolder name under ``.cache/``; one cache instance per namespace.
        ttl: seconds until entries expire. ``None`` = never expire.
        key_fn: optional ``(*args, **kwargs) -> str`` to override the default
            JSON-hash key. Use when inputs contain non-JSON-serializable types
            where ``repr`` is not stable across runs.
    """

    def decorator(fn: Callable) -> Callable:
        cache = get_cache(namespace)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = key_fn(*args, **kwargs) if key_fn else _default_key(args, kwargs)
            hit = cache.get(key, default=_MISS)
            if hit is not _MISS:
                return hit
            result = fn(*args, **kwargs)
            cache.set(key, result, expire=ttl)
            return result

        wrapper.cache_clear = cache.clear  # type: ignore[attr-defined]
        wrapper.cache_namespace = namespace  # type: ignore[attr-defined]
        return wrapper

    return decorator


_MISS = object()
