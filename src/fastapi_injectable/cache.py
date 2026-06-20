import asyncio
import threading
from typing import Any
from weakref import WeakKeyDictionary

from .concurrency import loop_manager


class DependencyCache:
    """Per-event-loop dependency resolution cache.

    Each event loop gets its own cache dict, keyed weakly by the loop so a
    loop's cache is dropped automatically once the loop is garbage-collected.

    Partitioning by loop is what makes ``use_cache=True`` safe. On a cache hit
    FastAPI's ``solve_dependencies`` returns the cached value without re-entering
    the provider. A resource resolved on one event loop -- and any loop-bound
    asyncio primitive it holds (an ``asyncio.Lock``, an ``httpx``/``anyio``
    connection pool, an ``asyncpg`` connection, ...) -- must never be served to a
    *different* loop: awaiting it there blocks forever with no error, or raises
    ``RuntimeError: ... got Future ... attached to a different loop``. That is the
    silent deadlock in https://github.com/JasperSui/fastapi-injectable/issues/186.

    Because each loop has its own cache, values are still shared across calls that
    run on the *same* loop (e.g. repeated ``get_injected_obj`` calls, whose
    synchronous helper runs on one stable policy loop), while two distinct loops
    -- the very common pytest topology of a session-scoped fixture loop plus
    per-test function loops, or an app loop plus a worker loop -- never share a
    resolved instance.
    """

    def __init__(self) -> None:
        # One cache dict per event loop. ``WeakKeyDictionary`` lets a loop's cache
        # disappear automatically once nothing else references the loop.
        self._caches: WeakKeyDictionary[asyncio.AbstractEventLoop, dict[Any, Any]] = WeakKeyDictionary()
        # Fallback used only when no event loop can be resolved at all (extremely
        # rare; keeps ``get()`` total so callers never crash on a missing loop).
        self._loopless_cache: dict[Any, Any] = {}
        # A plain threading lock (not an ``asyncio.Lock``): the cache is touched
        # from arbitrary loops and threads, and an ``asyncio.Lock`` would bind to
        # the first loop that awaits it and then reject every other loop. The
        # guarded sections only mutate dicts, so they never block.
        self._lock = threading.Lock()

    def _current_loop(self) -> asyncio.AbstractEventLoop | None:
        """The event loop that owns the cache for the current call.

        Prefers the running loop (the loop the resolution actually executes on).
        Falls back to ``loop_manager``'s loop for synchronous entry points so
        repeated sync calls on a stable policy loop keep sharing one cache.
        """
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            pass
        try:
            loop = loop_manager.get_loop()
        except Exception:  # noqa: BLE001 - never let loop discovery break resolution
            return None
        return None if loop.is_closed() else loop

    def get(self) -> dict[Any, Any]:
        """Return the cache dict for the current event loop.

        A distinct loop always gets a distinct dict, so a cached value is never
        reused across loops (see the class docstring and #186).
        """
        loop = self._current_loop()
        if loop is None:
            return self._loopless_cache
        with self._lock:
            cache = self._caches.get(loop)
            if cache is None:
                cache = {}
                self._caches[loop] = cache
            return cache

    async def clear(self) -> None:
        """Clear every per-loop cache."""
        with self._lock:
            self._caches.clear()
            self._loopless_cache.clear()


dependency_cache = DependencyCache()
