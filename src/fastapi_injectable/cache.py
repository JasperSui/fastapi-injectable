import asyncio
import threading
from typing import Any
from weakref import WeakKeyDictionary

from .concurrency import resolving_synchronously


class DependencyCache:
    """Dependency resolution cache, partitioned per event loop for async callers.

    On a cache hit FastAPI's ``solve_dependencies`` returns the cached value without
    re-entering the provider. A resource resolved on one event loop -- and any
    loop-bound asyncio primitive it holds (an ``asyncio.Lock``, an ``httpx``/``anyio``
    connection pool, an ``asyncpg`` connection, ...) -- must never be served to a
    *different*, concurrently-live loop: awaiting it there blocks forever with no
    error, or raises ``RuntimeError: ... got Future ... attached to a different loop``.
    That is the silent deadlock in
    https://github.com/JasperSui/fastapi-injectable/issues/186.

    So each running event loop gets its own cache dict, keyed weakly by the loop
    (dropped automatically once the loop is garbage-collected). The very common
    pytest topology of a session-scoped fixture loop plus per-test function loops --
    or an app loop plus a worker loop -- therefore never shares a resolved instance
    across loops, while values are still shared across calls on the *same* loop.

    The synchronous API (``get_injected_obj`` / ``run_coroutine_sync``) is different:
    ``loop_manager`` drives it on a transient/policy loop that can change from one
    call to the next. Partitioning by that loop would defeat caching entirely, so
    while a synchronous resolution is in flight (flagged by
    :data:`~fastapi_injectable.concurrency.resolving_synchronously`) all calls share a
    single cache regardless of which throwaway loop they happen to run on.
    """

    def __init__(self) -> None:
        # One cache dict per (genuine async) event loop. ``WeakKeyDictionary`` lets a
        # loop's cache disappear automatically once nothing else references the loop.
        self._caches: WeakKeyDictionary[asyncio.AbstractEventLoop, dict[Any, Any]] = WeakKeyDictionary()
        # Shared cache for the synchronous API (and for any call with no running loop),
        # whose underlying loop is transient and must not be used to partition.
        self._shared_cache: dict[Any, Any] = {}
        # A plain threading lock (not an ``asyncio.Lock``): the cache is touched
        # from arbitrary loops and threads, and an ``asyncio.Lock`` would bind to
        # the first loop that awaits it and then reject every other loop. The
        # guarded sections only mutate dicts, so they never block.
        self._lock = threading.Lock()

    def _isolation_loop(self) -> asyncio.AbstractEventLoop | None:
        """The loop whose cache the current call must use, or ``None`` for the shared cache.

        Genuine async callers (``async_get_injected_obj`` awaited on a running loop)
        isolate per loop, so a resolved instance is never reused across loops (#186).
        Synchronous resolution driven by ``loop_manager`` runs on transient loops that
        vary call-to-call, so it shares one cache instead -- as does any call made with
        no running loop at all.
        """
        if resolving_synchronously.get():
            return None
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return None

    def get(self) -> dict[Any, Any]:
        """Return the cache dict for the current call (see the class docstring and #186)."""
        loop = self._isolation_loop()
        if loop is None:
            return self._shared_cache
        with self._lock:
            cache = self._caches.get(loop)
            if cache is None:
                cache = {}
                self._caches[loop] = cache
            return cache

    async def clear(self) -> None:
        """Clear the shared cache and every per-loop cache."""
        with self._lock:
            self._caches.clear()
            self._shared_cache.clear()


dependency_cache = DependencyCache()
