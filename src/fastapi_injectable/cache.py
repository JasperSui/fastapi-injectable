import asyncio
from typing import Any

from .concurrency import loop_manager


class DependencyCache:
    """Dependency resolution cache, invalidated when its event loop dies.

    The resolved dependency values are shared in a single dict, but that dict
    is dropped as soon as the event loop that populated it has been closed and
    a different loop becomes active.

    On a cache hit FastAPI's ``solve_dependencies`` returns the cached value
    without re-entering the provider. A resource resolved on a now-closed loop
    (and any loop-bound asyncio primitive it holds, e.g. an ``asyncio.Lock``)
    must never be reused on a live loop -- doing so blocks forever with no
    error. That is the silent deadlock in
    https://github.com/JasperSui/fastapi-injectable/issues/186.

    Values resolved on a loop that is still open are kept even when a different
    loop becomes active, so in-process cache sharing across calls (e.g. repeated
    ``get_injected_obj`` calls, whose synchronous helper may run on a different
    event loop object each time) keeps returning the same instances.
    """

    def __init__(self) -> None:
        self._cache: dict[Any, Any] = {}
        self._lock = asyncio.Lock()
        # The event loop that populated the current cache contents. A strong
        # reference to a single loop is kept (and replaced whenever the active
        # loop changes), which is enough to ask whether it has been closed.
        self._owner_loop: asyncio.AbstractEventLoop | None = None

    def _drop_cache_if_owner_loop_closed(self) -> None:
        """Drop cached values that belong to a closed event loop (see #186)."""
        current = loop_manager.get_loop()
        if self._owner_loop is current:
            return
        if self._owner_loop is not None and self._owner_loop.is_closed():
            self._cache.clear()
        self._owner_loop = current

    def get(self) -> dict[Any, Any]:
        """Get the current cache, dropping values owned by a closed loop."""
        self._drop_cache_if_owner_loop_closed()
        return self._cache

    async def clear(self) -> None:
        """Clear the cache."""
        if not self._cache:
            return

        async with self._lock:
            self._cache.clear()


dependency_cache = DependencyCache()
