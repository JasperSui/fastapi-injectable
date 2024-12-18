import asyncio
from collections.abc import Callable
from typing import Any


class DependencyCache:
    def __init__(self) -> None:
        self._cache: dict[tuple[Callable[..., Any], tuple[str]], Any] = {}
        self._lock = asyncio.Lock()

    def get(self) -> dict[tuple[Callable[..., Any], tuple[str]], Any]:
        """Get the current cache."""
        return self._cache

    async def clear(self) -> None:
        """Clear the cache."""
        if not self._cache:
            return

        async with self._lock:
            self._cache.clear()


dependency_cache = DependencyCache()
