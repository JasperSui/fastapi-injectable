import asyncio

import pytest

from src.fastapi_injectable.cache import DependencyCache


@pytest.fixture
def cache() -> DependencyCache:
    return DependencyCache()


def test_initial_cache_empty(cache: DependencyCache) -> None:
    assert cache.get() == {}


async def test_clear_empty_cache(cache: DependencyCache) -> None:
    await cache.clear()
    assert cache.get() == {}


async def test_clear_non_empty_cache(cache: DependencyCache) -> None:
    def func() -> None:
        return None

    cache._cache[(func, ("key",), "scope")] = "value"
    assert cache.get() == {(func, ("key",), "scope"): "value"}
    await cache.clear()
    assert cache.get() == {}


async def test_clear_lock(cache: DependencyCache) -> None:
    # Test that clear acquires the lock properly
    cache._cache = {(lambda: None, ("key",), "scope"): "value"}

    async with cache._lock:
        # Try to clear while lock is held; should not deadlock
        clear_task = asyncio.create_task(cache.clear())
        await asyncio.sleep(0.1)  # Allow clear to attempt
        assert not clear_task.done()
    await clear_task
    assert cache.get() == {}
