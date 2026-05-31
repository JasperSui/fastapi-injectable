import asyncio
from typing import Any

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


def test_cache_is_shared_within_the_same_event_loop() -> None:
    cache = DependencyCache()
    sentinel = object()

    async def scenario() -> None:
        cache.get()["key"] = sentinel
        assert cache.get()["key"] is sentinel

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(scenario())
    finally:
        loop.close()


def test_cache_is_retained_across_still_open_event_loops() -> None:
    """A value resolved on a loop that is still open is reused on another loop.

    This preserves in-process cache sharing for repeated ``get_injected_obj``
    calls, whose synchronous helper may run on a different event loop object
    each time while every loop stays open.
    """
    cache = DependencyCache()
    sentinel = object()

    async def store() -> None:
        cache.get()["key"] = sentinel

    async def fetch() -> object | None:
        return cache.get().get("key")

    loop_a = asyncio.new_event_loop()
    loop_b = asyncio.new_event_loop()
    try:
        loop_a.run_until_complete(store())
        # loop_a is still OPEN, so its cached value must survive on loop_b.
        assert loop_b.run_until_complete(fetch()) is sentinel
    finally:
        loop_a.close()
        loop_b.close()


def test_cache_is_dropped_when_owner_loop_closed() -> None:
    """Regression for https://github.com/JasperSui/fastapi-injectable/issues/186.

    A value resolved on a now-closed event loop must never be served to a
    later, different loop -- it may hold a loop-bound asyncio primitive that
    would deadlock silently if reused.
    """
    cache = DependencyCache()

    async def store() -> None:
        cache.get()["key"] = object()

    loop_a = asyncio.new_event_loop()
    loop_a.run_until_complete(store())
    loop_a.close()  # owner loop is now dead

    async def fetch() -> dict[Any, Any]:
        return cache.get()

    loop_b = asyncio.new_event_loop()
    try:
        fresh = loop_b.run_until_complete(fetch())
    finally:
        loop_b.close()

    assert fresh == {}
