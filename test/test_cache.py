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

    cache.get()[(func, ("key",), "scope")] = "value"
    assert cache.get() == {(func, ("key",), "scope"): "value"}
    await cache.clear()
    assert cache.get() == {}


def test_clear_drops_every_per_loop_cache() -> None:
    """``clear()`` empties the caches of all loops, not just the current one."""
    cache = DependencyCache()

    async def store() -> None:
        cache.get()["key"] = object()

    async def fetch() -> dict[object, object]:
        return cache.get()

    loop_a = asyncio.new_event_loop()
    loop_b = asyncio.new_event_loop()
    try:
        loop_a.run_until_complete(store())
        loop_b.run_until_complete(store())
        assert len(cache._caches) == 2
        loop_a.run_until_complete(cache.clear())
        assert len(cache._caches) == 0
        assert loop_a.run_until_complete(fetch()) == {}
        assert loop_b.run_until_complete(fetch()) == {}
    finally:
        loop_a.close()
        loop_b.close()


def test_cache_is_shared_within_the_same_event_loop() -> None:
    cache = DependencyCache()
    sentinel = object()

    async def scenario() -> None:
        cache.get()["key"] = sentinel
        # The same loop must keep returning the same cache dict / values.
        assert cache.get()["key"] is sentinel
        assert cache.get() is cache.get()

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(scenario())
    finally:
        loop.close()


def test_cache_is_isolated_across_distinct_open_loops() -> None:
    """Regression for https://github.com/JasperSui/fastapi-injectable/issues/186.

    Two *concurrently open* loops must never share a resolved value. A resource
    built on loop A may hold a loop-A-bound asyncio/anyio primitive; serving it
    to loop B blocks forever with no error. This is the common pytest topology of
    a session-scoped fixture loop plus per-test function loops -- both alive at
    once -- which the previous "drop only when the owner loop is *closed*" logic
    failed to protect.
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
        # loop_a is still OPEN, yet loop_b must NOT see loop_a's value.
        assert loop_b.run_until_complete(fetch()) is None
        # And loop_a still sees its own value (in-loop sharing preserved).
        assert loop_a.run_until_complete(fetch()) is sentinel
    finally:
        loop_a.close()
        loop_b.close()


def test_cache_is_dropped_when_owner_loop_closed() -> None:
    """A value resolved on a now-closed loop is never served to a later loop."""
    cache = DependencyCache()

    async def store() -> None:
        cache.get()["key"] = object()

    loop_a = asyncio.new_event_loop()
    loop_a.run_until_complete(store())
    loop_a.close()  # owner loop is now dead

    async def fetch() -> dict[object, object]:
        return cache.get()

    loop_b = asyncio.new_event_loop()
    try:
        fresh = loop_b.run_until_complete(fetch())
    finally:
        loop_b.close()

    assert fresh == {}


def test_closed_loop_cache_is_garbage_collected() -> None:
    """A loop's cache dict does not outlive the loop (weak keying)."""
    import gc

    cache = DependencyCache()

    async def store() -> None:
        cache.get()["key"] = object()

    loop = asyncio.new_event_loop()
    loop.run_until_complete(store())
    loop.close()
    del loop
    gc.collect()

    assert len(cache._caches) == 0
