"""Regression test for cross-event-loop reuse of an InjectableScope (issue #186 follow-up B).

An ``InjectableScope`` object held by the user and reused across event loops must not
share a resolved resource between loops. Its cache is partitioned per loop, exactly
like the global dependency cache: a resource resolved on loop A is never served to a
different loop B. Doing so awaits a loop-A-bound primitive on loop B, which on Python
3.13+ raises ``RuntimeError: ... attached to a different loop`` and on older runtimes
hangs forever.
"""

import asyncio

from fastapi_injectable.scope import InjectableScope
from fastapi_injectable.util import async_get_injected_obj


def test_explicit_scope_cache_is_isolated_per_event_loop() -> None:
    """A scope reused across loops must not serve a loop-A resource to loop B."""
    call_count = 0

    class PooledResource:
        def __init__(self) -> None:
            # Capture the creation loop, exactly as real async clients do.
            self._loop = asyncio.get_running_loop()

        async def request(self) -> int:
            # Drive work through the creation loop. If this instance is reused on a
            # different loop, awaiting this future never completes there.
            fut: asyncio.Future[int] = self._loop.create_future()
            self._loop.call_soon(fut.set_result, 1)
            return await fut

    async def get_resource() -> PooledResource:
        nonlocal call_count
        call_count += 1
        return PooledResource()

    scope = InjectableScope()

    async def scenario() -> int:
        res = await async_get_injected_obj(get_resource, use_cache=True, scope=scope)
        # Same loop + same scope: caching is preserved, the instance is reused.
        res_again = await async_get_injected_obj(get_resource, use_cache=True, scope=scope)
        assert res_again is res
        return await res.request()

    loop_a = asyncio.new_event_loop()
    loop_b = asyncio.new_event_loop()
    try:
        assert loop_a.run_until_complete(scenario()) == 1
        # loop_a is still OPEN; reusing the SAME scope on loop_b must resolve its own
        # resource, not serve loop_a's (which would hang / raise cross-loop).
        assert loop_b.run_until_complete(scenario()) == 1
    finally:
        loop_a.close()
        loop_b.close()

    # Two distinct loops -> two independent resolutions (no cross-loop sharing).
    assert call_count == 2


def test_explicit_scope_caches_per_loop_but_shares_within_a_loop() -> None:
    """Within one loop a reused scope keeps caching; across loops it does not."""
    call_count = 0

    async def get_value() -> object:
        nonlocal call_count
        call_count += 1
        return object()

    scope = InjectableScope()

    async def resolve_twice() -> bool:
        first = await async_get_injected_obj(get_value, use_cache=True, scope=scope)
        second = await async_get_injected_obj(get_value, use_cache=True, scope=scope)
        return first is second

    loop_a = asyncio.new_event_loop()
    loop_b = asyncio.new_event_loop()
    try:
        assert loop_a.run_until_complete(resolve_twice()) is True  # cached within loop A
        assert loop_b.run_until_complete(resolve_twice()) is True  # cached within loop B
    finally:
        loop_a.close()
        loop_b.close()

    # 2 resolutions total: one per loop (the second call on each loop is a cache hit).
    assert call_count == 2
