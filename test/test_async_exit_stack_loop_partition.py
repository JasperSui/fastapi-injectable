"""Regression tests for cross-event-loop exit-stack cleanup (issue #186 follow-up A).

The exit-stack registry must be partitioned by the event loop each async-generator
dependency was *entered* on. A stack entered while resolving on loop A holds that
loop's teardown coroutines, which may ``await`` loop-A-bound primitives (an
``asyncio.Future``/``Event``/``Lock``, an ``httpx``/``anyio`` pool, ...). Closing such
a stack on a *different* loop awaits a loop-A primitive from the wrong loop, which on
Python 3.13+ raises ``RuntimeError: ... attached to a different loop`` and on older
runtimes (or behind a no-guard suspend point) hangs forever.
"""

import asyncio
import threading
from collections.abc import Iterator
from contextlib import suppress

import pytest

from fastapi_injectable.async_exit_stack import AsyncExitStackManager


def _provider() -> None:  # marker object used only as a registry key
    raise NotImplementedError


_provider.__name__ = "provider"


def _run_loop_forever(loop: asyncio.AbstractEventLoop, ready: threading.Event) -> None:
    asyncio.set_event_loop(loop)
    loop.call_soon(ready.set)
    loop.run_forever()


@pytest.fixture
def background_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """A second event loop running forever in its own (daemon) thread."""
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    thread = threading.Thread(target=_run_loop_forever, args=(loop, ready), daemon=True)
    thread.start()
    ready.wait(timeout=5)
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


async def test_cleanup_all_stacks_runs_teardown_on_its_owning_loop(
    background_loop: asyncio.AbstractEventLoop,
) -> None:
    """A stack entered on loop A is torn down on loop A, even when cleanup runs on loop B."""
    manager = AsyncExitStackManager()
    state: dict[str, object] = {}

    async def _enter_on_loop_a() -> None:
        # A pending future bound to loop A; resolved a little later *by loop A*.
        owned_future = background_loop.create_future()
        background_loop.call_later(0.05, owned_future.set_result, "ok")

        stack = await manager.get_stack(_provider)

        async def _teardown() -> None:
            state["teardown_loop"] = asyncio.get_running_loop()
            # Awaiting a loop-A future on the wrong loop raises immediately (3.13+)
            # or hangs; on loop A it resolves cleanly.
            await owned_future
            state["done"] = True

        stack.push_async_callback(_teardown)

    # Enter the async resource on loop A.
    asyncio.run_coroutine_threadsafe(_enter_on_loop_a(), background_loop).result(timeout=5)

    # Trigger cleanup from loop B (this test's loop). The owning loop (A) is alive
    # and running in another thread, so the close must be dispatched onto it.
    await asyncio.wait_for(manager.cleanup_all_stacks(raise_exception=True), timeout=10)

    assert state.get("done") is True
    assert state["teardown_loop"] is background_loop


async def test_cleanup_stack_runs_teardown_on_its_owning_loop(
    background_loop: asyncio.AbstractEventLoop,
) -> None:
    """The single-func cleanup path also tears down on the owning loop."""
    manager = AsyncExitStackManager()
    state: dict[str, object] = {}

    async def _enter_on_loop_a() -> None:
        owned_future = background_loop.create_future()
        background_loop.call_later(0.05, owned_future.set_result, "ok")
        stack = await manager.get_stack(_provider)

        async def _teardown() -> None:
            state["teardown_loop"] = asyncio.get_running_loop()
            await owned_future
            state["done"] = True

        stack.push_async_callback(_teardown)

    asyncio.run_coroutine_threadsafe(_enter_on_loop_a(), background_loop).result(timeout=5)

    await asyncio.wait_for(manager.cleanup_stack(_provider, raise_exception=True), timeout=10)

    assert state.get("done") is True
    assert state["teardown_loop"] is background_loop


async def test_cleanup_all_stacks_drops_stack_when_owning_loop_is_closed() -> None:
    """When the owning loop is already closed, its resources are dead: drop, never block."""
    manager = AsyncExitStackManager()
    teardown_ran = {"value": False}
    holder: dict[str, asyncio.AbstractEventLoop] = {}

    def _setup_then_close_loop() -> None:
        # Enter the stack on a loop in this worker thread, then close that loop,
        # so the owning loop is already dead by the time cleanup runs on loop B.
        loop = asyncio.new_event_loop()

        async def _enter() -> None:
            stack = await manager.get_stack(_provider)

            async def _teardown() -> None:
                teardown_ran["value"] = True
                # Would hang/raise if run on the wrong loop; must never be invoked.
                await asyncio.sleep(0)

            stack.push_async_callback(_teardown)

        loop.run_until_complete(_enter())
        loop.close()
        holder["loop"] = loop

    thread = threading.Thread(target=_setup_then_close_loop)
    thread.start()
    thread.join(timeout=5)
    assert holder["loop"].is_closed()

    # Must complete promptly without awaiting the dead loop's teardown.
    await asyncio.wait_for(manager.cleanup_all_stacks(raise_exception=True), timeout=10)

    assert teardown_ran["value"] is False
    # Registry is emptied regardless.
    with suppress(Exception):
        assert not list(manager._stacks)
