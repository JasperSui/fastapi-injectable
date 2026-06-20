import asyncio
import gc
from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import Any
from unittest.mock import AsyncMock, Mock
from weakref import WeakKeyDictionary

import pytest

from fastapi_injectable.async_exit_stack import AsyncExitStackManager
from fastapi_injectable.exception import DependencyCleanupError


@pytest.fixture
def manager() -> AsyncExitStackManager:
    return AsyncExitStackManager()


@pytest.fixture
def mock_func() -> Mock:
    mock = Mock()
    mock.__name__ = "mock_func"
    return mock


@pytest.fixture
def mock_stack() -> AsyncMock:
    mock = AsyncMock(spec=AsyncExitStack)
    mock.aclose.return_value = None
    return mock


def _register(manager: AsyncExitStackManager, func: Callable[..., Any], stack: AsyncExitStack) -> None:
    """Register ``stack`` for ``func`` under the current running loop (the owner)."""
    loop = asyncio.get_running_loop()
    per_loop = manager._stacks.get(loop)
    if per_loop is None:
        per_loop = WeakKeyDictionary()
        manager._stacks[loop] = per_loop
    per_loop[func] = stack


def _contains(manager: AsyncExitStackManager, func: object) -> bool:
    return any(func in per_loop for per_loop in manager._stacks.values())


async def test_get_stack_creates_new_stack(manager: AsyncExitStackManager, mock_func: Mock) -> None:
    stack = await manager.get_stack(mock_func)
    assert isinstance(stack, AsyncExitStack)
    # Stored under the current event loop (the loop the resolution runs on).
    loop = asyncio.get_running_loop()
    assert manager._stacks[loop][mock_func] is stack


async def test_get_stack_returns_same_stack_for_same_func(manager: AsyncExitStackManager, mock_func: Mock) -> None:
    first = await manager.get_stack(mock_func)
    second = await manager.get_stack(mock_func)
    assert first is second


async def test_cleanup_stack_closes_and_removes_current_loop_stack(
    manager: AsyncExitStackManager, mock_func: Mock, mock_stack: AsyncMock
) -> None:
    _register(manager, mock_func, mock_stack)
    await manager.cleanup_stack(mock_func)

    mock_stack.aclose.assert_awaited_once()
    assert not _contains(manager, mock_func)


async def test_cleanup_stack_with_empty_stacks(manager: AsyncExitStackManager, mock_func: Mock) -> None:
    await manager.cleanup_stack(mock_func)
    assert len(manager._stacks) == 0


async def test_cleanup_stack_with_existing_func_raise_exception(
    manager: AsyncExitStackManager, mock_func: Mock, mock_stack: AsyncMock
) -> None:
    mock_stack.aclose.side_effect = Exception("Cleanup failed")
    _register(manager, mock_func, mock_stack)

    # NOTE: We don't use pytest.raises(DependencyCleanupError) because it's not working somehow,
    # so we use Exception instead, and check the exception type in the assert.
    with pytest.raises(Exception, match="Failed to cleanup stack for mock_func") as exc_info:
        await manager.cleanup_stack(mock_func, raise_exception=True)

    assert isinstance(exc_info.value, DependencyCleanupError)
    assert not _contains(manager, mock_func)


async def test_cleanup_stack_with_nonexistent_func(manager: AsyncExitStackManager, mock_func: Mock) -> None:
    await manager.cleanup_stack(mock_func)
    assert not _contains(manager, mock_func)


async def test_cleanup_stack_with_decorated_func(
    manager: AsyncExitStackManager, mock_func: Mock, mock_stack: AsyncMock
) -> None:
    # Simulate a decorated function whose __original_func__ keys the registry.
    decorated_func = Mock()
    decorated_func.__original_func__ = mock_func

    _register(manager, mock_func, mock_stack)
    await manager.cleanup_stack(decorated_func)

    mock_stack.aclose.assert_awaited_once()
    assert not _contains(manager, mock_func)


async def test_cleanup_stack_skips_loops_without_target_func(
    manager: AsyncExitStackManager, mock_func: Mock, mock_stack: AsyncMock
) -> None:
    """cleanup_stack only closes the target func; loops that don't hold it are left intact."""
    idle_loop = asyncio.new_event_loop()
    other_func = Mock()
    other_func.__name__ = "other_func"
    other_stack = AsyncMock(spec=AsyncExitStack)
    other_stack.aclose.return_value = None
    try:
        _register(manager, mock_func, mock_stack)  # under the current loop
        per_loop: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()
        per_loop[other_func] = other_stack
        manager._stacks[idle_loop] = per_loop  # a different loop, different func

        await manager.cleanup_stack(mock_func)

        mock_stack.aclose.assert_awaited_once()
        other_stack.aclose.assert_not_awaited()
        assert other_func in manager._stacks[idle_loop]
    finally:
        idle_loop.close()


async def test_cleanup_all_stacks_with_stacks(
    manager: AsyncExitStackManager, mock_func: Mock, mock_stack: AsyncMock
) -> None:
    other_func = Mock()
    other_func.__name__ = "other_func"
    other_stack = AsyncMock(spec=AsyncExitStack)
    other_stack.aclose.return_value = None
    _register(manager, mock_func, mock_stack)
    _register(manager, other_func, other_stack)

    await manager.cleanup_all_stacks()

    assert len(manager._stacks) == 0
    mock_stack.aclose.assert_awaited_once()
    other_stack.aclose.assert_awaited_once()


async def test_cleanup_all_stacks_with_no_stacks(manager: AsyncExitStackManager) -> None:
    await manager.cleanup_all_stacks()
    assert len(manager._stacks) == 0


async def test_cleanup_all_stacks_with_error(
    manager: AsyncExitStackManager, mock_func: Mock, mock_stack: AsyncMock
) -> None:
    mock_stack.aclose.side_effect = Exception("Cleanup failed")
    _register(manager, mock_func, mock_stack)

    # Since raise_exception=False, we should not expect an exception
    await manager.cleanup_all_stacks(raise_exception=False)

    assert len(manager._stacks) == 0


async def test_cleanup_all_stacks_with_error_raise_exception(
    manager: AsyncExitStackManager, mock_func: Mock, mock_stack: AsyncMock
) -> None:
    mock_stack.aclose.side_effect = Exception("Cleanup failed")
    _register(manager, mock_func, mock_stack)

    # NOTE: We don't use pytest.raises(DependencyCleanupError) because it's not working somehow,
    # so we use Exception instead, and check the exception type in the assert.
    with pytest.raises(Exception, match="Failed to cleanup one or more dependency stacks") as exc_info:
        await manager.cleanup_all_stacks(raise_exception=True)

    assert isinstance(exc_info.value, DependencyCleanupError)
    assert len(manager._stacks) == 0


async def test_cleanup_all_stacks_with_empty_stacks(manager: AsyncExitStackManager) -> None:
    await manager.cleanup_all_stacks()
    assert len(manager._stacks) == 0


async def test_cleanup_stack_runtime_error_names_func_not_loop(
    manager: AsyncExitStackManager, mock_func: Mock, mock_stack: AsyncMock
) -> None:
    """A RuntimeError raised during teardown is reported against the func, not blamed on the loop."""
    mock_stack.aclose.side_effect = RuntimeError("teardown boom")
    _register(manager, mock_func, mock_stack)

    with pytest.raises(Exception, match="teardown boom") as exc_info:
        await manager.cleanup_stack(mock_func, raise_exception=True)

    assert isinstance(exc_info.value, DependencyCleanupError)
    message = str(exc_info.value)
    assert "mock_func" in message
    assert "something wrong with the loop" not in message
    assert isinstance(exc_info.value.__cause__, RuntimeError)


async def test_cleanup_all_stacks_runtime_error_names_error_not_loop(
    manager: AsyncExitStackManager, mock_func: Mock, mock_stack: AsyncMock
) -> None:
    """A RuntimeError during teardown of all stacks reports the underlying error, not the loop."""
    mock_stack.aclose.side_effect = RuntimeError("teardown boom")
    _register(manager, mock_func, mock_stack)

    with pytest.raises(Exception, match="teardown boom") as exc_info:
        await manager.cleanup_all_stacks(raise_exception=True)

    assert isinstance(exc_info.value, DependencyCleanupError)
    message = str(exc_info.value)
    assert "something wrong with the loop" not in message
    assert isinstance(exc_info.value.__cause__, RuntimeError)


async def test_cleanup_drops_stack_owned_by_idle_non_current_loop(
    manager: AsyncExitStackManager, mock_func: Mock, mock_stack: AsyncMock
) -> None:
    """A stack owned by a loop that is neither current nor running is dropped, not closed."""
    idle_loop = asyncio.new_event_loop()  # open, never running, not the current loop
    try:
        per_loop: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()
        per_loop[mock_func] = mock_stack
        manager._stacks[idle_loop] = per_loop

        await manager.cleanup_all_stacks(raise_exception=True)

        mock_stack.aclose.assert_not_awaited()
        assert len(manager._stacks) == 0
    finally:
        idle_loop.close()


def test_owning_loop_falls_back_to_loop_manager_without_running_loop(manager: AsyncExitStackManager) -> None:
    """Outside any running loop, the owning loop comes from ``loop_manager`` (sync entry points)."""
    assert manager._running_loop() is None
    # With no running loop, _owning_loop() defers to loop_manager (the sync-entry loop).
    owning = manager._owning_loop()
    assert isinstance(owning, asyncio.AbstractEventLoop)


async def test_concurrent_get_stack_access(manager: AsyncExitStackManager, mock_func: Mock) -> None:
    tasks = [manager.get_stack(mock_func) for _ in range(5)]
    stacks = await asyncio.gather(*tasks)

    assert all(stack is stacks[0] for stack in stacks)
    loop = asyncio.get_running_loop()
    assert len(manager._stacks[loop]) == 1


async def test_weakref_cleanup(manager: AsyncExitStackManager) -> None:
    class Temporary:
        def __call__(self) -> None:
            pass

    temp = Temporary()
    loop = asyncio.get_running_loop()
    stack = await manager.get_stack(temp)  # noqa: F841
    assert temp in manager._stacks[loop]

    del temp
    # Force garbage collection to ensure the weak reference is cleaned up.
    gc.collect()

    assert len(manager._stacks[loop]) == 0
