import asyncio
from contextlib import AsyncExitStack
from unittest.mock import AsyncMock, Mock

import pytest

from src.fastapi_injectable.exception import DependencyCleanupError
from src.fastapi_injectable.manager import AsyncExitStackManager


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
    return AsyncMock(spec=AsyncExitStack)


async def test_get_stack_creates_new_stack(manager: AsyncExitStackManager, mock_func: Mock) -> None:
    stack = await manager.get_stack(mock_func)
    assert isinstance(stack, AsyncExitStack)
    assert mock_func in manager._stacks
    assert manager._stacks[mock_func] is stack


async def test_get_stack_returns_existing_stack(manager: AsyncExitStackManager, mock_func: Mock) -> None:
    stack1 = await manager.get_stack(mock_func)
    stack2 = await manager.get_stack(mock_func)
    assert stack1 is stack2


async def test_cleanup_stack_with_existing_func(
    manager: AsyncExitStackManager, mock_func: Mock, mock_stack: AsyncMock
) -> None:
    manager._stacks[mock_func] = mock_stack
    await manager.cleanup_stack(mock_func)

    mock_stack.aclose.assert_awaited_once()
    assert mock_func not in manager._stacks


async def test_cleanup_stack_with_existing_func_but_with_empty_stack(
    manager: AsyncExitStackManager, mock_func: Mock
) -> None:
    manager._stacks = AsyncMock(some_key="some_value")
    mock_stacks_pop = AsyncMock(return_value=None)
    manager._stacks.pop = mock_stacks_pop

    await manager.cleanup_stack(mock_func)  # try to cleanup a function that is not in the stack

    mock_stacks_pop.assert_called_once_with(mock_func, None)


async def test_cleanup_stack_with_existing_func_raise_exception(
    manager: AsyncExitStackManager, mock_func: Mock, mock_stack: AsyncMock
) -> None:
    manager._stacks[mock_func] = mock_stack
    mock_stack.aclose.side_effect = Exception("Cleanup failed")

    with pytest.raises(DependencyCleanupError):
        await manager.cleanup_stack(mock_func, raise_exception=True)

    mock_stack.aclose.assert_awaited_once()
    assert mock_func not in manager._stacks


async def test_cleanup_stack_with_nonexistent_func(manager: AsyncExitStackManager, mock_func: Mock) -> None:
    await manager.cleanup_stack(mock_func)
    assert mock_func not in manager._stacks


async def test_cleanup_stack_with_decorated_func(
    manager: AsyncExitStackManager, mock_func: Mock, mock_stack: AsyncMock
) -> None:
    # Simulate a decorated function
    decorated_func = Mock()
    decorated_func.__original_func__ = mock_func

    manager._stacks[mock_func] = mock_stack
    await manager.cleanup_stack(decorated_func)

    mock_stack.aclose.assert_awaited_once()
    assert mock_func not in manager._stacks


async def test_cleanup_all_stacks_with_stacks(
    manager: AsyncExitStackManager, mock_func: Mock, mock_stack: AsyncMock
) -> None:
    manager._stacks[mock_func] = mock_stack
    manager._stacks[Mock()] = AsyncMock(spec=AsyncExitStack)

    await manager.cleanup_all_stacks()

    assert len(manager._stacks) == 0
    mock_stack.aclose.assert_awaited_once()


async def test_cleanup_all_stacks_with_error(
    manager: AsyncExitStackManager, mock_func: Mock, mock_stack: AsyncMock
) -> None:
    manager._stacks[mock_func] = mock_stack
    mock_stack.aclose.side_effect = Exception("Cleanup failed")

    # Since raise_exception=False, we should not expect an exception
    await manager.cleanup_all_stacks(raise_exception=False)

    assert len(manager._stacks) == 0
    mock_stack.aclose.assert_awaited_once()


async def test_cleanup_all_stacks_with_error_raise_exception(
    manager: AsyncExitStackManager, mock_func: Mock, mock_stack: AsyncMock
) -> None:
    manager._stacks[mock_func] = mock_stack
    mock_stack.aclose.side_effect = Exception("Cleanup failed")

    with pytest.raises(DependencyCleanupError):
        await manager.cleanup_all_stacks(raise_exception=True)

    assert len(manager._stacks) == 0
    mock_stack.aclose.assert_awaited_once()


async def test_cleanup_all_stacks_with_empty_stacks(manager: AsyncExitStackManager) -> None:
    await manager.cleanup_all_stacks()
    assert len(manager._stacks) == 0


async def test_concurrent_get_stack_access(manager: AsyncExitStackManager, mock_func: Mock) -> None:
    tasks = [manager.get_stack(mock_func) for _ in range(5)]
    stacks = await asyncio.gather(*tasks)

    assert all(stack is stacks[0] for stack in stacks)
    assert len(manager._stacks) == 1


async def test_weakref_cleanup(manager: AsyncExitStackManager) -> None:
    class Temporary:
        def __call__(self) -> None:
            pass

    temp = Temporary()
    stack = await manager.get_stack(temp)  # noqa: F841
    assert temp in manager._stacks

    del temp
    # Force garbage collection to ensure weak reference is cleaned up
    import gc

    gc.collect()

    assert len(manager._stacks) == 0
