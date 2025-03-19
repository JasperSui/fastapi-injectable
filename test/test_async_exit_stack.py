import asyncio
from contextlib import AsyncExitStack
from unittest.mock import AsyncMock, Mock, patch

import pytest

from fastapi_injectable.async_exit_stack import AsyncExitStackManager
from fastapi_injectable.concurrency import LoopManager
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


def create_mocked_loop_manager() -> Mock:
    mock = Mock(spec=LoopManager)
    mock.in_loop.return_value = True
    return mock


async def test_get_stack_creates_new_stack(manager: AsyncExitStackManager, mock_func: Mock) -> None:
    stack = await manager.get_stack(mock_func)
    assert isinstance(stack, AsyncExitStack)
    assert mock_func in manager._stacks
    assert manager._stacks[mock_func] is stack


@patch(
    "fastapi_injectable.async_exit_stack.loop_manager",
    new_callable=create_mocked_loop_manager,
)
async def test_cleanup_stack_with_existing_func_in_loop(
    mock_loop_manager: Mock, manager: AsyncExitStackManager, mock_func: Mock, mock_stack: AsyncMock
) -> None:
    manager._stacks[mock_func] = mock_stack
    await manager.cleanup_stack(mock_func)

    mock_loop_manager.in_loop.assert_called_once()
    assert mock_func not in manager._stacks


@patch(
    "fastapi_injectable.async_exit_stack.loop_manager",
    new_callable=create_mocked_loop_manager,
)
async def test_cleanup_stack_with_existing_func_not_in_loop(
    mock_loop_manager: Mock, manager: AsyncExitStackManager, mock_func: Mock, mock_stack: AsyncMock
) -> None:
    manager._stacks[mock_func] = mock_stack
    mock_loop_manager.in_loop.return_value = False
    await manager.cleanup_stack(mock_func)

    mock_loop_manager.in_loop.assert_called_once()
    mock_loop_manager.run_in_loop.assert_called_once()
    assert mock_func not in manager._stacks


@patch(
    "fastapi_injectable.async_exit_stack.loop_manager",
    new_callable=create_mocked_loop_manager,
)
async def test_cleanup_stack_with_empty_stacks(
    mock_loop_manager: Mock, manager: AsyncExitStackManager, mock_func: Mock, mock_stack: AsyncMock
) -> None:
    await manager.cleanup_stack(mock_func)
    assert manager._stacks == {}

    mock_loop_manager.in_loop.assert_not_called()


@patch(
    "fastapi_injectable.async_exit_stack.loop_manager",
    new_callable=create_mocked_loop_manager,
)
async def test_cleanup_stack_with_existing_func_raise_exception(
    mock_loop_manager: Mock, manager: AsyncExitStackManager, mock_func: Mock, mock_stack: AsyncMock
) -> None:
    manager._stacks[mock_func] = mock_stack
    mock_loop_manager.in_loop.return_value = False
    mock_loop_manager.run_in_loop.side_effect = Exception("Cleanup failed")

    # NOTE: We don't use pytest.raises(DependencyCleanupError) because it's not working somehow,
    # so we use Exception instead, and check the exception type in the assert.
    with pytest.raises(Exception, match="Failed to cleanup stack for mock_func") as exc_info:
        await manager.cleanup_stack(mock_func, raise_exception=True)

    assert isinstance(exc_info.value, DependencyCleanupError)

    mock_loop_manager.run_in_loop.assert_called_once()
    assert mock_func not in manager._stacks


@patch(
    "fastapi_injectable.async_exit_stack.loop_manager",
    new_callable=create_mocked_loop_manager,
)
async def test_cleanup_stack_with_nonexistent_func(
    mock_loop_manager: Mock, manager: AsyncExitStackManager, mock_func: Mock
) -> None:
    await manager.cleanup_stack(mock_func)
    assert mock_func not in manager._stacks
    mock_loop_manager.in_loop.assert_not_called()


@patch(
    "fastapi_injectable.async_exit_stack.loop_manager",
    new_callable=create_mocked_loop_manager,
)
async def test_cleanup_stack_with_decorated_func(
    mock_loop_manager: Mock, manager: AsyncExitStackManager, mock_func: Mock, mock_stack: AsyncMock
) -> None:
    # Simulate a decorated function
    decorated_func = Mock()
    decorated_func.__original_func__ = mock_func

    manager._stacks[mock_func] = mock_stack
    await manager.cleanup_stack(decorated_func)
    mock_loop_manager.in_loop.assert_called_once()
    assert mock_func not in manager._stacks


@patch(
    "fastapi_injectable.async_exit_stack.loop_manager",
    new_callable=create_mocked_loop_manager,
)
async def test_cleanup_all_stacks_with_stacks(
    mock_loop_manager: Mock, manager: AsyncExitStackManager, mock_func: Mock, mock_stack: AsyncMock
) -> None:
    manager._stacks[mock_func] = mock_stack
    manager._stacks[Mock()] = AsyncMock(spec=AsyncExitStack)

    await manager.cleanup_all_stacks()

    assert len(manager._stacks) == 0
    mock_loop_manager.in_loop.assert_called_once()


@patch(
    "fastapi_injectable.async_exit_stack.loop_manager",
    new_callable=create_mocked_loop_manager,
)
async def test_cleanup_all_stacks_with_no_stacks(
    mock_loop_manager: Mock,
    manager: AsyncExitStackManager,
) -> None:
    await manager.cleanup_all_stacks()
    mock_loop_manager.in_loop.assert_not_called()


@patch(
    "fastapi_injectable.async_exit_stack.loop_manager",
    new_callable=create_mocked_loop_manager,
)
async def test_cleanup_all_stacks_with_error(
    mock_loop_manager: Mock, manager: AsyncExitStackManager, mock_func: Mock, mock_stack: AsyncMock
) -> None:
    manager._stacks[mock_func] = mock_stack
    mock_loop_manager.run_in_loop.side_effect = Exception("Cleanup failed")

    # Since raise_exception=False, we should not expect an exception
    await manager.cleanup_all_stacks(raise_exception=False)

    assert len(manager._stacks) == 0
    mock_loop_manager.in_loop.assert_called_once()


@patch(
    "fastapi_injectable.async_exit_stack.loop_manager",
    new_callable=create_mocked_loop_manager,
)
async def test_cleanup_all_stacks_with_error_raise_exception(
    mock_loop_manager: Mock, manager: AsyncExitStackManager, mock_func: Mock, mock_stack: AsyncMock
) -> None:
    manager._stacks[mock_func] = mock_stack
    mock_loop_manager.in_loop.return_value = False
    mock_loop_manager.run_in_loop.side_effect = Exception("Cleanup failed")

    # NOTE: We don't use pytest.raises(DependencyCleanupError) because it's not working somehow,
    # so we use Exception instead, and check the exception type in the assert.
    with pytest.raises(Exception, match="Failed to cleanup one or more dependency stacks") as exc_info:
        await manager.cleanup_all_stacks(raise_exception=True)

    assert isinstance(exc_info.value, DependencyCleanupError)
    assert len(manager._stacks) == 0
    mock_loop_manager.in_loop.assert_called_once()


@patch(
    "fastapi_injectable.async_exit_stack.loop_manager",
    new_callable=create_mocked_loop_manager,
)
async def test_cleanup_all_stacks_with_empty_stacks(mock_loop_manager: Mock, manager: AsyncExitStackManager) -> None:
    await manager.cleanup_all_stacks()
    assert len(manager._stacks) == 0
    mock_loop_manager.in_loop.assert_not_called()


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
