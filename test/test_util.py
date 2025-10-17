import signal
from collections.abc import AsyncGenerator, Generator
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.fastapi_injectable.util import (
    cleanup_all_exit_stacks,
    cleanup_exit_stack_of_func,
    clear_dependency_cache,
    setup_graceful_shutdown,
)


@pytest.fixture
def mock_run_coroutine_sync() -> Generator[Mock, None, None]:
    with patch("src.fastapi_injectable.util.run_coroutine_sync") as mock:
        yield mock


@pytest.fixture
def mock_async_exit_stack_manager() -> Generator[Mock, None, None]:
    with patch("src.fastapi_injectable.util.async_exit_stack_manager") as mock:
        mock.cleanup_stack = AsyncMock()
        mock.cleanup_all_stacks = AsyncMock()
        yield mock


@pytest.fixture
def mock_dependency_cache() -> Generator[Mock, None, None]:
    with patch("src.fastapi_injectable.util.dependency_cache") as mock:
        mock.clear = AsyncMock()
        yield mock


@pytest.fixture(autouse=True)
async def clear_registered_app() -> AsyncGenerator[None, None]:
    """Clear any registered FastAPI app before and after each test."""
    # Clear before test
    from src.fastapi_injectable import main

    async with main._app_lock:
        main._app = None
    yield
    # Clear after test
    async with main._app_lock:
        main._app = None


async def test_cleanup_exit_stack_of_func(mock_async_exit_stack_manager: Mock) -> None:
    def func() -> None:
        return None

    await cleanup_exit_stack_of_func(func)
    mock_async_exit_stack_manager.cleanup_stack.assert_awaited_once_with(func, raise_exception=False)


async def test_cleanup_all_exit_stacks(mock_async_exit_stack_manager: Mock) -> None:
    await cleanup_all_exit_stacks()
    mock_async_exit_stack_manager.cleanup_all_stacks.assert_awaited_once()


async def test_clear_dependency_cache(mock_dependency_cache: Mock) -> None:
    await clear_dependency_cache()
    mock_dependency_cache.clear.assert_awaited_once()


def test_setup_graceful_shutdown(mock_run_coroutine_sync: Mock) -> None:
    with patch("src.fastapi_injectable.util.atexit.register") as mock_register:  # noqa: SIM117
        with patch("src.fastapi_injectable.util.signal.signal") as mock_signal:
            setup_graceful_shutdown()

            mock_register.assert_called_once()
            assert mock_signal.call_count == 2  # SIGINT and SIGTERM
            mock_signal.assert_any_call(signal.SIGINT, mock_register.call_args[0][0])
            mock_signal.assert_any_call(signal.SIGTERM, mock_register.call_args[0][0])


def test_setup_graceful_shutdown_custom_signals(mock_run_coroutine_sync: Mock) -> None:
    custom_signals = [signal.SIGINT]

    with patch("src.fastapi_injectable.util.atexit.register") as mock_register:  # noqa: SIM117
        with patch("src.fastapi_injectable.util.signal.signal") as mock_signal:
            setup_graceful_shutdown(signals=custom_signals)

            mock_register.assert_called_once()
            assert mock_signal.call_count == 1
            mock_signal.assert_any_call(signal.SIGINT, mock_register.call_args[0][0])


async def test_setup_graceful_shutdown_handler_called(
    mock_run_coroutine_sync: Mock,
) -> None:
    with patch("src.fastapi_injectable.util.atexit.register") as mock_register:  # noqa: SIM117
        with patch("src.fastapi_injectable.util.signal.signal") as mock_signal:  # noqa: F841
            setup_graceful_shutdown()

            # Get the registered cleanup handler
            cleanup_handler = mock_register.call_args[0][0]

            # Call the handler directly to test it
            cleanup_handler()

            mock_run_coroutine_sync.assert_called_once()

            # Get the coroutine that was passed to run_coroutine_sync
            cleanup_coro = mock_run_coroutine_sync.call_args[0][0]

            # Actually await the coroutine
            await cleanup_coro

            assert cleanup_coro.__name__ == "cleanup_all_exit_stacks"
