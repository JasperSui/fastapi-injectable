import signal
from collections.abc import AsyncGenerator, Generator
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.fastapi_injectable.util import (
    cleanup_all_exit_stacks,
    cleanup_exit_stack_of_func,
    clear_dependency_cache,
    get_injected_obj,
    setup_graceful_shutdown,
)


class DummyDependency:
    def __init__(self, attr_1: int | None = None, attr_2: str | None = None) -> None:
        self.attr_1 = attr_1
        self.attr_2 = attr_2


def dummy_get_dependency() -> DummyDependency:
    return DummyDependency()


async def dummy_async_get_dependency() -> DummyDependency:
    return DummyDependency()


@pytest.fixture
def mock_injectable() -> Generator[Mock, None, None]:
    with patch("src.fastapi_injectable.util.injectable") as mock:
        yield mock


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


def test_get_injected_obj_sync(mock_injectable: Mock, mock_run_coroutine_sync: Mock) -> None:
    mock_injectable.return_value = lambda: DummyDependency()
    result = get_injected_obj(dummy_get_dependency)

    mock_injectable.assert_called_once_with(dummy_get_dependency, use_cache=True)
    assert isinstance(result, DummyDependency)
    mock_run_coroutine_sync.assert_not_called()


def test_get_injected_obj_async(mock_injectable: Mock, mock_run_coroutine_sync: Mock) -> None:
    mock_injectable.return_value = lambda: DummyDependency()
    mock_run_coroutine_sync.return_value = DummyDependency()

    result: DummyDependency = get_injected_obj(dummy_async_get_dependency)

    mock_injectable.assert_called_once_with(dummy_async_get_dependency, use_cache=True)
    assert isinstance(result, DummyDependency)
    mock_run_coroutine_sync.assert_called_once()


async def test_get_injected_obj_async_generator(mock_injectable: Mock, mock_run_coroutine_sync: Mock) -> None:
    async def dummy_async_gen_dependency() -> AsyncGenerator[DummyDependency, None]:
        yield DummyDependency()

    mock_injectable.return_value = lambda: dummy_async_gen_dependency()
    mock_run_coroutine_sync.return_value = DummyDependency()

    result: DummyDependency = get_injected_obj(dummy_async_gen_dependency)

    mock_injectable.assert_called_once_with(dummy_async_gen_dependency, use_cache=True)
    assert isinstance(result, DummyDependency)
    mock_run_coroutine_sync.assert_called_once()


def test_get_injected_obj_sync_generator(mock_injectable: Mock, mock_run_coroutine_sync: Mock) -> None:
    def dummy_gen_dependency() -> Generator[DummyDependency, None, None]:
        yield DummyDependency()

    mock_injectable.return_value = lambda: dummy_gen_dependency()
    result: DummyDependency = get_injected_obj(dummy_gen_dependency)

    mock_injectable.assert_called_once_with(dummy_gen_dependency, use_cache=True)
    assert isinstance(result, DummyDependency)
    mock_run_coroutine_sync.assert_not_called()


def test_get_injected_obj_with_args(mock_injectable: Mock, mock_run_coroutine_sync: Mock) -> None:
    mock_injectable.return_value = lambda *args, **kwargs: DummyDependency(*args, **kwargs)
    result = get_injected_obj(dummy_get_dependency, args=[42, "test"])

    mock_injectable.assert_called_once_with(dummy_get_dependency, use_cache=True)
    assert result.attr_1 == 42
    assert result.attr_2 == "test"
    mock_run_coroutine_sync.assert_not_called()


def test_get_injected_obj_with_kwargs(mock_injectable: Mock, mock_run_coroutine_sync: Mock) -> None:
    mock_injectable.return_value = lambda *args, **kwargs: DummyDependency(*args, **kwargs)
    result = get_injected_obj(dummy_get_dependency, kwargs={"attr_1": 42, "attr_2": "test"})

    mock_injectable.assert_called_once_with(dummy_get_dependency, use_cache=True)
    assert result.attr_1 == 42
    assert result.attr_2 == "test"
    mock_run_coroutine_sync.assert_not_called()


def test_get_injected_obj_with_args_and_kwargs(mock_injectable: Mock, mock_run_coroutine_sync: Mock) -> None:
    mock_injectable.return_value = lambda *args, **kwargs: DummyDependency(*args, **kwargs)
    result = get_injected_obj(dummy_get_dependency, args=[42], kwargs={"attr_2": "test"})

    mock_injectable.assert_called_once_with(dummy_get_dependency, use_cache=True)
    assert result.attr_1 == 42
    assert result.attr_2 == "test"
    mock_run_coroutine_sync.assert_not_called()


async def test_get_injected_obj_async_with_args_kwargs(mock_injectable: Mock, mock_run_coroutine_sync: Mock) -> None:
    mock_injectable.return_value = lambda *args, **kwargs: DummyDependency(*args, **kwargs)
    mock_run_coroutine_sync.return_value = DummyDependency(attr_1=42, attr_2="test")

    result = get_injected_obj(dummy_async_get_dependency, args=[42], kwargs={"attr_2": "test"})

    mock_injectable.assert_called_once_with(dummy_async_get_dependency, use_cache=True)
    assert result.attr_1 == 42
    assert result.attr_2 == "test"
    mock_run_coroutine_sync.assert_called_once()


def test_get_injected_obj_sync_generator_with_args_kwargs(mock_injectable: Mock, mock_run_coroutine_sync: Mock) -> None:
    def dummy_gen_with_args(*args: Any, **kwargs: Any) -> Generator[DummyDependency, None, None]:  # noqa: ANN401
        yield DummyDependency(*args, **kwargs)

    mock_injectable.return_value = lambda *args, **kwargs: dummy_gen_with_args(*args, **kwargs)
    result = get_injected_obj(dummy_gen_with_args, args=[42], kwargs={"attr_2": "test"})

    mock_injectable.assert_called_once_with(dummy_gen_with_args, use_cache=True)
    assert result.attr_1 == 42
    assert result.attr_2 == "test"
    mock_run_coroutine_sync.assert_not_called()


async def test_get_injected_obj_async_generator_with_args_kwargs(
    mock_injectable: Mock, mock_run_coroutine_sync: Mock
) -> None:
    async def dummy_async_gen_with_args(*args: Any, **kwargs: Any) -> AsyncGenerator[DummyDependency, None]:  # noqa: ANN401
        yield DummyDependency(*args, **kwargs)

    mock_injectable.return_value = lambda *args, **kwargs: dummy_async_gen_with_args(*args, **kwargs)
    mock_run_coroutine_sync.return_value = DummyDependency(attr_1=42, attr_2="test")

    result = get_injected_obj(dummy_async_gen_with_args, args=[42], kwargs={"attr_2": "test"})

    mock_injectable.assert_called_once_with(dummy_async_gen_with_args, use_cache=True)
    assert result.attr_1 == 42
    assert result.attr_2 == "test"
    mock_run_coroutine_sync.assert_called_once()


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


async def test_setup_graceful_shutdown_handler_called(mock_run_coroutine_sync: Mock) -> None:
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
