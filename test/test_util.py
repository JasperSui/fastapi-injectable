import logging
import signal
import subprocess
import sys
import threading
from collections.abc import AsyncGenerator, Generator
from typing import Any
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
            registered_signals = [call.args[0] for call in mock_signal.call_args_list]
            assert signal.SIGINT in registered_signals
            assert signal.SIGTERM in registered_signals


def test_setup_graceful_shutdown_custom_signals(mock_run_coroutine_sync: Mock) -> None:
    custom_signals = [signal.SIGINT]

    with patch("src.fastapi_injectable.util.atexit.register") as mock_register:  # noqa: SIM117
        with patch("src.fastapi_injectable.util.signal.signal") as mock_signal:
            setup_graceful_shutdown(signals=custom_signals)

            mock_register.assert_called_once()
            assert mock_signal.call_count == 1
            registered_signals = [call.args[0] for call in mock_signal.call_args_list]
            assert signal.SIGINT in registered_signals


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


def test_setup_graceful_shutdown_sigterm_handler_terminates(mock_run_coroutine_sync: Mock) -> None:
    with (
        patch("src.fastapi_injectable.util.atexit.register"),
        patch("src.fastapi_injectable.util.signal.getsignal", return_value=signal.SIG_DFL),
        patch("src.fastapi_injectable.util.signal.signal") as mock_signal,
    ):
        setup_graceful_shutdown(signals=[signal.SIGTERM])

    # The handler installed for the signal (second positional arg of signal.signal).
    handler = mock_signal.call_args_list[0].args[1]

    with pytest.raises(SystemExit) as exc_info:
        handler(signal.SIGTERM, None)

    assert exc_info.value.code == 0
    mock_run_coroutine_sync.assert_called_once()
    # Close the (mocked, never-awaited) cleanup coroutine to keep output pristine.
    mock_run_coroutine_sync.call_args[0][0].close()


def test_setup_graceful_shutdown_sigint_handler_terminates(mock_run_coroutine_sync: Mock) -> None:
    with (
        patch("src.fastapi_injectable.util.atexit.register"),
        patch("src.fastapi_injectable.util.signal.getsignal", return_value=signal.SIG_DFL),
        patch("src.fastapi_injectable.util.signal.signal") as mock_signal,
    ):
        setup_graceful_shutdown(signals=[signal.SIGINT])

    handler = mock_signal.call_args_list[0].args[1]

    with pytest.raises(KeyboardInterrupt):
        handler(signal.SIGINT, None)

    mock_run_coroutine_sync.call_args[0][0].close()


def test_setup_graceful_shutdown_chains_previous_handler(mock_run_coroutine_sync: Mock) -> None:
    previous_handler = Mock()

    with (
        patch("src.fastapi_injectable.util.atexit.register"),
        patch("src.fastapi_injectable.util.signal.getsignal", return_value=previous_handler),
        patch("src.fastapi_injectable.util.signal.signal") as mock_signal,
    ):
        setup_graceful_shutdown(signals=[signal.SIGTERM])

    handler = mock_signal.call_args_list[0].args[1]

    with pytest.raises(SystemExit):
        handler(signal.SIGTERM, None)

    # The pre-existing handler must be invoked, not clobbered.
    previous_handler.assert_called_once_with(signal.SIGTERM, None)
    mock_run_coroutine_sync.call_args[0][0].close()


def test_setup_graceful_shutdown_skips_signal_handlers_off_main_thread(
    mock_run_coroutine_sync: Mock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    captured: dict[str, Any] = {}

    def run_in_thread() -> None:
        # Do NOT mock signal.signal here: the real main-thread guard must prevent
        # the call, otherwise CPython raises ValueError off the main thread.
        with patch("src.fastapi_injectable.util.atexit.register") as mock_register:
            try:
                setup_graceful_shutdown(signals=[signal.SIGTERM])
                captured["error"] = None
            except Exception as exc:  # noqa: BLE001
                captured["error"] = exc
            captured["atexit_called"] = mock_register.called

    with caplog.at_level(logging.WARNING, logger="fastapi_injectable"):
        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()

    assert captured["error"] is None  # no ValueError off the main thread
    assert captured["atexit_called"] is True  # atexit cleanup is still registered
    assert any("main thread" in record.message for record in caplog.records)


def test_setup_graceful_shutdown_install_signal_handlers_false(mock_run_coroutine_sync: Mock) -> None:
    with (
        patch("src.fastapi_injectable.util.atexit.register") as mock_register,
        patch("src.fastapi_injectable.util.signal.signal") as mock_signal,
    ):
        setup_graceful_shutdown(install_signal_handlers=False)

    mock_register.assert_called_once()  # atexit cleanup still registered
    mock_signal.assert_not_called()  # but no OS signal handlers installed


_TERMINATION_SCRIPT = """
import os
import signal
import sys
import time

from fastapi_injectable import setup_graceful_shutdown

setup_graceful_shutdown(signals=[signal.{signal_name}])
os.kill(os.getpid(), signal.{signal_name})
time.sleep(5)
sys.stdout.write("STILL ALIVE")
sys.stdout.flush()
sys.exit(0)
"""


def test_setup_graceful_shutdown_terminates_process_on_sigterm() -> None:
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _TERMINATION_SCRIPT.format(signal_name="SIGTERM")],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert "STILL ALIVE" not in result.stdout
    assert result.returncode == 0  # SystemExit(0) on SIGTERM


def test_setup_graceful_shutdown_terminates_process_on_sigint() -> None:
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _TERMINATION_SCRIPT.format(signal_name="SIGINT")],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert "STILL ALIVE" not in result.stdout
    assert result.returncode != 0  # uncaught KeyboardInterrupt on SIGINT
