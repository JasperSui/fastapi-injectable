import asyncio
import threading
import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from fastapi_injectable.exception import RunCoroutineSyncMaxRetriesError
from src.fastapi_injectable.concurrency import LoopManager, run_coroutine_sync


async def test_loop_manager_shutdown() -> None:
    manager = LoopManager()
    manager.start()

    # Mock the loop and thread
    mock_loop = Mock()
    mock_loop.is_running.return_value = True
    mock_loop.is_closed.return_value = False
    manager._loop = mock_loop

    # Create and start a dummy thread
    mock_thread = threading.Thread(target=lambda: None)
    mock_thread.start()
    manager._thread = mock_thread

    # Test shutdown
    manager.shutdown()

    assert manager._shutting_down is True
    mock_loop.call_soon_threadsafe.assert_called_once_with(mock_loop.stop)
    mock_loop.close.assert_called_once()


def test_loop_closes_when_not_shutting_down() -> None:
    manager = LoopManager()

    # Mock the loop to track close() calls
    mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
    mock_loop.is_running.return_value = False
    mock_loop.is_closed.return_value = False

    with patch("asyncio.new_event_loop", return_value=mock_loop):
        manager.start()
        # Give the thread a moment to start
        time.sleep(0.1)
        # Force the loop to stop
        mock_loop.run_forever.side_effect = Exception("Force stop")

        # Wait for thread to finish
        assert isinstance(manager._thread, threading.Thread)
        manager._thread.join(timeout=1)

        # Verify loop.close() was called since _shutting_down is False
        mock_loop.close.assert_called_once()


async def test_loop_manager_shutdown_no_running_loop() -> None:
    manager = LoopManager()
    manager.start()

    # Mock the loop that's not running
    mock_loop = Mock()
    mock_loop.is_running.return_value = False
    mock_loop.is_closed.return_value = False
    manager._loop = mock_loop

    # Test shutdown
    manager.shutdown()

    assert manager._shutting_down is True
    mock_loop.call_soon_threadsafe.assert_not_called()
    mock_loop.close.assert_called_once()


def test_loop_manager_shutdown_with_no_thread_or_loop() -> None:
    manager = LoopManager()
    manager.shutdown()  # Should not raise any errors


def test_loop_manager_shutdown_with_running_loop_no_thread() -> None:
    manager = LoopManager()
    manager._loop = Mock()

    manager._loop.is_running.return_value = True
    manager._loop.is_closed.return_value = False
    manager.shutdown()

    assert manager._shutting_down is True
    manager._loop.close.assert_called_once()


@patch("fastapi_injectable.concurrency.asyncio.run_coroutine_threadsafe")
async def test_run_coroutine_sync_max_retries(mock_run_coroutine_threadsafe: Mock) -> None:
    # Mock coroutine that creates a new coroutine each time
    async def mock_coro() -> None:
        pass

    # Make run_coroutine_threadsafe always raise RuntimeError
    mock_run_coroutine_threadsafe.side_effect = RuntimeError("Event loop is closed")

    # Test max retries exceeded
    with pytest.raises(RunCoroutineSyncMaxRetriesError, match="Maximum retries .* reached"):
        run_coroutine_sync(mock_coro(), max_retries=2)


async def test_run_coroutine_sync_other_runtime_error() -> None:
    async def mock_coro() -> None:
        msg = "Some other error"
        raise RuntimeError(msg)

    # Test that other RuntimeErrors are re-raised
    with pytest.raises(RuntimeError, match="Some other error"):
        run_coroutine_sync(mock_coro())
