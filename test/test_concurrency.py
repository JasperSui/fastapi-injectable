import asyncio
import concurrent.futures
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.fastapi_injectable.concurrency import LoopManager, loop_manager, run_coroutine_sync


@pytest.fixture
def loop_manager_instance() -> LoopManager:
    """Create a fresh LoopManager instance for testing."""
    return LoopManager()


def test_init_default_values(loop_manager_instance: LoopManager) -> None:
    """Test that the LoopManager initializes with the expected default values."""
    assert loop_manager_instance._loop_strategy == "current"
    assert loop_manager_instance._shutting_down is False
    assert loop_manager_instance._lock is not None
    assert loop_manager_instance._isolated_loop is None
    assert loop_manager_instance._background_loop is None
    assert loop_manager_instance._background_loop_thread is None
    assert loop_manager_instance._background_loop_result_timeout == 30.0
    assert loop_manager_instance._background_loop_result_max_retries == 5


def test_loop_strategy_property(loop_manager_instance: LoopManager) -> None:
    """Test the loop_strategy property."""
    loop_manager_instance.set_loop_strategy("current")
    assert loop_manager_instance.loop_strategy == "current"

    loop_manager_instance.set_loop_strategy("isolated")
    assert loop_manager_instance.loop_strategy == "isolated"  # type: ignore[comparison-overlap]

    loop_manager_instance.set_loop_strategy("background_thread")  # type: ignore[unreachable]
    assert loop_manager_instance.loop_strategy == "background_thread"


@patch("src.fastapi_injectable.concurrency.asyncio.get_running_loop")
def test_get_loop_current_strategy(mock_get_running_loop: Mock, loop_manager_instance: LoopManager) -> None:
    """Test get_loop with 'current' strategy."""
    mock_loop = Mock()
    mock_get_running_loop.return_value = mock_loop
    loop_manager_instance.set_loop_strategy("current")

    result = loop_manager_instance.get_loop()

    mock_get_running_loop.assert_called_once()
    assert result == mock_loop


def test_get_loop_current_strategy_fallback(loop_manager_instance: LoopManager) -> None:
    """Test get_loop with 'current' strategy when no running loop exists (Python 3.14+ compatibility)."""
    mock_loop = Mock()
    mock_policy = Mock()
    mock_policy.get_event_loop.return_value = mock_loop

    with (
        patch("src.fastapi_injectable.concurrency.asyncio.get_running_loop", side_effect=RuntimeError),
        patch("src.fastapi_injectable.concurrency.asyncio.get_event_loop_policy", return_value=mock_policy),
    ):
        loop_manager_instance.set_loop_strategy("current")

        result = loop_manager_instance.get_loop()

        assert result == mock_loop
        mock_policy.get_event_loop.assert_called_once()


def test_get_loop_current_strategy_fallback_create_new(loop_manager_instance: LoopManager) -> None:
    """Test get_loop with 'current' strategy when policy.get_event_loop() raises RuntimeError (Python 3.14+)."""
    mock_loop = Mock()
    mock_policy = Mock()
    mock_policy.get_event_loop.side_effect = RuntimeError
    mock_policy.new_event_loop.return_value = mock_loop

    with (
        patch("src.fastapi_injectable.concurrency.asyncio.get_running_loop", side_effect=RuntimeError),
        patch("src.fastapi_injectable.concurrency.asyncio.get_event_loop_policy", return_value=mock_policy),
    ):
        loop_manager_instance.set_loop_strategy("current")

        result = loop_manager_instance.get_loop()

        assert result == mock_loop
        mock_policy.get_event_loop.assert_called_once()
        mock_policy.new_event_loop.assert_called_once()


def test_get_loop_isolated_strategy(loop_manager_instance: LoopManager) -> None:
    """Test get_loop with 'isolated' strategy."""
    mock_loop = Mock()
    mock_policy = Mock()
    mock_policy.new_event_loop.return_value = mock_loop

    with patch("src.fastapi_injectable.concurrency.asyncio.get_event_loop_policy", return_value=mock_policy):
        loop_manager_instance.set_loop_strategy("isolated")

        # First call should create a new loop
        result1 = loop_manager_instance.get_loop()
        assert result1 == mock_loop
        mock_policy.new_event_loop.assert_called_once()

        # Second call should reuse the existing loop
        mock_policy.new_event_loop.reset_mock()
        result2 = loop_manager_instance.get_loop()
        assert result2 == mock_loop
        mock_policy.new_event_loop.assert_not_called()


@patch("src.fastapi_injectable.concurrency.threading.Thread")
@patch("src.fastapi_injectable.concurrency.asyncio.new_event_loop")
def test_get_loop_background_thread_strategy(
    mock_new_event_loop: Mock, mock_thread: Mock, loop_manager_instance: LoopManager
) -> None:
    """Test get_loop with 'background_thread' strategy."""
    mock_loop = Mock()
    mock_new_event_loop.return_value = mock_loop
    mock_thread_instance = Mock()
    mock_thread.return_value = mock_thread_instance

    loop_manager_instance.set_loop_strategy("background_thread")

    # First call should create a new loop and start a thread
    result1 = loop_manager_instance.get_loop()
    assert result1 == mock_loop
    mock_new_event_loop.assert_called_once()
    mock_thread.assert_called_once_with(
        target=loop_manager_instance._run_background_loop, daemon=True, name="fastapi-injectable-daemon-thread"
    )
    mock_thread_instance.start.assert_called_once()

    # Second call should reuse the existing loop
    mock_new_event_loop.reset_mock()
    mock_thread.reset_mock()
    result2 = loop_manager_instance.get_loop()
    assert result2 == mock_loop
    mock_new_event_loop.assert_not_called()
    mock_thread.assert_not_called()


@patch("src.fastapi_injectable.concurrency.asyncio.get_running_loop")
def test_in_loop_true(mock_get_running_loop: Mock, loop_manager_instance: LoopManager) -> None:
    """Test in_loop when the loop is the current running loop."""
    mock_loop = Mock()
    with patch.object(loop_manager_instance, "get_loop", return_value=mock_loop):
        mock_get_running_loop.return_value = mock_loop
        assert loop_manager_instance.in_loop() is True


@patch("src.fastapi_injectable.concurrency.asyncio.get_running_loop")
def test_in_loop_false_different_loop(mock_get_running_loop: Mock, loop_manager_instance: LoopManager) -> None:
    """Test in_loop when the loop is not the current running loop."""
    with patch.object(loop_manager_instance, "get_loop", return_value=Mock()):
        mock_get_running_loop.return_value = Mock()  # Different loop
        assert loop_manager_instance.in_loop() is False


@patch("src.fastapi_injectable.concurrency.asyncio.get_running_loop")
def test_in_loop_false_no_running_loop(mock_get_running_loop: Mock, loop_manager_instance: LoopManager) -> None:
    """Test in_loop when there is no running loop."""
    mock_get_running_loop.side_effect = RuntimeError("No running event loop")
    assert loop_manager_instance.in_loop() is False


def test_run_in_loop_current_strategy(loop_manager_instance: LoopManager) -> None:
    """Test run_in_loop with 'current' strategy."""
    mock_loop = Mock()
    mock_coro = AsyncMock()

    with patch.object(loop_manager_instance, "get_loop", return_value=mock_loop):
        loop_manager_instance.set_loop_strategy("current")
        loop_manager_instance.run_in_loop(mock_coro)

        mock_loop.run_until_complete.assert_called_once_with(mock_coro)


def test_run_in_loop_isolated_strategy(loop_manager_instance: LoopManager) -> None:
    """Test run_in_loop with 'isolated' strategy."""
    mock_loop = Mock()
    mock_coro = AsyncMock()

    with patch.object(loop_manager_instance, "get_loop", return_value=mock_loop):
        loop_manager_instance.set_loop_strategy("isolated")
        loop_manager_instance.run_in_loop(mock_coro)

        mock_loop.run_until_complete.assert_called_once_with(mock_coro)


@patch("src.fastapi_injectable.concurrency.asyncio.run_coroutine_threadsafe")
def test_run_in_loop_background_thread_strategy(
    mock_run_coro_threadsafe: Mock, loop_manager_instance: LoopManager
) -> None:
    """Test run_in_loop with 'background_thread' strategy."""
    mock_loop = Mock()
    mock_coro = AsyncMock()
    mock_future = Mock()
    mock_run_coro_threadsafe.return_value = mock_future

    with patch.object(loop_manager_instance, "get_loop", return_value=mock_loop):  # noqa: SIM117
        with patch.object(loop_manager_instance, "_wait_with_retries") as mock_wait:
            loop_manager_instance.set_loop_strategy("background_thread")
            loop_manager_instance.run_in_loop(mock_coro)

            # Instead of checking exact arguments, just verify it was called once
            # and the second argument is the loop
            mock_run_coro_threadsafe.assert_called_once()
            assert mock_run_coro_threadsafe.call_args[0][1] == mock_loop
            mock_wait.assert_called_once_with(mock_future)


@patch("src.fastapi_injectable.concurrency.asyncio.run_coroutine_threadsafe")
@patch("src.fastapi_injectable.concurrency.asyncio.iscoroutine")
def test_run_in_loop_background_thread_strategy_with_gather(
    mock_iscoroutine: Mock, mock_run_coro_threadsafe: Mock, loop_manager_instance: LoopManager
) -> None:
    """Test run_in_loop with 'background_thread' strategy using asyncio.gather."""
    # Setup
    mock_loop = Mock()
    mock_gather = Mock(spec=asyncio.Future)  # Mock the gather result as a non-coroutine awaitable
    mock_future = Mock()
    mock_wrapper_coro = Mock()  # This will be our wrapper coroutine

    # Configure mocks
    mock_iscoroutine.return_value = False  # Make it treat our mock_gather as a non-coroutine
    mock_run_coro_threadsafe.return_value = mock_future

    with patch.object(loop_manager_instance, "get_loop", return_value=mock_loop):  # noqa: SIM117
        with patch.object(loop_manager_instance, "_wait_with_retries") as mock_wait:
            with patch(
                "asyncio.coroutines._is_coroutine", return_value=True
            ):  # Make our wrapper recognized as a coroutine
                # Run the method under test
                loop_manager_instance.set_loop_strategy("background_thread")

                # Mock the wrapper coroutine creation inside run_in_loop
                def side_effect(coro: Mock, loop: asyncio.AbstractEventLoop) -> Any:  # noqa: ANN401
                    # Capture the wrapper coroutine that was created
                    nonlocal mock_wrapper_coro
                    # Verify this is not our original mock_gather but a wrapper
                    assert coro is not mock_gather
                    mock_wrapper_coro = coro
                    return mock_future

                mock_run_coro_threadsafe.side_effect = side_effect

                # Call the method with our gather-like object
                loop_manager_instance.run_in_loop(mock_gather)

                # Verify a wrapper coroutine was created and passed to run_coroutine_threadsafe
                mock_run_coro_threadsafe.assert_called_once()
                assert mock_run_coro_threadsafe.call_args[0][1] == mock_loop

                # Verify _wait_with_retries was called with the future
                mock_wait.assert_called_once_with(mock_future)


def test_wait_with_retries_success(loop_manager_instance: LoopManager) -> None:
    """Test _wait_with_retries with successful future."""
    mock_future = Mock(spec=concurrent.futures.Future)
    mock_future.result.return_value = "test_result"

    result: str = loop_manager_instance._wait_with_retries(mock_future)

    assert result == "test_result"
    mock_future.result.assert_called_once_with(timeout=30.0)


def test_wait_with_retries_timeout(loop_manager_instance: LoopManager) -> None:
    """Test _wait_with_retries with timeout."""
    mock_future = Mock(spec=concurrent.futures.Future)
    mock_future.result.side_effect = concurrent.futures.TimeoutError()

    # Override retries for faster test
    loop_manager_instance._background_loop_result_max_retries = 2
    loop_manager_instance._background_loop_result_timeout = 0.1

    with pytest.raises(TimeoutError):
        loop_manager_instance._wait_with_retries(mock_future)

    assert mock_future.result.call_count == 2
    mock_future.cancel.assert_called_once()


@patch("src.fastapi_injectable.concurrency.asyncio.get_event_loop")
def test_shutdown_isolated_strategy(mock_get_event_loop: Mock, loop_manager_instance: LoopManager) -> None:
    """Test shutdown with isolated strategy."""
    mock_isolated_loop = Mock()
    mock_isolated_loop.is_closed.return_value = False

    loop_manager_instance.set_loop_strategy("isolated")
    loop_manager_instance._isolated_loop = mock_isolated_loop

    loop_manager_instance.shutdown()

    assert loop_manager_instance._shutting_down is True
    mock_isolated_loop.close.assert_called_once()


@patch("src.fastapi_injectable.concurrency.asyncio.get_event_loop")
def test_shutdown_background_thread_strategy(mock_get_event_loop: Mock, loop_manager_instance: LoopManager) -> None:
    """Test shutdown with background_thread strategy."""
    mock_bg_loop = Mock()
    mock_bg_thread = Mock()

    loop_manager_instance.set_loop_strategy("background_thread")
    loop_manager_instance._background_loop = mock_bg_loop
    loop_manager_instance._background_loop_thread = mock_bg_thread

    loop_manager_instance.shutdown()

    assert loop_manager_instance._shutting_down is True
    mock_bg_loop.call_soon_threadsafe.assert_called_once_with(mock_bg_loop.stop)
    mock_bg_thread.join.assert_called_once_with(timeout=1)
    mock_bg_loop.close.assert_called_once()


@patch("src.fastapi_injectable.concurrency.asyncio.get_event_loop")
def test_shutdown_no_loop_initialized(mock_get_event_loop: Mock, loop_manager_instance: LoopManager) -> None:
    """Test shutdown when no loop has been initialized."""
    # This test ensures no exception is raised when loops are not initialized
    loop_manager_instance.shutdown()
    assert loop_manager_instance._shutting_down is True


def test_run_coroutine_sync() -> None:
    """Test run_coroutine_sync function."""
    mock_coro = AsyncMock()

    with patch.object(loop_manager, "run_in_loop") as mock_run_in_loop:
        mock_run_in_loop.return_value = "test_result"
        result: str = run_coroutine_sync(mock_coro)

        assert result == "test_result"
        mock_run_in_loop.assert_called_once_with(mock_coro)


def test_run_coroutine_sync_real() -> None:
    """Test run_coroutine_sync with a real coroutine."""

    async def test_coro() -> str:
        return "test_result"

    result = run_coroutine_sync(test_coro())
    assert result == "test_result"


def test_run_coroutine_sync_success() -> None:
    async def coro() -> str:
        return "test"

    assert run_coroutine_sync(coro()) == "test"
