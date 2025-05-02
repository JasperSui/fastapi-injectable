import asyncio
import atexit
import concurrent.futures
import threading
from collections.abc import Awaitable, Coroutine
from typing import Any, Literal, TypeVar

T = TypeVar("T")


class LoopManager:
    def __init__(self) -> None:
        """Initialize the loop manager."""
        self._loop_strategy: Literal["current", "isolated", "background_thread"] = "current"
        self._shutting_down = False
        self._lock = threading.Lock()
        self._isolated_loop: asyncio.AbstractEventLoop | None = None
        self._background_loop: asyncio.AbstractEventLoop | None = None
        self._background_loop_thread: threading.Thread | None = None
        self._background_loop_result_timeout = 30.0
        self._background_loop_result_max_retries = 5

    def _get_background_loop(self) -> asyncio.AbstractEventLoop:
        """Get the dedicated background loop, starting it if necessary."""
        if self._background_loop is not None:
            return self._background_loop
        with self._lock:
            self._background_loop = asyncio.new_event_loop()
            self._background_loop_thread = threading.Thread(
                target=self._run_background_loop, daemon=True, name="fastapi-injectable-daemon-thread"
            )
            self._background_loop_thread.start()
        return self._background_loop

    def _run_background_loop(self) -> None:  # pragma: no cover
        """Run the background loop forever."""
        assert self._background_loop is not None  # noqa: S101
        asyncio.set_event_loop(self._background_loop)
        try:
            self._background_loop.run_forever()
        finally:
            if not self._shutting_down:
                self._background_loop.close()

    def _get_isolated_loop(self) -> asyncio.AbstractEventLoop:
        """Get the isolated loop, creating it if necessary."""
        if self._isolated_loop is not None:
            return self._isolated_loop
        with self._lock:
            self._isolated_loop = asyncio.new_event_loop()
        return self._isolated_loop

    @property
    def loop_strategy(self) -> Literal["current", "isolated", "background_thread"]:
        """Get the current setting for whether to use the current loop."""
        return self._loop_strategy

    def set_loop_strategy(self, strategy: Literal["current", "isolated", "background_thread"]) -> None:
        """Set the current setting for whether to use the current loop."""
        with self._lock:
            self._loop_strategy = strategy

    def get_loop(self) -> asyncio.AbstractEventLoop:
        """Get the appropriate event loop based on the configured strategy.

        Returns:
            The appropriate event loop based on strategy or input.

        Raises:
            RuntimeError: If using 'current' strategy and no loop exists in current thread.
        """
        if self._loop_strategy == "current":
            # Use the current thread's event loop
            return asyncio.get_event_loop()

        if self._loop_strategy == "isolated":
            # Get or create isolated loop
            return self._get_isolated_loop()

        # "background_thread"
        # Get or create background thread loop
        return self._get_background_loop()

    def _wait_with_retries(self, future: concurrent.futures.Future[T]) -> T:
        retries = 0
        max_retries = self._background_loop_result_max_retries
        timeout = self._background_loop_result_timeout

        while retries < max_retries:
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:  # noqa: PERF203
                retries += 1

        future.cancel()
        msg = f"Operation timed out after {max_retries} attempts " f"(total {max_retries * timeout} seconds)"
        raise TimeoutError(msg)

    def in_loop(self) -> bool:
        loop = self.get_loop()
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
        return running_loop is loop

    def run_in_loop(self, awaitable: Awaitable[T]) -> T:
        """Run coroutine in the appropriate loop.

        Args:
            awaitable: The awaitable to execute.

        Returns:
            The result of the awaitable execution.
        """
        loop = self.get_loop()

        if self._loop_strategy in {"current", "isolated"}:
            return loop.run_until_complete(awaitable)

        if asyncio.iscoroutine(awaitable):
            coro: Coroutine[Any, Any, T] = awaitable  # pragma: no cover
        else:
            # Prepare coroutine for run_coroutine_threadsafe
            async def _wrapper() -> T:
                return await awaitable  # pragma: no cover

            coro = _wrapper()

        return self._wait_with_retries(asyncio.run_coroutine_threadsafe(coro, loop))

    def shutdown(self) -> None:
        with self._lock:
            self._shutting_down = True

            if self._loop_strategy == "background_thread":
                assert self._background_loop is not None  # noqa: S101
                assert self._background_loop_thread is not None  # noqa: S101
                self._background_loop.call_soon_threadsafe(self._background_loop.stop)
                self._background_loop_thread.join(timeout=1)
                self._background_loop.close()

            if self._loop_strategy == "isolated" and self._isolated_loop and not self._isolated_loop.is_closed():
                self._isolated_loop.close()


loop_manager = LoopManager()
atexit.register(loop_manager.shutdown)


def run_coroutine_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine synchronously.

    Args:
        coro: The coroutine to execute.

    Returns:
        The result of the coroutine execution.

    Raises:
        Any exception raised by the coroutine or during execution.
    """
    return loop_manager.run_in_loop(
        coro
    )  # NOTE(Jasper Sui): This can run not only coroutine, but also Future and Task.
