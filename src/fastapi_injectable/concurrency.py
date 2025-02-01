import asyncio
import atexit
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

from fastapi_injectable.exception import RunCoroutineSyncMaxRetriesError

T = TypeVar("T")


class LoopManager:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._shutting_down = False

    def get_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None or self._loop.is_closed():
                self.start()
            assert self._loop is not None  # noqa: S101
            return self._loop

    def start(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="async-util-loop")
        self._thread.start()

    async def run_in_loop(self, coro: Coroutine[Any, Any, T] | asyncio.Future[T]) -> T:
        """Run coroutine in the managed loop.

        Args:
            coro: Coroutine to run

        Returns:
            Result of the coroutine
        """
        current_loop = asyncio.get_event_loop()
        loop = self.get_loop()
        try:
            asyncio.set_event_loop(loop)
            return await coro
        finally:
            asyncio.set_event_loop(current_loop)

    def _run_loop(self) -> None:
        assert self._loop is not None  # noqa: S101
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            if not self._shutting_down:
                self._loop.close()

    def shutdown(self) -> None:
        with self._lock:
            self._shutting_down = True
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=1)
            if self._loop and not self._loop.is_closed():
                self._loop.close()


loop_manager = LoopManager()
atexit.register(loop_manager.shutdown)


def run_coroutine_sync(
    coro: Coroutine[Any, Any, T], *, timeout: float = 30, retries: int = 1, max_retries: int = 5
) -> T:
    """Synchronously run an async coroutine, with support for both main and non-main threads.

    Args:
        coro: The coroutine to execute.
        timeout: Timeout for execution when running in a thread pool.
        retries: Number of retries to run the coroutine.
        max_retries: Maximum number of retries.

    Returns:
        The result of the coroutine execution.

    Raises:
        Any exception raised by the coroutine or during execution.

    Notes:
        - In the main thread, if the event loop is running, a new thread is used to run the coroutine.
        - In non-main threads, asyncio's `run_coroutine_threadsafe` is used for compatibility.
    """
    if retries > max_retries:
        msg = f"Maximum retries ({max_retries}) reached while running coroutine."
        raise RunCoroutineSyncMaxRetriesError(msg)

    try:
        loop = loop_manager.get_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout)
    except RuntimeError as e:
        if "Event loop is closed" in str(e):
            loop_manager.shutdown()
            loop_manager.start()
            return run_coroutine_sync(coro, timeout=timeout, retries=retries + 1, max_retries=max_retries)
        raise
