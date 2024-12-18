import asyncio
import threading
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

T = TypeVar("T")


def run_coroutine_sync(coro: Coroutine[Any, Any, T], *, timeout: float = 30) -> T:
    """Synchronously run an async coroutine, with support for both main and non-main threads.

    Args:
        coro: The coroutine to execute.
        timeout: Timeout for execution when running in a thread pool.

    Returns:
        The result of the coroutine execution.

    Raises:
        Any exception raised by the coroutine or during execution.

    Notes:
        - In the main thread, if the event loop is running, a new thread is used to run the coroutine.
        - In non-main threads, asyncio's `run_coroutine_threadsafe` is used for compatibility.
    """

    def execute_in_new_event_loop() -> T:
        """Run the coroutine in a new asyncio event loop."""
        new_loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(new_loop)
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:  # No current event loop
        return asyncio.run(coro)

    if threading.current_thread() is threading.main_thread():
        if not loop.is_running() and not loop.is_closed():
            return loop.run_until_complete(coro)
        # Event loop is already running, use a separate thread
        with ThreadPoolExecutor() as executor:
            future = executor.submit(execute_in_new_event_loop)
            return future.result(timeout=timeout)
    else:
        # For non-main threads
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=timeout)
