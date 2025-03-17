import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def run_coroutine_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine synchronously.

    Args:
        coro: The coroutine to execute.

    Returns:
        The result of the coroutine execution.

    Raises:
        Any exception raised by the coroutine or during execution.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError as e:
        msg = "Failed to get a runnable event loop"
        raise RuntimeError(msg) from e

    return loop.run_until_complete(coro)
