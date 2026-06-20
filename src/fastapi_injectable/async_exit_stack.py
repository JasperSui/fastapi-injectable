import asyncio
import threading
from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import Any
from weakref import WeakKeyDictionary

from .concurrency import loop_manager
from .exception import DependencyCleanupError
from .logging import logger

_Loop = asyncio.AbstractEventLoop


class AsyncExitStackManager:
    """Per-event-loop registry of dependency exit stacks.

    A stack is entered while resolving a dependency on a particular event loop,
    and it owns that dependency's async-generator teardowns. Those teardowns may
    ``await`` loop-bound primitives (an ``asyncio.Future``/``Event``/``Lock``, an
    ``httpx``/``anyio`` connection pool, an ``asyncpg`` connection, ...). Such a
    stack must therefore be closed on the *same* loop it was entered on: closing
    it on a different loop awaits a loop-A primitive from loop B, which on Python
    3.13+ raises ``RuntimeError: ... attached to a different loop`` and on older
    runtimes (or behind a no-guard suspend point) hangs forever. That is the
    exit-stack half of https://github.com/JasperSui/fastapi-injectable/issues/186.

    The registry is keyed first by the owning loop (weakly, so a loop's stacks
    disappear once the loop is gone) and then by the provider function (weakly, to
    preserve provider-garbage-collection cleanup). Cleanup dispatches each stack's
    ``aclose()`` back onto its owning loop; a stack whose loop has already died is
    dropped (its resources are dead anyway) rather than closed cross-loop.
    """

    def __init__(self) -> None:
        self._stacks: WeakKeyDictionary[_Loop, WeakKeyDictionary[Callable[..., Any], AsyncExitStack]] = (
            WeakKeyDictionary()
        )
        # A plain threading.Lock (not an asyncio.Lock). The lock guards only the
        # synchronous registry mutations below; every ``await stack.aclose()`` runs
        # OUTSIDE the lock (the stack is popped from the registry first), so the
        # lock never spans an await -- it can never block a loop, and it works
        # across loops and free-threaded builds. An asyncio.Lock would instead bind
        # to the first loop that awaited it and reject the other concurrently-live
        # loop that drives cross-loop cleanup.
        self._lock = threading.Lock()

    @staticmethod
    def _running_loop() -> _Loop | None:
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return None

    def _owning_loop(self) -> _Loop:
        """The loop a stack created *now* belongs to (the loop resolution runs on)."""
        running = self._running_loop()
        if running is not None:
            return running
        return loop_manager.get_loop()

    async def get_stack(self, func: Callable[..., Any]) -> AsyncExitStack:
        """Retrieve or create the exit stack for ``func`` on the current event loop.

        Args:
            func: The function to associate with an exit stack

        Returns:
            AsyncExitStack: The exit stack for the given function on this loop
        """
        loop = self._owning_loop()
        with self._lock:
            per_loop = self._stacks.get(loop)
            if per_loop is None:
                per_loop = WeakKeyDictionary()
                self._stacks[loop] = per_loop
            stack = per_loop.get(func)
            if stack is None:
                stack = AsyncExitStack()
                per_loop[func] = stack
            return stack

    async def _close_stack(self, loop: _Loop, stack: AsyncExitStack) -> None:
        """Close one stack on its owning loop.

        - owning loop is the current running loop -> close it here directly;
        - owning loop is alive and running in another thread -> dispatch the close
          onto it and await the result without blocking this loop;
        - owning loop is closed or no longer running -> its async-generator
          resources are already dead; drop the stack instead of closing it
          cross-loop (which would hang or raise "attached to a different loop").
        """
        running = self._running_loop()
        if loop is running:
            await stack.aclose()
            return
        if loop.is_closed() or not loop.is_running():
            return
        future = asyncio.run_coroutine_threadsafe(stack.aclose(), loop)
        await asyncio.wrap_future(future)

    async def cleanup_stack(self, func: Callable[..., Any], *, raise_exception: bool = False) -> None:
        """Clean up the stack(s) associated with the given function.

        Args:
            func: The function whose exit stack should be cleaned up
            raise_exception: If True, raises DependencyCleanupError when cleanup fails

        Raises:
            DependencyCleanupError: When cleanup fails and raise_exception is True
        """
        original_func = getattr(func, "__original_func__", func)

        with self._lock:
            entries: list[tuple[_Loop, AsyncExitStack]] = []
            for _loop, per_loop in self._stacks.items():
                stack = per_loop.pop(original_func, None)
                if stack is not None:
                    entries.append((_loop, stack))
            self._prune_empty_locked()

        if not entries:
            return

        exception_: Exception | None = None
        msg = ""
        try:
            await asyncio.gather(*(self._close_stack(loop, stack) for loop, stack in entries))
        except RuntimeError as e:
            msg = f"Failed to cleanup stack for {func.__name__} during teardown: {e}"
            logger.warning(msg)
            exception_ = e
        except Exception as e:  # noqa: BLE001
            msg = f"Failed to cleanup stack for {func.__name__}"
            logger.exception(msg)
            exception_ = e

        if exception_ is not None and raise_exception:
            raise DependencyCleanupError(msg) from exception_

    async def cleanup_all_stacks(self, *, raise_exception: bool = False) -> None:
        """Clean up all stacks, each on its own owning loop.

        Args:
            raise_exception: If True, raises DependencyCleanupError when any cleanup fails

        Raises:
            DependencyCleanupError: When any cleanup fails and raise_exception is True
        """
        with self._lock:
            entries = [(loop, stack) for loop, per_loop in self._stacks.items() for stack in per_loop.values()]
            self._stacks.clear()

        if not entries:
            return

        exception_: Exception | None = None
        msg = ""
        try:
            await asyncio.gather(*(self._close_stack(loop, stack) for loop, stack in entries))
        except RuntimeError as e:
            msg = f"Failed to cleanup one or more dependency stacks during teardown: {e}"
            logger.warning(msg)
            exception_ = e
        except Exception as e:  # noqa: BLE001
            msg = "Failed to cleanup one or more dependency stacks"
            logger.exception(msg)
            exception_ = e

        if exception_ is not None and raise_exception:
            raise DependencyCleanupError(msg) from exception_

    def _prune_empty_locked(self) -> None:
        """Drop now-empty per-loop sub-registries. Caller must hold ``self._lock``."""
        empty = [loop for loop, per_loop in self._stacks.items() if not per_loop]
        for loop in empty:
            del self._stacks[loop]


async_exit_stack_manager = AsyncExitStackManager()
