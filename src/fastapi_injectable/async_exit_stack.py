import asyncio
from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import Any
from weakref import WeakKeyDictionary

from .concurrency import loop_manager
from .exception import DependencyCleanupError
from .logging import logger


class AsyncExitStackManager:
    def __init__(self) -> None:
        self._stacks: WeakKeyDictionary[Callable[..., Any], AsyncExitStack] = WeakKeyDictionary()
        self._lock = asyncio.Lock()

    async def get_stack(self, func: Callable[..., Any]) -> AsyncExitStack:
        """Retrieve or create a stack and loop for managing async resources.

        Args:
            func: The function to associate with an exit stack

        Returns:
            AsyncExitStack: The exit stack for the given function
        """
        async with self._lock:
            if func not in self._stacks:
                self._stacks[func] = AsyncExitStack()
            return self._stacks[func]

    async def cleanup_stack(self, func: Callable[..., Any], *, raise_exception: bool = False) -> None:
        """Clean up the stack associated with the given function.

        Args:
            func: The function whose exit stack should be cleaned up
            raise_exception: If True, raises DependencyCleanupError when cleanup fails

        Raises:
            DependencyCleanupError: When cleanup fails and raise_exception is True
        """
        if not self._stacks:
            return  # pragma: no cover

        original_func = getattr(func, "__original_func__", func)
        exception_: Exception | None = None
        async with self._lock:
            stack = self._stacks.pop(original_func, None)
            if not stack:
                return  # pragma: no cover
            try:
                if loop_manager.in_loop():
                    await stack.aclose()
                else:
                    loop_manager.run_in_loop(stack.aclose())  # pragma: no cover
            except RuntimeError as e:  # pragma: no cover
                msg = (
                    f"Cannot cleanup stack for {func.__name__} because there is something wrong with the loop. "
                    "Resources may not be properly released."
                )
                logger.warning(msg)
                exception_ = e
            except Exception as e:  # noqa: BLE001 # pragma: no cover
                msg = f"Failed to cleanup stack for {func.__name__}"
                logger.exception(msg)
                exception_ = e

        if exception_ and raise_exception:
            raise DependencyCleanupError(msg) from exception_  # pragma: no cover

    async def cleanup_all_stacks(self, *, raise_exception: bool = False) -> None:
        """Clean up all stacks.

        Args:
            raise_exception: If True, raises DependencyCleanupError when any cleanup fails

        Raises:
            DependencyCleanupError: When any cleanup fails and raise_exception is True
        """
        if not self._stacks:
            return

        exception_: Exception | None = None
        async with self._lock:
            tasks = [stack.aclose() for stack in self._stacks.values()]
            self._stacks.clear()

            if not tasks:
                return  # pragma: no cover

            try:
                if loop_manager.in_loop():
                    await asyncio.gather(*tasks)
                else:
                    loop_manager.run_in_loop(asyncio.gather(*tasks))  # pragma: no cover
            except RuntimeError as e:  # pragma: no cover
                msg = (
                    "Cannot cleanup all stacks because there is something wrong with the loop. "
                    "Resources may not be properly released."
                )
                logger.warning(msg)
                exception_ = e
            except Exception as e:  # noqa: BLE001 # pragma: no cover
                msg = "Failed to cleanup one or more dependency stacks"
                logger.exception(msg)
                exception_ = e

        if exception_ and raise_exception:
            raise DependencyCleanupError(msg) from exception_  # pragma: no cover


async_exit_stack_manager = AsyncExitStackManager()
