import asyncio
from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import Any
from weakref import WeakKeyDictionary


class AsyncExitStackManager:
    def __init__(self) -> None:
        self._stacks: WeakKeyDictionary[Callable[..., Any], AsyncExitStack] = WeakKeyDictionary()
        self._lock = asyncio.Lock()

    async def get_stack(self, func: Callable[..., Any]) -> AsyncExitStack:
        """Retrieve or create a stack for managing async resources."""
        async with self._lock:
            if func not in self._stacks:
                self._stacks[func] = AsyncExitStack()
            return self._stacks[func]

    async def cleanup_stack(self, func: Callable[..., Any]) -> None:
        """Clean up the stack associated with the given func."""
        if not self._stacks:
            return

        # If the func is wrapped by injectable, we need to use the original function to get the stack
        original_func = getattr(func, "__original_func__", func)

        async with self._lock:
            stack = self._stacks.pop(original_func, None)
            if stack:
                await stack.aclose()

    async def cleanup_all_stacks(self) -> None:
        """Clean up all stacks."""
        if not self._stacks:
            return

        async with self._lock:
            tasks = [stack.aclose() for stack in self._stacks.values()]
            self._stacks.clear()
            await asyncio.gather(*tasks)


async_exit_stack_manager = AsyncExitStackManager()
