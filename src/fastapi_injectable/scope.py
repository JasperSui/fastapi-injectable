from __future__ import annotations

import contextvars
from contextlib import AsyncExitStack, asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import TracebackType

    from typing_extensions import Self

_current_scope: contextvars.ContextVar[InjectableScope | None] = contextvars.ContextVar(
    "fastapi_injectable_current_scope", default=None
)


class InjectableScope:
    """An isolated dependency lifecycle: a private exit stack + dependency cache.

    Entering the scope (``async with``) registers it as the active scope for the
    current asyncio context. Any dependency resolution that happens while it is
    active routes its generator cleanup into this scope's exit stack and its
    cached values into this scope's cache. Leaving the scope closes the exit
    stack (running all cleanup) and drops the cache.
    """

    def __init__(self, *, use_cache: bool = True) -> None:
        self._exit_stack = AsyncExitStack()
        self._cache: dict[Any, Any] | None = {} if use_cache else None
        self._token: contextvars.Token[InjectableScope | None] | None = None

    @property
    def exit_stack(self) -> AsyncExitStack:
        """The scope's exit stack.

        Push your own cleanup callbacks here to ride the same lifecycle as the
        scope's dependencies.
        """
        return self._exit_stack

    def get_cache(self) -> dict[Any, Any] | None:
        """The scope's dependency cache (``None`` when caching is disabled)."""
        return self._cache

    async def __aenter__(self) -> Self:
        await self._exit_stack.__aenter__()
        self._token = _current_scope.set(self)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        # Reset the active scope BEFORE closing the stack: this guarantees that
        # any dependency resolution triggered during cleanup never routes back
        # into the stack we are currently tearing down.
        if self._token is not None:
            _current_scope.reset(self._token)
            self._token = None
        return await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)


@asynccontextmanager
async def injectable_scope(*, use_cache: bool = True) -> AsyncIterator[InjectableScope]:
    """Open an isolated dependency scope for the duration of the ``async with`` block.

    Inside the block every dependency resolved (via ``async_get_injected_obj`` or
    by calling an ``@injectable`` function) uses this scope's private exit stack
    and cache. Leaving the block cleans up only this scope's resources, isolated
    from any other concurrent scope.

    Example::

        async def process_event(event):
            async with injectable_scope():
                dep = await async_get_injected_obj(get_dep)
                await handle(event, dep)
            # only this event's resources are cleaned up here
    """
    async with InjectableScope(use_cache=use_cache) as scope:
        yield scope
