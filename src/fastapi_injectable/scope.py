from __future__ import annotations

import asyncio
import contextvars
import threading
from contextlib import AsyncExitStack, asynccontextmanager
from typing import TYPE_CHECKING, Any
from weakref import WeakKeyDictionary

from .concurrency import loop_manager

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import TracebackType

    from typing_extensions import Self

_Loop = asyncio.AbstractEventLoop

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

    The cache is partitioned by event loop. A scope object reused across event
    loops -- resolved on loop A then on loop B while both are alive -- must never
    serve a loop-A resource to loop B: a cached value commonly holds a loop-bound
    primitive (an ``asyncio`` future/lock, an ``httpx``/``anyio`` pool, an
    ``asyncpg`` connection), and awaiting it on a different loop blocks forever or
    raises "attached to a different loop". This mirrors the global dependency
    cache fix behind https://github.com/JasperSui/fastapi-injectable/issues/186.
    A scope used on a single loop caches exactly as before.

    The exit stack is a single stack: its lifecycle is bounded by ``async with``
    (or an explicit ``scope.exit_stack`` the caller manages), so it is always
    closed on the same loop it was populated on. Partitioning it per loop would
    break that explicit single-stack contract (e.g. resolving under
    ``get_injected_obj(..., scope=scope)`` and then closing ``scope.exit_stack``
    from sync code), so the stack stays whole.
    """

    def __init__(self, *, use_cache: bool = True) -> None:
        self._exit_stack = AsyncExitStack()
        self._use_cache = use_cache
        # One cache dict per event loop, keyed weakly so a loop's cache disappears
        # once the loop is gone. ``None`` semantics (caching disabled) are kept via
        # the ``_use_cache`` flag rather than a nullable dict.
        self._caches: WeakKeyDictionary[_Loop, dict[Any, Any]] = WeakKeyDictionary()
        self._token: contextvars.Token[InjectableScope | None] | None = None
        # A plain threading.Lock guarding only the synchronous get-or-create of the
        # per-loop cache below; it never spans an await.
        self._lock = threading.Lock()

    def _current_loop(self) -> _Loop:
        """The loop whose cache this scope should use right now."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return loop_manager.get_loop()

    @property
    def exit_stack(self) -> AsyncExitStack:
        """The scope's exit stack.

        Push your own cleanup callbacks here to ride the same lifecycle as the
        scope's dependencies.
        """
        return self._exit_stack

    def get_cache(self) -> dict[Any, Any] | None:
        """The scope's dependency cache for the current loop (``None`` when caching is disabled)."""
        if not self._use_cache:
            return None
        loop = self._current_loop()
        with self._lock:
            cache = self._caches.get(loop)
            if cache is None:
                cache = {}
                self._caches[loop] = cache
            return cache

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
