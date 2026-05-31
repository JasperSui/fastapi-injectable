from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack

import pytest

from fastapi_injectable.scope import InjectableScope, _current_scope, injectable_scope
from fastapi_injectable.util import cleanup_all_exit_stacks


@pytest.fixture(autouse=True)
async def _clean_global_stacks() -> AsyncGenerator[None, None]:
    await cleanup_all_exit_stacks()
    yield
    await cleanup_all_exit_stacks()


async def test_injectable_scope_sets_and_resets_active_scope() -> None:
    assert _current_scope.get() is None
    async with injectable_scope() as scope:
        assert isinstance(scope, InjectableScope)
        assert _current_scope.get() is scope
    assert _current_scope.get() is None


async def test_scope_exposes_exit_stack_and_cache() -> None:
    scope = InjectableScope()
    assert isinstance(scope.exit_stack, AsyncExitStack)
    assert scope.get_cache() == {}

    no_cache = InjectableScope(use_cache=False)
    assert no_cache.get_cache() is None


async def test_nested_injectable_scope_restores_outer() -> None:
    async with injectable_scope() as outer:
        assert _current_scope.get() is outer
        async with injectable_scope() as inner:
            assert _current_scope.get() is inner
        assert _current_scope.get() is outer
    assert _current_scope.get() is None


async def test_injectable_scope_resets_contextvar_on_exception() -> None:
    async def _raise_inside_scope() -> None:
        async with injectable_scope():
            assert _current_scope.get() is not None
            msg = "boom"
            raise ValueError(msg)

    with pytest.raises(ValueError, match="boom"):
        await _raise_inside_scope()
    assert _current_scope.get() is None


from typing import Annotated  # noqa: E402

from fastapi import Depends  # noqa: E402

from fastapi_injectable.async_exit_stack import async_exit_stack_manager  # noqa: E402
from fastapi_injectable.util import async_get_injected_obj  # noqa: E402


class Mayor:
    def __init__(self) -> None:
        self._is_cleaned_up = False

    def cleanup(self) -> None:
        self._is_cleaned_up = True


async def test_resolution_inside_scope_cleans_up_on_exit_and_skips_global_manager() -> None:
    async def get_mayor() -> AsyncGenerator[Mayor, None]:
        mayor = Mayor()
        yield mayor
        mayor.cleanup()

    async with injectable_scope():
        mayor = await async_get_injected_obj(get_mayor)
        assert mayor._is_cleaned_up is False
        # nothing leaks into the global, function-keyed manager
        assert len(async_exit_stack_manager._stacks) == 0

    assert mayor._is_cleaned_up is True


import asyncio  # noqa: E402


async def test_parallel_scopes_clean_up_independently() -> None:
    opened: dict[str, Mayor] = {}
    proceed: dict[str, asyncio.Event] = {"a": asyncio.Event(), "b": asyncio.Event()}
    done: dict[str, asyncio.Event] = {"a": asyncio.Event(), "b": asyncio.Event()}
    both_open = asyncio.Event()

    async def get_mayor() -> AsyncGenerator[Mayor, None]:
        mayor = Mayor()
        yield mayor
        mayor.cleanup()

    async def process(name: str) -> None:
        async with injectable_scope():
            mayor = await async_get_injected_obj(get_mayor)
            opened[name] = mayor
            if len(opened) == 2:
                both_open.set()
            await proceed[name].wait()
        done[name].set()  # scope fully closed here

    task_a = asyncio.create_task(process("a"))
    task_b = asyncio.create_task(process("b"))
    try:
        await both_open.wait()
        assert opened["a"]._is_cleaned_up is False
        assert opened["b"]._is_cleaned_up is False
        assert opened["a"] is not opened["b"]  # isolated caches -> distinct instances

        # Let A finish and tear down; B must be unaffected.
        proceed["a"].set()
        await done["a"].wait()
        assert opened["a"]._is_cleaned_up is True
        assert opened["b"]._is_cleaned_up is False  # <- isolation proven

        proceed["b"].set()
        await asyncio.gather(task_a, task_b)
    finally:
        for task in (task_a, task_b):
            if not task.done():
                task.cancel()

    assert opened["a"]._is_cleaned_up is True
    assert opened["b"]._is_cleaned_up is True


class Capital:
    def __init__(self, mayor: Mayor) -> None:
        self.mayor = mayor


async def test_nested_scopes_clean_up_in_lifo_order() -> None:
    async def get_mayor() -> AsyncGenerator[Mayor, None]:
        mayor = Mayor()
        yield mayor
        mayor.cleanup()

    async with injectable_scope():
        outer_mayor = await async_get_injected_obj(get_mayor)
        async with injectable_scope():
            inner_mayor = await async_get_injected_obj(get_mayor)
            assert inner_mayor is not outer_mayor
            assert inner_mayor._is_cleaned_up is False
            assert outer_mayor._is_cleaned_up is False
        # inner scope closed
        assert inner_mayor._is_cleaned_up is True
        assert outer_mayor._is_cleaned_up is False
    assert outer_mayor._is_cleaned_up is True


async def test_in_scope_cache_dedups_shared_subdependency() -> None:
    async def get_mayor() -> AsyncGenerator[Mayor, None]:
        yield Mayor()

    async def get_capital_a(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Capital:
        return Capital(mayor)

    async def get_capital_b(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Capital:
        return Capital(mayor)

    async with injectable_scope():
        a = await async_get_injected_obj(get_capital_a)
        b = await async_get_injected_obj(get_capital_b)
        assert a.mayor is b.mayor  # same scope → shared cache → one Mayor
        first_mayor = a.mayor

    async with injectable_scope():
        c = await async_get_injected_obj(get_capital_a)
        assert c.mayor is not first_mayor  # different scope → fresh cache


async def test_scope_use_cache_false_disables_dedup() -> None:
    async def get_mayor() -> AsyncGenerator[Mayor, None]:
        yield Mayor()

    async def get_capital_a(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Capital:
        return Capital(mayor)

    async def get_capital_b(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Capital:
        return Capital(mayor)

    async with injectable_scope(use_cache=False):
        a = await async_get_injected_obj(get_capital_a, use_cache=False)
        b = await async_get_injected_obj(get_capital_b, use_cache=False)
        assert a.mayor is not b.mayor  # no cache → distinct Mayors


async def test_explicit_scope_routes_resolution_without_owning_lifecycle() -> None:
    async def get_mayor() -> AsyncGenerator[Mayor, None]:
        mayor = Mayor()
        yield mayor
        mayor.cleanup()

    scope = InjectableScope()
    assert _current_scope.get() is None

    # Manage the scope's stack lifecycle by hand (do NOT enter it as the active
    # scope); route a single resolution into it with scope=.
    async with scope.exit_stack:
        mayor = await async_get_injected_obj(get_mayor, scope=scope)
        assert _current_scope.get() is None  # scope= must not leak the ContextVar
        assert mayor._is_cleaned_up is False

    assert mayor._is_cleaned_up is True  # cleaned only when WE closed the stack
