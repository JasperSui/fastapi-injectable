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
