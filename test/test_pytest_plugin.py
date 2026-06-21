from contextlib import AsyncExitStack
from typing import Any
from unittest.mock import Mock
from weakref import WeakKeyDictionary

import pytest

from src.fastapi_injectable import pytest_plugin
from src.fastapi_injectable.async_exit_stack import async_exit_stack_manager
from src.fastapi_injectable.cache import dependency_cache
from src.fastapi_injectable.concurrency import loop_manager


def test_reset_injectable_state_clears_cache_and_stacks() -> None:
    def marker() -> None:
        """A stand-in provider key for the exit-stack manager."""

    dependency_cache.get()[("leaked",)] = object()
    # Seed the per-loop registry under the loop reset() will run cleanup on.
    per_loop: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()
    per_loop[marker] = AsyncExitStack()
    async_exit_stack_manager._stacks[loop_manager.get_loop()] = per_loop
    assert dependency_cache.get()
    assert async_exit_stack_manager._stacks

    pytest_plugin.reset_injectable_state()

    assert dependency_cache.get() == {}
    assert dict(async_exit_stack_manager._stacks) == {}


def test_addoption_registers_autouse_ini() -> None:
    parser = Mock()

    pytest_plugin.pytest_addoption(parser)

    parser.addini.assert_called_once()
    (name,), kwargs = parser.addini.call_args
    assert name == "injectable_autouse_cleanup"
    assert kwargs["type"] == "bool"
    assert kwargs["default"] is True


def test_cleanup_around_test_resets_when_enabled() -> None:
    dependency_cache.get()[("before",)] = object()

    gen = pytest_plugin._cleanup_around_test(enabled=True)
    next(gen)  # runs the "before" reset
    assert dependency_cache.get() == {}

    dependency_cache.get()[("during",)] = object()
    with pytest.raises(StopIteration):
        next(gen)  # runs the "after" reset
    assert dependency_cache.get() == {}


def test_cleanup_around_test_noop_when_disabled() -> None:
    sentinel = object()
    dependency_cache.get()[("kept",)] = sentinel

    gen = pytest_plugin._cleanup_around_test(enabled=False)
    next(gen)
    # Disabled: state is left untouched both before and after the yield.
    assert dependency_cache.get().get(("kept",)) is sentinel
    with pytest.raises(StopIteration):
        next(gen)
    assert dependency_cache.get().get(("kept",)) is sentinel

    dependency_cache.get().clear()


def test_injectable_cleanup_fixture_provides_clean_state(injectable_cleanup: None) -> None:
    # Requesting the shipped fixture runs its "before" reset, so the test starts clean.
    assert dependency_cache.get() == {}


def test_plugin_prevents_cross_test_leakage(pytester: pytest.Pytester) -> None:
    # The plugin auto-loads in the subprocess via its pytest11 entry point.
    pytester.makepyfile(
        """
        from fastapi_injectable.cache import dependency_cache

        def test_a_populates_cache() -> None:
            dependency_cache.get()[("leaked",)] = object()
            assert dependency_cache.get()

        def test_b_starts_clean() -> None:
            # The autouse cleanup wiped test_a's leftover global state.
            assert dependency_cache.get() == {}
        """
    )

    result = pytester.runpytest_subprocess()

    result.assert_outcomes(passed=2)
