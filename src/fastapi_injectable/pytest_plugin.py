"""Pytest plugin that keeps fastapi-injectable's process-global state isolated between tests.

fastapi-injectable keeps process-global singletons -- the dependency cache
(:data:`fastapi_injectable.cache.dependency_cache`) and the exit-stack manager
(:data:`fastapi_injectable.async_exit_stack.async_exit_stack_manager`). Because they
outlive any single test, a value cached -- or a generator left open -- by one test can
leak into the next. This plugin resets both so that "tests are isolated" is the default.

Installing fastapi-injectable registers this plugin automatically (via the ``pytest11``
entry point). The reset runs as an autouse fixture, enabled by default; disable it with::

    [tool.pytest.ini_options]
    injectable_autouse_cleanup = false

and request the :func:`injectable_cleanup` fixture explicitly where you still want it.
"""

from collections.abc import Iterator

import pytest

from .concurrency import run_coroutine_sync
from .util import cleanup_all_exit_stacks, clear_dependency_cache

_AUTOUSE_INI = "injectable_autouse_cleanup"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``injectable_autouse_cleanup`` ini option (enabled by default)."""
    parser.addini(
        _AUTOUSE_INI,
        help=(
            "Reset fastapi-injectable's global dependency cache and exit stacks around "
            "every test (default: true). Set to false to manage isolation yourself with "
            "the `injectable_cleanup` fixture."
        ),
        type="bool",
        default=True,
    )


async def _areset_injectable_state() -> None:
    await cleanup_all_exit_stacks()
    await clear_dependency_cache()


def reset_injectable_state() -> None:
    """Reset fastapi-injectable's process-global cache and exit stacks.

    Wrapped with :func:`run_coroutine_sync` so it works from sync and async tests alike,
    and so it honours whatever loop strategy ``loop_manager`` is configured with.
    """
    run_coroutine_sync(_areset_injectable_state())


def _cleanup_around_test(*, enabled: bool) -> Iterator[None]:
    if enabled:
        reset_injectable_state()
    yield
    if enabled:
        reset_injectable_state()


@pytest.fixture
def injectable_cleanup() -> Iterator[None]:
    """Reset fastapi-injectable's global cache + exit stacks around the test.

    Request this explicitly (e.g. ``pytestmark = pytest.mark.usefixtures("injectable_cleanup")``)
    to isolate a test or module when you have turned the autouse behaviour off.
    """
    yield from _cleanup_around_test(enabled=True)


@pytest.fixture(autouse=True)
def _injectable_autouse_cleanup(request: pytest.FixtureRequest) -> Iterator[None]:
    """Autouse counterpart of :func:`injectable_cleanup`, gated by the ini option."""
    yield from _cleanup_around_test(enabled=request.config.getini(_AUTOUSE_INI))
