"""End-to-end regression tests for https://github.com/JasperSui/fastapi-injectable/issues/255.

When a decorated function raises, generator dependencies must be able to see the
in-flight exception during teardown -- exactly as FastAPI unwinds a request's exit
stack -- so their ``except``/rollback branches run. Two paths provide this:

- ``cleanup_exit_stack_of_func(func, exc=...)`` / ``cleanup_all_exit_stacks(exc=...)``
  for the global, function-keyed stacks;
- ``async with injectable_scope():`` which forwards the exception automatically.
"""

from collections.abc import AsyncGenerator, Callable, Generator
from typing import Annotated

import pytest
from fastapi import Depends

from fastapi_injectable.concurrency import run_coroutine_sync
from fastapi_injectable.decorator import injectable
from fastapi_injectable.scope import injectable_scope
from fastapi_injectable.util import cleanup_all_exit_stacks, cleanup_exit_stack_of_func


class FakeConnection:
    """Tracks which teardown branch of its provider generator ran."""

    def __init__(self) -> None:
        self.committed = False
        self.rolled_back = False
        self.closed = False
        self.seen_exception: BaseException | None = None


def _make_async_provider(conn: FakeConnection) -> Callable[[], AsyncGenerator[FakeConnection, None]]:
    async def get_connection() -> AsyncGenerator[FakeConnection, None]:
        try:
            yield conn
        except Exception as exc:
            conn.rolled_back = True
            conn.seen_exception = exc
            raise
        else:
            conn.committed = True
        finally:
            conn.closed = True

    return get_connection


def _make_sync_provider(conn: FakeConnection) -> Callable[[], Generator[FakeConnection, None, None]]:
    def get_connection() -> Generator[FakeConnection, None, None]:
        try:
            yield conn
        except Exception as exc:
            conn.rolled_back = True
            conn.seen_exception = exc
            raise
        else:
            conn.committed = True
        finally:
            conn.closed = True

    return get_connection


@pytest.fixture(autouse=True)
async def _clean_global_stacks() -> AsyncGenerator[None, None]:
    await cleanup_all_exit_stacks()
    yield
    await cleanup_all_exit_stacks()


async def test_cleanup_all_exit_stacks_with_exc_runs_rollback_branch() -> None:
    """The issue's exact repro: rollback (not commit) runs when cleanup gets the exception."""
    conn = FakeConnection()
    get_connection = _make_async_provider(conn)

    @injectable
    async def do_work(connection: Annotated[FakeConnection, Depends(get_connection)]) -> None:
        msg = "boom"
        raise ValueError(msg)

    caught: BaseException | None = None
    try:
        await do_work()  # type: ignore[call-arg]
    except ValueError as exc:
        caught = exc
    finally:
        await cleanup_all_exit_stacks(exc=caught, raise_exception=True)

    assert caught is not None
    assert conn.rolled_back is True
    assert conn.committed is False
    assert conn.closed is True
    assert conn.seen_exception is caught


async def test_cleanup_exit_stack_of_func_with_exc_runs_rollback_branch() -> None:
    conn = FakeConnection()
    get_connection = _make_async_provider(conn)

    @injectable
    async def do_work(connection: Annotated[FakeConnection, Depends(get_connection)]) -> None:
        msg = "boom"
        raise ValueError(msg)

    caught: BaseException | None = None
    try:
        await do_work()  # type: ignore[call-arg]
    except ValueError as exc:
        caught = exc

    await cleanup_exit_stack_of_func(do_work, exc=caught, raise_exception=True)

    assert conn.rolled_back is True
    assert conn.committed is False
    assert conn.closed is True
    assert conn.seen_exception is caught


async def test_cleanup_without_exc_runs_commit_branch() -> None:
    """Existing behavior is unchanged: no exception -> commit branch runs on cleanup."""
    conn = FakeConnection()
    get_connection = _make_async_provider(conn)

    @injectable
    async def do_work(connection: Annotated[FakeConnection, Depends(get_connection)]) -> FakeConnection:
        return connection

    result = await do_work()  # type: ignore[call-arg]
    await cleanup_all_exit_stacks(raise_exception=True)

    assert result is conn
    assert conn.committed is True
    assert conn.rolled_back is False
    assert conn.closed is True
    assert conn.seen_exception is None


async def test_injectable_scope_forwards_exception_to_generator_dependencies() -> None:
    """Inside ``injectable_scope`` the exception reaches dependencies with no extra plumbing."""
    conn = FakeConnection()
    get_connection = _make_async_provider(conn)

    @injectable
    async def do_work(connection: Annotated[FakeConnection, Depends(get_connection)]) -> None:
        msg = "boom"
        raise ValueError(msg)

    with pytest.raises(ValueError, match="boom") as exc_info:
        async with injectable_scope():
            await do_work()  # type: ignore[call-arg]

    assert conn.rolled_back is True
    assert conn.committed is False
    assert conn.closed is True
    assert conn.seen_exception is exc_info.value


async def test_injectable_scope_without_exception_runs_commit_branch() -> None:
    conn = FakeConnection()
    get_connection = _make_async_provider(conn)

    @injectable
    async def do_work(connection: Annotated[FakeConnection, Depends(get_connection)]) -> FakeConnection:
        return connection

    async with injectable_scope():
        result = await do_work()  # type: ignore[call-arg]

    assert result is conn
    assert conn.committed is True
    assert conn.rolled_back is False
    assert conn.closed is True


async def test_exception_reaches_sync_generator_dependency() -> None:
    """Sync generator dependencies (run through FastAPI's threadpool CM) also see the exception."""
    conn = FakeConnection()
    get_connection = _make_sync_provider(conn)

    @injectable
    async def do_work(connection: Annotated[FakeConnection, Depends(get_connection)]) -> None:
        msg = "boom"
        raise ValueError(msg)

    caught: BaseException | None = None
    try:
        await do_work()  # type: ignore[call-arg]
    except ValueError as exc:
        caught = exc

    await cleanup_all_exit_stacks(exc=caught, raise_exception=True)

    assert conn.rolled_back is True
    assert conn.committed is False
    assert conn.closed is True
    assert conn.seen_exception is caught


def test_sync_entrypoint_cleanup_with_exc_runs_rollback_branch() -> None:
    """The whole flow from sync code: sync wrapper + run_coroutine_sync cleanup."""
    conn = FakeConnection()
    get_connection = _make_async_provider(conn)

    @injectable
    def do_work(connection: Annotated[FakeConnection, Depends(get_connection)]) -> None:
        msg = "boom"
        raise ValueError(msg)

    caught: BaseException | None = None
    try:
        do_work()  # type: ignore[call-arg]
    except ValueError as exc:
        caught = exc

    run_coroutine_sync(cleanup_all_exit_stacks(exc=caught, raise_exception=True))

    assert conn.rolled_back is True
    assert conn.committed is False
    assert conn.closed is True
    assert conn.seen_exception is caught


async def test_dependency_that_swallows_exception_is_supported() -> None:
    """A generator that handles the exception without re-raising still cleans up fine."""
    conn = FakeConnection()

    async def get_connection() -> AsyncGenerator[FakeConnection, None]:
        try:
            yield conn
        except Exception as exc:  # noqa: BLE001
            conn.rolled_back = True
            conn.seen_exception = exc
        finally:
            conn.closed = True

    @injectable
    async def do_work(connection: Annotated[FakeConnection, Depends(get_connection)]) -> None:
        msg = "boom"
        raise ValueError(msg)

    caught: BaseException | None = None
    try:
        await do_work()  # type: ignore[call-arg]
    except ValueError as exc:
        caught = exc

    await cleanup_all_exit_stacks(exc=caught, raise_exception=True)

    assert conn.rolled_back is True
    assert conn.closed is True
    assert conn.seen_exception is caught
