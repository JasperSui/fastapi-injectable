import atexit
import inspect
import signal
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from typing import Any, ParamSpec, TypeVar, cast

from .cache import dependency_cache
from .concurrency import run_coroutine_sync
from .decorator import injectable
from .manager import async_exit_stack_manager

T = TypeVar("T")
P = ParamSpec("P")


def get_injected_obj(
    func: (
        Callable[P, T]
        | Callable[P, Awaitable[T]]
        | Callable[P, Generator[T, Any, Any]]
        | Callable[P, AsyncGenerator[T, Any]]
    ),
    *,
    use_cache: bool = True,
    raise_exception: bool = False,
) -> T:
    """Get an injected object from a dependency function with FastAPI's dependency injection.

    This function handles different types of callables (sync/async functions and generators) and
    returns the first yielded/returned value after resolving dependencies.

    Args:
        func: The dependency function to inject. Can be:
            - A regular synchronous function
            - An async function (coroutine)
            - A synchronous generator
            - An async generator
        use_cache: Whether to cache resolved dependencies. Defaults to True.
        raise_exception: Whether to raise exceptions during dependency resolution.
            If False, exceptions are logged as warnings. Defaults to False.

    Returns:
        The first value yielded/returned by the dependency function after injection.

    Examples:
        ```python
        # With a regular function
        def get_service() -> Service:
            return Service()

        service = get_injected_obj(get_service)

        # With an async function
        async def get_async_service() -> Service:
            return await create_service()

        service = get_injected_obj(get_async_service)

        # With a generator (for cleanup)
        def get_db() -> Generator[Database, None, None]:
            db = Database()
            yield db
            db.cleanup()

        db = get_injected_obj(get_db)
        ```

    Notes:
        - For generator functions, only the first yielded value is returned
        - Cleanup code in generators will be executed when calling cleanup functions
        - Uses FastAPI's dependency injection system under the hood
    """
    # TODO(Jasper Sui): Correct the type hints down below.  # noqa: TD003
    injectable_func = injectable(func, use_cache=use_cache, raise_exception=raise_exception)

    if inspect.isasyncgenfunction(func):
        # Handle async generator
        async_gen = injectable_func()  # type: ignore[call-arg]
        return cast(T, run_coroutine_sync(anext(async_gen)))  # type: ignore[call-overload]

    if inspect.isgeneratorfunction(func):
        # Handle sync generator
        gen = injectable_func()  # type: ignore[call-arg]
        return cast(T, next(gen))  # type: ignore[call-overload]

    if inspect.iscoroutinefunction(func):
        # Handle coroutine
        return run_coroutine_sync(injectable_func())  # type: ignore[call-arg, arg-type]

    # Handle regular function
    return cast(T, injectable_func())  # type: ignore[call-arg]


async def cleanup_exit_stack_of_func(func: Callable[..., Any]) -> None:
    """Clean up the exit stack associated with a specific function.

    Args:
        func: The function whose exit stack should be cleaned up.

    Notes:
        - This ensures that resources such as context managers or other async cleanup routines
          are properly closed for the given function.
    """
    await async_exit_stack_manager.cleanup_stack(func)


async def cleanup_all_exit_stacks() -> None:
    """Clean up all active exit stacks.

    Notes:
        - This method iterates through all registered exit stacks and ensures they are properly closed.
        - Typically used during application shutdown to release all managed resources.
    """
    await async_exit_stack_manager.cleanup_all_stacks()


async def clear_dependency_cache() -> None:
    """Clear the dependency resolution cache.

    Notes:
        - This is useful to free up memory or reset state in scenarios where dependencies
          might have changed dynamically.
    """
    await dependency_cache.clear()


def setup_graceful_shutdown(signals: list[signal.Signals] | None = None) -> None:
    """Register handlers to perform cleanup during application shutdown.

    Args:
        signals: A list of OS signals that should trigger the cleanup process.
                 Defaults to [SIGINT, SIGTERM].

    Notes:
        - When a registered signal is received, this function ensures that all resources
          (e.g., exit stacks) are properly released before the application exits.
        - Also registers a cleanup routine via `atexit` to handle unexpected shutdown scenarios.
    """
    if signals is None:
        signals = [signal.SIGINT, signal.SIGTERM]

    def sync_cleanup(*_: Any) -> None:  # noqa: ANN401
        run_coroutine_sync(cleanup_all_exit_stacks())

    atexit.register(sync_cleanup)
    for sig in signals:
        signal.signal(sig, sync_cleanup)
