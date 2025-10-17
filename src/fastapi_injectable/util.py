import atexit
import inspect
import signal
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Sequence
from functools import partial
from typing import Annotated, Any, ParamSpec, TypeVar, cast, get_type_hints, overload

from fastapi import Depends

from .async_exit_stack import async_exit_stack_manager
from .cache import dependency_cache
from .concurrency import run_coroutine_sync
from .decorator import injectable

T = TypeVar("T")
T2 = TypeVar("T2")
P = ParamSpec("P")

PROVIDER_TO_WRAPPER_FUNC_MAP: dict[Callable[..., Any], list[Callable[[Any], Any]]] = {}


def _create_depends_function(
    provider: Callable[..., Any],
) -> Callable[..., Any]:
    """Build a pass-through dependency for FastAPI.

    Related issue: https://github.com/JasperSui/fastapi-injectable/issues/153

    The returned callable has a single parameter whose annotation is:
        Annotated[<provider_return_type>, Depends(provider)]
    and it simply returns that parameter.

    Type checkers see this as Callable[[T], T] with T inferred from the provider's return type.

    Raises:
        TypeError if the provider's return type cannot be determined.
    """
    # Runtime: resolve the *concrete* return type for FastAPI's inspection
    try:
        hints = get_type_hints(provider, include_extras=True)
    except Exception:  # pragma: no cover # noqa: BLE001
        hints = {}

    rt = hints.get("return", inspect.Signature.empty)

    def inner(dep: T2) -> T2:
        return dep

    # Nice signature for docs/inspection
    inner.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        parameters=[
            inspect.Parameter(
                "dep",
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=Annotated[
                    rt, Depends(provider)
                ],  # this is the key part for FastAPI to resolve the dependency
            )
        ],
        return_annotation=rt,
    )

    inner.__name__ = f"{getattr(provider, '__name__', 'provider')}_extractor"

    # Store the wrapper function for cleanup the provider later
    if provider not in PROVIDER_TO_WRAPPER_FUNC_MAP:
        PROVIDER_TO_WRAPPER_FUNC_MAP[provider] = []
    PROVIDER_TO_WRAPPER_FUNC_MAP[provider].append(inner)

    return inner


def _create_async_depends_function(
    provider: Callable[..., Any],
) -> Callable[..., Any]:
    """Build an async pass-through dependency for FastAPI.

    Similar to _create_depends_function but returns an async function,
    which causes the injectable decorator to use the async_wrapper path
    that directly awaits resolve_dependencies() without using run_coroutine_sync().

    This is specifically for async_get_injected_obj() to work in running event loops.
    """
    # Runtime: resolve the *concrete* return type for FastAPI's inspection
    try:
        hints = get_type_hints(provider, include_extras=True)
    except Exception:  # pragma: no cover # noqa: BLE001
        hints = {}

    rt = hints.get("return", inspect.Signature.empty)

    async def inner(dep: T2) -> T2:
        # If the dependency is awaitable (async function), await it
        if inspect.isawaitable(dep):
            return await dep  # type: ignore[no-any-return]
        # If it's an async generator, get the first value
        if inspect.isasyncgen(dep):
            async for value in dep:  # pragma: no cover
                return value  # type: ignore[no-any-return]
        return dep

    # Nice signature for docs/inspection
    inner.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        parameters=[
            inspect.Parameter(
                "dep",
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=Annotated[
                    rt, Depends(provider)
                ],  # this is the key part for FastAPI to resolve the dependency
            )
        ],
        return_annotation=rt,
    )

    inner.__name__ = f"{getattr(provider, '__name__', 'provider')}_async_extractor"

    # Store the wrapper function for cleanup the provider later
    if provider not in PROVIDER_TO_WRAPPER_FUNC_MAP:
        PROVIDER_TO_WRAPPER_FUNC_MAP[provider] = []
    PROVIDER_TO_WRAPPER_FUNC_MAP[provider].append(inner)

    return inner


@overload
def get_injected_obj(
    func: Callable[..., Awaitable[T]],
    args: list[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
    *,
    use_cache: bool = True,
) -> T: ...


@overload
def get_injected_obj(
    func: Callable[..., Generator[T, Any, Any]],
    args: list[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
    *,
    use_cache: bool = True,
) -> T: ...


@overload
def get_injected_obj(
    func: Callable[..., AsyncGenerator[T, Any]],
    args: list[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
    *,
    use_cache: bool = True,
) -> T: ...


@overload
def get_injected_obj(
    func: Callable[..., T],
    args: list[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
    *,
    use_cache: bool = True,
) -> T: ...


def get_injected_obj(
    func: (
        Callable[P, T]
        | Callable[P, Awaitable[T]]
        | Callable[P, Generator[T, Any, Any]]
        | Callable[P, AsyncGenerator[T, Any]]
    ),
    args: list[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
    *,
    use_cache: bool = True,
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
        args: Positional arguments to pass to the dependency function.
        kwargs: Keyword arguments to pass to the dependency function.
        use_cache: Whether to cache resolved dependencies. Defaults to True.

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
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    if args or kwargs:
        func = partial(func, *args, **kwargs)

    wrapped_func = _create_depends_function(func)
    injectable_func = injectable(wrapped_func, use_cache=use_cache)
    result = injectable_func()  # type: ignore[no-untyped-call]

    if inspect.isawaitable(result):
        return cast("T", run_coroutine_sync(result))  # type: ignore[arg-type]
    return cast("T", result)


@overload
async def async_get_injected_obj(
    func: Callable[..., Awaitable[T]],
    args: list[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
    *,
    use_cache: bool = True,
) -> T: ...


@overload
async def async_get_injected_obj(
    func: Callable[..., AsyncGenerator[T, Any]],
    args: list[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
    *,
    use_cache: bool = True,
) -> T: ...


async def async_get_injected_obj(
    func: Callable[P, Awaitable[T]] | Callable[P, AsyncGenerator[T, Any]],
    args: list[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
    *,
    use_cache: bool = True,
) -> T:
    """Async version of get_injected_obj() for use in running event loops.

    Use this function when you need to inject dependencies from within async
    contexts like Kafka consumers, async callbacks, or other scenarios where
    an event loop is already running.

    This function only accepts async functions (coroutines) and async generators.
    For sync functions, use get_injected_obj() instead.

    Args:
        func: The async dependency function to inject. Must be:
            - An async function (coroutine)
            - An async generator
        args: Positional arguments to pass to the dependency function.
        kwargs: Keyword arguments to pass to the dependency function.
        use_cache: Whether to cache resolved dependencies. Defaults to True.

    Returns:
        The first value yielded/returned by the dependency function after injection.

    Examples:
        ```python
        # In an async callback (e.g., Kafka consumer)
        async def get_service() -> Service:
            return Service()

        async def consume(message):
            service = await async_get_injected_obj(get_service)
            await service.process(message)

        # With an async generator (for cleanup)
        async def get_db() -> AsyncGenerator[Database, None]:
            db = Database()
            yield db
            await db.close()

        db = await async_get_injected_obj(get_db)
        ```

    Notes:
        - This function must be called from an async context (use await)
        - Only accepts async functions and async generators
        - For sync functions, use get_injected_obj() instead
        - For async generators, only the first yielded value is returned
        - Cleanup code in generators will be executed when calling cleanup functions
        - Uses FastAPI's dependency injection system under the hood
        - Unlike get_injected_obj(), this works in already-running event loops
    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    if args or kwargs:
        func = partial(func, *args, **kwargs)

    # Use async version to trigger async_wrapper path in injectable decorator
    wrapped_func = _create_async_depends_function(func)
    injectable_func = injectable(wrapped_func, use_cache=use_cache)
    coro = cast("Callable[..., Awaitable[T]]", injectable_func)()
    return await coro


async def cleanup_exit_stack_of_func(func: Callable[..., Any], *, raise_exception: bool = False) -> None:
    """Clean up the exit stack associated with a specific function.

    Args:
        func: The function whose exit stack should be cleaned up.
        raise_exception: Whether to raise exceptions during cleanup.
            If False, exceptions are logged as warnings. Defaults to False.

    Notes:
        - This ensures that resources such as context managers or other async cleanup routines
          are properly closed for the given function.

    Raises:
        DependencyCleanupError: When cleanup fails and raise_exception is True
    """
    for wrapper in PROVIDER_TO_WRAPPER_FUNC_MAP.get(func, [func]):
        await async_exit_stack_manager.cleanup_stack(wrapper, raise_exception=raise_exception)


async def cleanup_all_exit_stacks(*, raise_exception: bool = False) -> None:
    """Clean up all active exit stacks.

    Args:
        raise_exception: Whether to raise exceptions during cleanup.
            If False, exceptions are logged as warnings. Defaults to False.

    Notes:
        - This method iterates through all registered exit stacks and ensures they are properly closed.
        - Typically used during application shutdown to release all managed resources.

    Raises:
        DependencyCleanupError: When cleanup fails and raise_exception is True
    """
    await async_exit_stack_manager.cleanup_all_stacks(raise_exception=raise_exception)


async def clear_dependency_cache() -> None:
    """Clear the dependency resolution cache.

    Notes:
        - This is useful to free up memory or reset state in scenarios where dependencies
          might have changed dynamically.
    """
    await dependency_cache.clear()


def setup_graceful_shutdown(signals: Sequence[signal.Signals] | None = None, *, raise_exception: bool = False) -> None:
    """Register handlers to perform cleanup during application shutdown.

    Args:
        signals: A list of OS signals that should trigger the cleanup process.
                 Defaults to [SIGINT, SIGTERM].
        raise_exception: Whether to raise exceptions during cleanup.
            If False, exceptions are logged as warnings. Defaults to False.

    Notes:
        - When a registered signal is received, this function ensures that all resources
          (e.g., exit stacks) are properly released before the application exits.
        - Also registers a cleanup routine via `atexit` to handle unexpected shutdown scenarios.

    Raises:
        DependencyCleanupError: When cleanup fails and raise_exception is True
    """
    if signals is None:
        signals = [signal.SIGINT, signal.SIGTERM]

    def sync_cleanup(*_: Any) -> None:  # noqa: ANN401
        run_coroutine_sync(cleanup_all_exit_stacks(raise_exception=raise_exception))

    atexit.register(sync_cleanup)
    for sig in signals:
        signal.signal(sig, sync_cleanup)
