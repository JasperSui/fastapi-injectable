import inspect
from collections.abc import Awaitable, Callable, Coroutine, Generator
from functools import wraps
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, cast, overload

from .concurrency import run_coroutine_sync
from .main import resolve_dependencies

T = TypeVar("T")
P = ParamSpec("P")

if TYPE_CHECKING:

    def set_original_func(wrapper: Any, target: Any) -> None:  # noqa: ANN401
        pass
else:

    def set_original_func(wrapper: Any, target: Any) -> None:  # noqa: ANN401
        wrapper.__original_func__ = target


@overload
def injectable(
    func: Callable[P, T],
    *,
    use_cache: bool = True,
) -> Callable[P, T]: ...


@overload
def injectable(
    func: Callable[P, Generator[T, Any, Any]],
    *,
    use_cache: bool = True,
) -> Callable[P, T]: ...


@overload
def injectable(
    *,
    use_cache: bool = True,
) -> Callable[[Callable[P, T]], Callable[P, T]]: ...


def injectable(
    func: Callable[P, T] | Callable[P, Awaitable[T]] | None = None,
    *,
    use_cache: bool = True,
) -> (
    Callable[P, T]
    | Callable[P, Awaitable[T]]
    | Callable[[Callable[P, T] | Callable[P, Awaitable[T]]], Callable[P, T] | Callable[P, Awaitable[T]]]
):
    """Decorator to inject dependencies into any callable, sync or async."""

    def decorator(
        target: Callable[P, T] | Callable[P, Awaitable[T]],
    ) -> Callable[P, T] | Callable[P, Awaitable[T]]:
        is_async = inspect.iscoroutinefunction(target)

        @wraps(target)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            dependencies = await resolve_dependencies(func=target, use_cache=use_cache)
            return await cast(Callable[..., Coroutine[Any, Any, T]], target)(*args, **{**dependencies, **kwargs})

        @wraps(target)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            dependencies = run_coroutine_sync(resolve_dependencies(func=target, use_cache=use_cache))
            return cast(Callable[..., T], target)(*args, **{**dependencies, **kwargs})

        if is_async:
            set_original_func(async_wrapper, target)
            return async_wrapper

        set_original_func(sync_wrapper, target)
        return sync_wrapper

    if func is None:
        return decorator

    decorated_func = decorator(func)
    set_original_func(decorated_func, func)
    return decorated_func
