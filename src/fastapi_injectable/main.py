import asyncio
import importlib.metadata
import inspect
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from typing import Annotated, Any, ParamSpec, TypeVar, cast, get_args, get_origin

import fastapi.params
from fastapi import FastAPI, Request
from fastapi.dependencies.utils import get_dependant, solve_dependencies

from .async_exit_stack import async_exit_stack_manager
from .cache import dependency_cache
from .exception import FastAPICompatibilityError
from .scope import _current_scope

T = TypeVar("T")
P = ParamSpec("P")
_app: FastAPI | None = None
_app_lock = asyncio.Lock()

# Keep in sync with the ``fastapi`` pin in pyproject.toml.
_SUPPORTED_FASTAPI_RANGE = ">=0.112.4,<1.0.0"
# Keyword parameters of FastAPI's private ``solve_dependencies`` that
# ``resolve_dependencies`` passes by name. If a future FastAPI within the
# supported range drops one of these, resolution would otherwise fail at call
# time with a cryptic ``TypeError``; probing at import time turns that into an
# actionable, version-named error.
_REQUIRED_SOLVE_DEPENDENCIES_PARAMS = (
    "dependency_overrides_provider",
    "dependency_cache",
    "async_exit_stack",
    "embed_body_fields",
)


def _verify_fastapi_compatibility(solve: Callable[..., Any]) -> None:
    """Fail fast if FastAPI's private dependency-resolution API has drifted.

    Introspects ``solve_dependencies`` and raises ``FastAPICompatibilityError``
    (naming the installed FastAPI version and the supported range) when an
    expected keyword parameter is missing.

    Args:
        solve: FastAPI's ``solve_dependencies`` callable (injected for testing).
    """
    try:
        params = inspect.signature(solve).parameters
    except (ValueError, TypeError):  # pragma: no cover - builtins expose no signature
        return

    missing = [name for name in _REQUIRED_SOLVE_DEPENDENCIES_PARAMS if name not in params]
    if not missing:
        return

    try:
        installed = importlib.metadata.version("fastapi")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - fastapi is a hard dependency
        installed = "unknown"

    msg = (
        f"fastapi-injectable is incompatible with the installed FastAPI version ({installed}). "
        f"It relies on FastAPI's private dependency-resolution API "
        f"(fastapi.dependencies.utils.solve_dependencies), whose signature is missing the "
        f"expected parameter(s): {', '.join(missing)}. "
        f"Supported FastAPI range: {_SUPPORTED_FASTAPI_RANGE}. "
        f"Pin FastAPI to a supported version, or report this at "
        f"https://github.com/JasperSui/fastapi-injectable/issues."
    )
    raise FastAPICompatibilityError(msg)


_verify_fastapi_compatibility(solve_dependencies)


async def register_app(app: FastAPI) -> None:
    """Register the given FastAPI app for constructing fake request later."""
    global _app  # noqa: PLW0603
    async with _app_lock:
        _app = app


def _get_app() -> FastAPI | None:
    """Get the registered FastAPI app."""
    return _app


def _has_depends(param: inspect.Parameter) -> bool:
    """Check if a parameter has a Depends() annotation (either via Annotated metadata or as default)."""
    # Check Annotated[Type, Depends(...)] style
    if get_origin(param.annotation) is Annotated:
        for metadata in get_args(param.annotation)[1:]:
            if isinstance(metadata, fastapi.params.Depends):
                return True
    # Check default=Depends(...) style
    return isinstance(param.default, fastapi.params.Depends)


def _build_dependency_only_callable(
    func: Callable[..., Any],
) -> Callable[..., Any]:
    """Create a callable with only Depends() parameters in its signature.

    This prevents FastAPI's get_dependant from trying to parse non-dependency
    parameters (e.g. ``self``, Celery's bound ``task``) as query/body params,
    which would fail for types that are not valid Pydantic fields.
    """
    sig = inspect.signature(func)
    dep_params = [p for p in sig.parameters.values() if _has_depends(p)]

    # If all parameters are dependencies, no filtering needed
    if len(dep_params) == len(sig.parameters):
        return func

    def stub() -> None: ...  # pragma: no cover

    stub.__signature__ = sig.replace(parameters=dep_params)  # type: ignore[attr-defined]
    # get_dependant inspects annotations from __annotations__ as well in some FastAPI versions
    stub.__annotations__ = {p.name: p.annotation for p in dep_params}
    # Preserve identity for error messages and diagnostics
    stub.__name__ = getattr(func, "__name__", "stub")
    stub.__qualname__ = getattr(func, "__qualname__", "stub")
    return stub


async def resolve_dependencies(
    func: Callable[P, T] | Callable[P, Awaitable[T]],
    *,
    use_cache: bool = True,
    provided_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve dependencies for the given function using FastAPI's dependency injection system.

    This function resolves dependencies defined via FastAPI's dependency mechanism
    and returns a dictionary of resolved arguments for the given function.

    Args:
        func: The function for which dependencies need to be resolved. It can be a synchronous
            or asynchronous callable.
        use_cache: Whether to use a cache for dependency resolution. Defaults to True.
        provided_kwargs: Explicit kwargs passed by the caller (these override DI).

    Returns:
        A dictionary mapping argument names to resolved dependency values.

    Notes:
        - A fake HTTP request is created to mimic FastAPI's request-based dependency resolution.
    """
    provided_kwargs = provided_kwargs or {}
    dep_only_func = _build_dependency_only_callable(func)
    root_dep = get_dependant(path="command", call=dep_only_func)

    # Get names of actual dependency (Depends()) parameters
    dependency_names = {param.name for param in root_dep.dependencies if param.name}

    # Drop dependencies that are already satisfied by provided kwargs
    effective_dependencies = [dep for dep in root_dep.dependencies if dep.name not in provided_kwargs]
    root_dep.dependencies = effective_dependencies

    root_dep.call = cast("Callable[..., Any]", func)

    scope = _current_scope.get()
    if scope is not None:
        async_exit_stack = scope.exit_stack
    else:
        async_exit_stack = await async_exit_stack_manager.get_stack(root_dep.call)

    # Use isolated stacks for FastAPI's internal logic to prevent conflicts.
    # We must ensure they are closed when our main stack is closed.
    fastapi_inner_astack = AsyncExitStack()
    fastapi_function_astack = AsyncExitStack()

    async_exit_stack.push_async_callback(fastapi_inner_astack.aclose)
    async_exit_stack.push_async_callback(fastapi_function_astack.aclose)

    fake_request_scope: dict[str, Any] = {
        "type": "http",
        "headers": [],
        "query_string": "",
        # These two inner stacks are used to workaround the assertion in fastapi==0.121.0
        # Ref: https://github.com/fastapi/fastapi/commit/ac438b99342c859ae0e10f7064021125bd247bf5#diff-aef3dac481b68359f4edd6974fa3a047cfde595254a4567a560cebc9ccb0673fR575-R582 # noqa: E501
        "fastapi_inner_astack": fastapi_inner_astack,
        "fastapi_function_astack": fastapi_function_astack,
    }
    app = _get_app()
    if app is not None:
        fake_request_scope["app"] = app
    fake_request = Request(fake_request_scope)
    if scope is not None:
        cache = scope.get_cache() if use_cache else None
    else:
        cache = dependency_cache.get() if use_cache else None
    solved_dependency = await solve_dependencies(
        request=fake_request,
        dependant=root_dep,
        async_exit_stack=async_exit_stack,
        embed_body_fields=False,
        dependency_cache=cache,
        dependency_overrides_provider=app,
    )
    if cache is not None:
        cache.update(solved_dependency.dependency_cache)

    resolved = {
        param_name: value for param_name, value in solved_dependency.values.items() if param_name in dependency_names
    }

    return {**resolved, **provided_kwargs}
