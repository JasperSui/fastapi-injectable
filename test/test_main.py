from collections.abc import Generator
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.fastapi_injectable.main import DependencyResolveError, resolve_dependencies


class DummyDependency:
    pass


@pytest.fixture
def mock_solve_dependencies() -> Generator[Mock, None, None]:
    with patch("src.fastapi_injectable.main.solve_dependencies", new_callable=AsyncMock) as mock:
        yield mock


@pytest.fixture
def mock_get_dependant() -> Generator[Mock, None, None]:
    with patch("src.fastapi_injectable.main.get_dependant") as mock:
        yield mock


@pytest.fixture
def mock_dependency_cache() -> Generator[Mock, None, None]:
    with patch("src.fastapi_injectable.main.dependency_cache") as mock:
        mock.get.return_value = {}
        yield mock


@pytest.fixture
def mock_async_exit_stack_manager() -> Generator[Mock, None, None]:
    with patch("src.fastapi_injectable.main.async_exit_stack_manager") as mock:
        mock.get_stack = AsyncMock()
        yield mock


async def test_resolve_dependencies_no_dependencies(
    mock_solve_dependencies: AsyncMock,
    mock_get_dependant: Mock,
    mock_dependency_cache: Mock,
    mock_async_exit_stack_manager: Mock,
) -> None:
    mock_solve_dependencies.return_value = AsyncMock(values={}, dependency_cache={})

    def func() -> None:
        return None

    dependencies = await resolve_dependencies(func)

    mock_get_dependant.assert_called_once_with(path="command", call=func)
    mock_solve_dependencies.assert_awaited_once()
    assert dependencies == {}


async def test_resolve_dependencies_with_dependencies(
    mock_solve_dependencies: AsyncMock,
    mock_get_dependant: Mock,
    mock_dependency_cache: Mock,
    mock_async_exit_stack_manager: Mock,
) -> None:
    mock_solve_dependencies.return_value = AsyncMock(values={"dep": DummyDependency()}, dependency_cache={})

    def func(dep: DummyDependency) -> None:
        return None

    dependencies = await resolve_dependencies(func)

    assert dependencies == {"dep": mock_solve_dependencies.return_value.values["dep"]}


async def test_resolve_dependencies_with_cache(
    mock_solve_dependencies: AsyncMock,
    mock_get_dependant: Mock,
    mock_dependency_cache: Mock,
    mock_async_exit_stack_manager: Mock,
) -> None:
    dep_obj = DummyDependency()
    mock_solve_dependencies.return_value = AsyncMock(values={"dep": dep_obj}, dependency_cache={"dep": dep_obj})

    def func(dep: DummyDependency) -> None:
        return None

    mock_dependency_cache.get.return_value = Mock()
    dependencies = await resolve_dependencies(func, use_cache=True)

    mock_dependency_cache.get.assert_called_once()
    mock_solve_dependencies.assert_awaited_once()
    assert dependencies == {"dep": mock_solve_dependencies.return_value.values["dep"]}
    mock_dependency_cache.get.return_value.update.assert_called_once_with({"dep": dep_obj})


async def test_resolve_dependencies_without_cache(
    mock_solve_dependencies: AsyncMock,
    mock_get_dependant: Mock,
    mock_dependency_cache: Mock,
    mock_async_exit_stack_manager: Mock,
) -> None:
    mock_solve_dependencies.return_value = AsyncMock(
        values={"dep": DummyDependency()}, dependency_cache={"dep": DummyDependency()}
    )

    def func(dep: DummyDependency) -> None:
        return None

    dependencies = await resolve_dependencies(func, use_cache=False)

    mock_dependency_cache.get.assert_not_called()
    mock_solve_dependencies.assert_awaited_once()
    assert dependencies == {"dep": mock_solve_dependencies.return_value.values["dep"]}


async def test_resolve_dependencies_raise_exception_on_error(
    mock_solve_dependencies: AsyncMock,
    mock_get_dependant: Mock,
    mock_dependency_cache: Mock,
    mock_async_exit_stack_manager: Mock,
) -> None:
    mock_solve_dependencies.return_value = AsyncMock(errors=["Error"], values={}, dependency_cache={})

    def func() -> None:
        return None

    with pytest.raises(DependencyResolveError):
        await resolve_dependencies(func, raise_exception=True)


async def test_resolve_dependencies_log_warning_on_error(
    mock_solve_dependencies: AsyncMock,
    mock_get_dependant: Mock,
    mock_dependency_cache: Mock,
    mock_async_exit_stack_manager: Mock,
) -> None:
    mock_solve_dependencies.return_value = AsyncMock(errors=["Error"], values={}, dependency_cache={})

    def func() -> None:
        return None

    with patch("src.fastapi_injectable.main.logger") as mock_logger:
        dependencies = await resolve_dependencies(func, raise_exception=False)
        mock_logger.warning.assert_called_once_with(
            f"Something wrong when resolving dependencies of {func}, errors: {mock_solve_dependencies.return_value.errors}"  # noqa: E501
        )
        assert dependencies == {}
