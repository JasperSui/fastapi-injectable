from collections.abc import AsyncGenerator, AsyncIterator, Generator
from contextlib import asynccontextmanager
from functools import partial
from typing import Annotated, Any
from unittest.mock import Mock

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from src.fastapi_injectable import register_app
from src.fastapi_injectable.concurrency import loop_manager, run_coroutine_sync
from src.fastapi_injectable.decorator import injectable
from src.fastapi_injectable.util import (
    async_get_injected_obj,
    cleanup_all_exit_stacks,
    cleanup_exit_stack_of_func,
    get_injected_obj,
)


@pytest.fixture
async def clean_exit_stack_manager() -> AsyncGenerator[None, None]:
    # Clean up any existing stacks before test
    await cleanup_all_exit_stacks()
    yield
    # Clean up after test
    await cleanup_all_exit_stacks()


class Mayor:
    def __init__(self) -> None:
        self._is_cleaned_up = False

    def cleanup(self) -> None:
        self._is_cleaned_up = True


class Capital:
    def __init__(self, mayor: Mayor) -> None:
        self.mayor = mayor
        self._is_cleaned_up = False

    def cleanup(self) -> None:
        self._is_cleaned_up = True


class Country:
    def __init__(self, capital: Capital) -> None:
        self.capital = capital


def test_sync_generators_with_injectable_be_correctly_cleaned_up_by_cleanup_all_exit_stacks(
    clean_exit_stack_manager: None,
) -> None:
    def get_mayor() -> Generator[Mayor, None, None]:
        mayor = Mayor()
        yield mayor
        mayor.cleanup()

    def get_capital(
        mayor: Annotated[Mayor, Depends(get_mayor)],
    ) -> Generator[Capital, None, None]:
        capital = Capital(mayor)
        yield capital
        capital.cleanup()

    @injectable(use_cache=False)
    def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    country_1: Country = get_country()  # type: ignore  # noqa: PGH003
    assert country_1.capital._is_cleaned_up is False
    assert country_1.capital.mayor._is_cleaned_up is False

    country_2: Country = get_country()  # type: ignore  # noqa: PGH003
    assert country_2.capital._is_cleaned_up is False
    assert country_2.capital.mayor._is_cleaned_up is False

    assert country_1 is not country_2
    assert country_1.capital is not country_2.capital
    assert country_1.capital.mayor is not country_2.capital.mayor

    run_coroutine_sync(cleanup_all_exit_stacks(raise_exception=True))

    assert country_1.capital._is_cleaned_up is True
    assert country_1.capital.mayor._is_cleaned_up is True  # type: ignore[unreachable]
    assert country_2.capital._is_cleaned_up is True
    assert country_2.capital.mayor._is_cleaned_up is True


def test_async_generators_with_injectable_be_correctly_cleaned_up_by_cleanup_all_exit_stacks(
    clean_exit_stack_manager: None,
) -> None:
    async def get_mayor() -> AsyncGenerator[Mayor, None]:
        mayor = Mayor()
        yield mayor
        mayor.cleanup()

    async def get_capital(
        mayor: Annotated[Mayor, Depends(get_mayor)],
    ) -> AsyncGenerator[Capital, None]:
        capital = Capital(mayor)
        yield capital
        capital.cleanup()

    @injectable(use_cache=False)
    async def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    country_1: Country = run_coroutine_sync(get_country())  # type: ignore  # noqa: PGH003
    assert country_1.capital._is_cleaned_up is False
    assert country_1.capital.mayor._is_cleaned_up is False

    country_2: Country = run_coroutine_sync(get_country())  # type: ignore  # noqa: PGH003
    assert country_2.capital._is_cleaned_up is False
    assert country_2.capital.mayor._is_cleaned_up is False

    assert country_1 is not country_2
    assert country_1.capital is not country_2.capital
    assert country_1.capital.mayor is not country_2.capital.mayor

    run_coroutine_sync(cleanup_all_exit_stacks())

    assert country_1.capital._is_cleaned_up is True
    assert country_1.capital.mayor._is_cleaned_up is True  # type: ignore[unreachable]
    assert country_2.capital._is_cleaned_up is True
    assert country_2.capital.mayor._is_cleaned_up is True


def test_sync_and_async_generators_with_injectable_be_correctly_cleaned_up_by_cleanup_all_exit_stacks(
    clean_exit_stack_manager: None,
) -> None:
    def get_mayor() -> Generator[Mayor, None, None]:
        mayor = Mayor()
        yield mayor
        mayor.cleanup()

    async def get_capital(
        mayor: Annotated[Mayor, Depends(get_mayor)],
    ) -> AsyncGenerator[Capital, None]:
        capital = Capital(mayor)
        yield capital
        capital.cleanup()

    @injectable(use_cache=False)
    async def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    country_1: Country = run_coroutine_sync(get_country())  # type: ignore  # noqa: PGH003
    assert country_1.capital._is_cleaned_up is False
    assert country_1.capital.mayor._is_cleaned_up is False

    country_2: Country = run_coroutine_sync(get_country())  # type: ignore  # noqa: PGH003
    assert country_2.capital._is_cleaned_up is False
    assert country_2.capital.mayor._is_cleaned_up is False

    assert country_1 is not country_2
    assert country_1.capital is not country_2.capital
    assert country_1.capital.mayor is not country_2.capital.mayor

    run_coroutine_sync(cleanup_all_exit_stacks())

    assert country_1.capital._is_cleaned_up is True
    assert country_1.capital.mayor._is_cleaned_up is True  # type: ignore[unreachable]
    assert country_2.capital._is_cleaned_up is True
    assert country_2.capital.mayor._is_cleaned_up is True


def test_sync_generators_with_get_injected_obj_be_correctly_cleaned_up_by_cleanup_all_exit_stacks(
    clean_exit_stack_manager: None,
) -> None:
    def get_mayor() -> Generator[Mayor, None, None]:
        mayor = Mayor()
        yield mayor
        mayor.cleanup()

    def get_capital(
        mayor: Annotated[Mayor, Depends(get_mayor)],
    ) -> Generator[Capital, None, None]:
        capital = Capital(mayor)
        yield capital
        capital.cleanup()

    def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    country_1: Country = get_injected_obj(get_country, use_cache=False)
    assert country_1.capital._is_cleaned_up is False
    assert country_1.capital.mayor._is_cleaned_up is False

    country_2: Country = get_injected_obj(get_country, use_cache=False)
    assert country_2.capital._is_cleaned_up is False
    assert country_2.capital.mayor._is_cleaned_up is False

    assert country_1 is not country_2
    assert country_1.capital is not country_2.capital
    assert country_1.capital.mayor is not country_2.capital.mayor

    run_coroutine_sync(cleanup_all_exit_stacks())

    assert country_1.capital._is_cleaned_up is True
    assert country_1.capital.mayor._is_cleaned_up is True  # type: ignore[unreachable]
    assert country_2.capital._is_cleaned_up is True
    assert country_2.capital.mayor._is_cleaned_up is True


def test_async_generators_with_get_injected_obj_be_correctly_cleaned_up_by_cleanup_all_exit_stacks(
    clean_exit_stack_manager: None,
) -> None:
    async def get_mayor() -> AsyncGenerator[Mayor, None]:
        mayor = Mayor()
        yield mayor
        mayor.cleanup()

    async def get_capital(
        mayor: Annotated[Mayor, Depends(get_mayor)],
    ) -> AsyncGenerator[Capital, None]:
        capital = Capital(mayor)
        yield capital
        capital.cleanup()

    async def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    country_1: Country = get_injected_obj(get_country, use_cache=False)
    assert country_1.capital._is_cleaned_up is False
    assert country_1.capital.mayor._is_cleaned_up is False

    country_2: Country = get_injected_obj(get_country, use_cache=False)
    assert country_2.capital._is_cleaned_up is False
    assert country_2.capital.mayor._is_cleaned_up is False

    assert country_1 is not country_2
    assert country_1.capital is not country_2.capital
    assert country_1.capital.mayor is not country_2.capital.mayor

    run_coroutine_sync(cleanup_all_exit_stacks())

    assert country_1.capital._is_cleaned_up is True
    assert country_1.capital.mayor._is_cleaned_up is True  # type: ignore[unreachable]
    assert country_2.capital._is_cleaned_up is True
    assert country_2.capital.mayor._is_cleaned_up is True


def test_sync_and_async_generators_with_get_injected_obj_be_correctly_cleaned_up_by_cleanup_all_exit_stacks(
    clean_exit_stack_manager: None,
) -> None:
    def get_mayor() -> Generator[Mayor, None, None]:
        mayor = Mayor()
        yield mayor
        mayor.cleanup()

    async def get_capital(
        mayor: Annotated[Mayor, Depends(get_mayor)],
    ) -> AsyncGenerator[Capital, None]:
        capital = Capital(mayor)
        yield capital
        capital.cleanup()

    async def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    country_1: Country = get_injected_obj(get_country, use_cache=False)
    assert country_1.capital._is_cleaned_up is False
    assert country_1.capital.mayor._is_cleaned_up is False

    country_2: Country = get_injected_obj(get_country, use_cache=False)
    assert country_2.capital._is_cleaned_up is False
    assert country_2.capital.mayor._is_cleaned_up is False

    assert country_1 is not country_2
    assert country_1.capital is not country_2.capital
    assert country_1.capital.mayor is not country_2.capital.mayor

    run_coroutine_sync(cleanup_all_exit_stacks())

    assert country_1.capital._is_cleaned_up is True
    assert country_1.capital.mayor._is_cleaned_up is True  # type: ignore[unreachable]
    assert country_2.capital._is_cleaned_up is True
    assert country_2.capital.mayor._is_cleaned_up is True


def test_sync_generators_with_injectable_be_correctly_cleaned_up_by_cleanup_exit_stack_of_func(
    clean_exit_stack_manager: None,
) -> None:
    def get_mayor() -> Generator[Mayor, None, None]:
        mayor = Mayor()
        yield mayor
        mayor.cleanup()

    def get_capital(
        mayor: Annotated[Mayor, Depends(get_mayor)],
    ) -> Generator[Capital, None, None]:
        capital = Capital(mayor)
        yield capital
        capital.cleanup()

    @injectable(use_cache=False)
    def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    @injectable(use_cache=False)
    def another_get_country(
        capital: Annotated[Capital, Depends(get_capital)],
    ) -> Country:
        return Country(capital)

    country_1: Country = get_country()  # type: ignore  # noqa: PGH003
    assert country_1.capital._is_cleaned_up is False
    assert country_1.capital.mayor._is_cleaned_up is False

    country_2: Country = get_country()  # type: ignore  # noqa: PGH003
    assert country_2.capital._is_cleaned_up is False
    assert country_2.capital.mayor._is_cleaned_up is False

    another_country_1: Country = another_get_country()  # type: ignore  # noqa: PGH003
    assert another_country_1.capital._is_cleaned_up is False
    assert another_country_1.capital.mayor._is_cleaned_up is False

    assert country_1 is not country_2 is not another_country_1
    assert country_1.capital is not country_2.capital is not another_country_1.capital
    assert country_1.capital.mayor is not country_2.capital.mayor is not another_country_1.capital.mayor

    run_coroutine_sync(cleanup_exit_stack_of_func(get_country))

    assert country_1.capital._is_cleaned_up is True
    assert country_1.capital.mayor._is_cleaned_up is True  # type: ignore[unreachable]
    assert country_2.capital._is_cleaned_up is True
    assert country_2.capital.mayor._is_cleaned_up is True

    # Another country should not be cleaned up now
    assert another_country_1.capital._is_cleaned_up is False
    assert another_country_1.capital.mayor._is_cleaned_up is False

    run_coroutine_sync(cleanup_exit_stack_of_func(another_get_country))

    assert another_country_1.capital._is_cleaned_up is True
    assert another_country_1.capital.mayor._is_cleaned_up is True


def test_async_generators_with_injectable_be_correctly_cleaned_up_by_cleanup_exit_stack_of_func(
    clean_exit_stack_manager: None,
) -> None:
    async def get_mayor() -> AsyncGenerator[Mayor, None]:
        mayor = Mayor()
        yield mayor
        mayor.cleanup()

    async def get_capital(
        mayor: Annotated[Mayor, Depends(get_mayor)],
    ) -> AsyncGenerator[Capital, None]:
        capital = Capital(mayor)
        yield capital
        capital.cleanup()

    @injectable(use_cache=False)
    async def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    @injectable(use_cache=False)
    def another_get_country(
        capital: Annotated[Capital, Depends(get_capital)],
    ) -> Country:
        return Country(capital)

    country_1: Country = run_coroutine_sync(get_country())  # type: ignore  # noqa: PGH003
    assert country_1.capital._is_cleaned_up is False
    assert country_1.capital.mayor._is_cleaned_up is False

    country_2: Country = run_coroutine_sync(get_country())  # type: ignore  # noqa: PGH003
    assert country_2.capital._is_cleaned_up is False
    assert country_2.capital.mayor._is_cleaned_up is False

    another_country_1: Country = another_get_country()  # type: ignore  # noqa: PGH003
    assert another_country_1.capital._is_cleaned_up is False
    assert another_country_1.capital.mayor._is_cleaned_up is False

    assert country_1 is not country_2 is not another_country_1
    assert country_1.capital is not country_2.capital is not another_country_1.capital
    assert country_1.capital.mayor is not country_2.capital.mayor is not another_country_1.capital.mayor

    run_coroutine_sync(cleanup_exit_stack_of_func(get_country))

    assert country_1.capital._is_cleaned_up is True
    assert country_1.capital.mayor._is_cleaned_up is True  # type: ignore[unreachable]
    assert country_2.capital._is_cleaned_up is True
    assert country_2.capital.mayor._is_cleaned_up is True

    # Another country should not be cleaned up now
    assert another_country_1.capital._is_cleaned_up is False
    assert another_country_1.capital.mayor._is_cleaned_up is False

    run_coroutine_sync(cleanup_exit_stack_of_func(another_get_country))

    assert another_country_1.capital._is_cleaned_up is True
    assert another_country_1.capital.mayor._is_cleaned_up is True


def test_sync_and_async_generators_with_injectable_be_correctly_cleaned_up_by_cleanup_exit_stack_of_func(
    clean_exit_stack_manager: None,
) -> None:
    def get_mayor() -> Generator[Mayor, None, None]:
        mayor = Mayor()
        yield mayor
        mayor.cleanup()

    async def get_capital(
        mayor: Annotated[Mayor, Depends(get_mayor)],
    ) -> AsyncGenerator[Capital, None]:
        capital = Capital(mayor)
        yield capital
        capital.cleanup()

    @injectable(use_cache=False)
    async def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    @injectable(use_cache=False)
    def another_get_country(
        capital: Annotated[Capital, Depends(get_capital)],
    ) -> Country:
        return Country(capital)

    country_1: Country = run_coroutine_sync(get_country())  # type: ignore  # noqa: PGH003
    assert country_1.capital._is_cleaned_up is False
    assert country_1.capital.mayor._is_cleaned_up is False

    country_2: Country = run_coroutine_sync(get_country())  # type: ignore  # noqa: PGH003
    assert country_2.capital._is_cleaned_up is False
    assert country_2.capital.mayor._is_cleaned_up is False

    another_country_1: Country = another_get_country()  # type: ignore  # noqa: PGH003
    assert another_country_1.capital._is_cleaned_up is False
    assert another_country_1.capital.mayor._is_cleaned_up is False

    assert country_1 is not country_2
    assert country_1.capital is not country_2.capital
    assert country_1.capital.mayor is not country_2.capital.mayor

    run_coroutine_sync(cleanup_exit_stack_of_func(get_country))

    assert country_1.capital._is_cleaned_up is True
    assert country_1.capital.mayor._is_cleaned_up is True  # type: ignore[unreachable]
    assert country_2.capital._is_cleaned_up is True
    assert country_2.capital.mayor._is_cleaned_up is True

    # Another country should not be cleaned up now
    assert another_country_1.capital._is_cleaned_up is False
    assert another_country_1.capital.mayor._is_cleaned_up is False

    run_coroutine_sync(cleanup_exit_stack_of_func(another_get_country))

    assert another_country_1.capital._is_cleaned_up is True
    assert another_country_1.capital.mayor._is_cleaned_up is True


def test_sync_generators_with_get_injected_obj_be_correctly_cleaned_up_by_cleanup_exit_stack_of_func(
    clean_exit_stack_manager: None,
) -> None:
    def get_mayor() -> Generator[Mayor, None, None]:
        mayor = Mayor()
        yield mayor
        mayor.cleanup()

    def get_capital(
        mayor: Annotated[Mayor, Depends(get_mayor)],
    ) -> Generator[Capital, None, None]:
        capital = Capital(mayor)
        yield capital
        capital.cleanup()

    def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    def another_get_country(
        capital: Annotated[Capital, Depends(get_capital)],
    ) -> Country:
        return Country(capital)

    country_1: Country = get_injected_obj(get_country, use_cache=False)
    assert country_1.capital._is_cleaned_up is False
    assert country_1.capital.mayor._is_cleaned_up is False

    country_2: Country = get_injected_obj(get_country, use_cache=False)
    assert country_2.capital._is_cleaned_up is False
    assert country_2.capital.mayor._is_cleaned_up is False

    another_country_1: Country = get_injected_obj(another_get_country, use_cache=False)
    assert another_country_1.capital._is_cleaned_up is False
    assert another_country_1.capital.mayor._is_cleaned_up is False

    assert country_1 is not country_2 is not another_country_1
    assert country_1.capital is not country_2.capital is not another_country_1.capital
    assert country_1.capital.mayor is not country_2.capital.mayor is not another_country_1.capital.mayor

    run_coroutine_sync(cleanup_exit_stack_of_func(get_country))

    assert country_1.capital._is_cleaned_up is True
    assert country_1.capital.mayor._is_cleaned_up is True  # type: ignore[unreachable]
    assert country_2.capital._is_cleaned_up is True
    assert country_2.capital.mayor._is_cleaned_up is True

    # Another country should not be cleaned up now
    assert another_country_1.capital._is_cleaned_up is False
    assert another_country_1.capital.mayor._is_cleaned_up is False

    run_coroutine_sync(cleanup_exit_stack_of_func(another_get_country))

    assert another_country_1.capital._is_cleaned_up is True
    assert another_country_1.capital.mayor._is_cleaned_up is True


def test_async_generators_with_get_injected_obj_be_correctly_cleaned_up_by_cleanup_exit_stack_of_func(
    clean_exit_stack_manager: None,
) -> None:
    async def get_mayor() -> AsyncGenerator[Mayor, None]:
        mayor = Mayor()
        yield mayor
        mayor.cleanup()

    async def get_capital(
        mayor: Annotated[Mayor, Depends(get_mayor)],
    ) -> AsyncGenerator[Capital, None]:
        capital = Capital(mayor)
        yield capital
        capital.cleanup()

    async def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    def another_get_country(
        capital: Annotated[Capital, Depends(get_capital)],
    ) -> Country:
        return Country(capital)

    country_1: Country = get_injected_obj(get_country, use_cache=False)
    assert country_1.capital._is_cleaned_up is False
    assert country_1.capital.mayor._is_cleaned_up is False

    country_2: Country = get_injected_obj(get_country, use_cache=False)
    assert country_2.capital._is_cleaned_up is False
    assert country_2.capital.mayor._is_cleaned_up is False

    another_country_1: Country = get_injected_obj(another_get_country, use_cache=False)
    assert another_country_1.capital._is_cleaned_up is False
    assert another_country_1.capital.mayor._is_cleaned_up is False

    assert country_1 is not country_2 is not another_country_1
    assert country_1.capital is not country_2.capital is not another_country_1.capital
    assert country_1.capital.mayor is not country_2.capital.mayor is not another_country_1.capital.mayor

    run_coroutine_sync(cleanup_exit_stack_of_func(get_country))

    assert country_1.capital._is_cleaned_up is True
    assert country_1.capital.mayor._is_cleaned_up is True  # type: ignore[unreachable]
    assert country_2.capital._is_cleaned_up is True
    assert country_2.capital.mayor._is_cleaned_up is True

    # Another country should not be cleaned up now
    assert another_country_1.capital._is_cleaned_up is False
    assert another_country_1.capital.mayor._is_cleaned_up is False

    run_coroutine_sync(cleanup_exit_stack_of_func(another_get_country))

    assert another_country_1.capital._is_cleaned_up is True
    assert another_country_1.capital.mayor._is_cleaned_up is True


def test_sync_and_async_generators_with_get_injected_obj_be_correctly_cleaned_up_by_cleanup_exit_stack_of_func(
    clean_exit_stack_manager: None,
) -> None:
    def get_mayor() -> Generator[Mayor, None, None]:
        mayor = Mayor()
        yield mayor
        mayor.cleanup()

    async def get_capital(
        mayor: Annotated[Mayor, Depends(get_mayor)],
    ) -> AsyncGenerator[Capital, None]:
        capital = Capital(mayor)
        yield capital
        capital.cleanup()

    async def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    def another_get_country(
        capital: Annotated[Capital, Depends(get_capital)],
    ) -> Country:
        return Country(capital)

    country_1: Country = get_injected_obj(get_country, use_cache=False)
    assert country_1.capital._is_cleaned_up is False
    assert country_1.capital.mayor._is_cleaned_up is False

    country_2: Country = get_injected_obj(get_country, use_cache=False)
    assert country_2.capital._is_cleaned_up is False
    assert country_2.capital.mayor._is_cleaned_up is False

    another_country_1: Country = get_injected_obj(another_get_country, use_cache=False)
    assert another_country_1.capital._is_cleaned_up is False
    assert another_country_1.capital.mayor._is_cleaned_up is False

    assert country_1 is not country_2 is not another_country_1
    assert country_1.capital is not country_2.capital is not another_country_1.capital
    assert country_1.capital.mayor is not country_2.capital.mayor is not another_country_1.capital.mayor

    run_coroutine_sync(cleanup_exit_stack_of_func(get_country))

    assert country_1.capital._is_cleaned_up is True
    assert country_1.capital.mayor._is_cleaned_up is True  # type: ignore[unreachable]
    assert country_2.capital._is_cleaned_up is True
    assert country_2.capital.mayor._is_cleaned_up is True

    # Another country should not be cleaned up now
    assert another_country_1.capital._is_cleaned_up is False
    assert another_country_1.capital.mayor._is_cleaned_up is False

    run_coroutine_sync(cleanup_exit_stack_of_func(another_get_country))

    assert another_country_1.capital._is_cleaned_up is True
    assert another_country_1.capital.mayor._is_cleaned_up is True


def test_function_with_non_dependency_parameters_and_dependencies_be_resolved_correctly() -> None:
    def get_mayor() -> Generator[Mayor, None, None]:
        mayor = Mayor()
        yield mayor
        mayor.cleanup()

    def get_capital(
        mayor: Annotated[Mayor, Depends(get_mayor)],
    ) -> Generator[Capital, None, None]:
        capital = Capital(mayor)
        yield capital
        capital.cleanup()

    @injectable(use_cache=False)
    def get_country(
        basic_str: str,
        basic_int: int,
        basic_bool: bool,
        basic_dict: dict[str, Any],
        capital: Annotated[Capital, Depends(get_capital)],
    ) -> Country:
        assert isinstance(basic_str, str)
        assert isinstance(basic_int, int)
        assert isinstance(basic_bool, bool)
        assert isinstance(basic_dict, dict)
        return Country(capital)

    country: Country = get_country(basic_str="basic_str", basic_int=1, basic_bool=True, basic_dict={"key": "value"})  # type: ignore  # noqa: PGH003
    assert country is not None


def test_get_injected_obj_with_dependency_override_sync(
    clean_exit_stack_manager: None,
) -> None:
    """Tests that get_injected_obj respects dependency_overrides for sync dependencies."""

    def sync_dependency_override() -> int:
        return 1

    def use_sync_dependency_override() -> int:
        return get_injected_obj(sync_dependency_override)

    loop_manager.set_loop_strategy(
        "background_thread"
    )  # To avoid affecting the FastAPI app and httpx client event loop
    app = FastAPI()

    @app.get("/")
    def read_root() -> int:
        return use_sync_dependency_override()

    mock_dependency = Mock(return_value=2)
    app.dependency_overrides[sync_dependency_override] = lambda: mock_dependency()
    run_coroutine_sync(register_app(app))

    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == 2
    mock_dependency.assert_called_once()


async def test_get_injected_obj_with_dependency_override_async(
    clean_exit_stack_manager: None,
) -> None:
    """Tests that get_injected_obj respects dependency_overrides for async dependencies."""

    async def async_dependency_override() -> int:
        return 1

    def use_async_dependency_override() -> int:
        return get_injected_obj(async_dependency_override)

    loop_manager.set_loop_strategy(
        "background_thread"
    )  # To avoid affecting the FastAPI app and httpx client event loop
    app = FastAPI()

    @app.get("/")
    async def read_root() -> int:
        return use_async_dependency_override()

    mock_dependency = Mock(return_value=2)
    app.dependency_overrides[async_dependency_override] = lambda: mock_dependency()
    await register_app(app)

    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == 2
    mock_dependency.assert_called_once()


async def test_get_injected_obj_with_dependency_override_sync_generator(
    clean_exit_stack_manager: None,
) -> None:
    """Tests that get_injected_obj respects dependency_overrides for sync generators."""
    sync_cleanup_mock_override = Mock()

    def sync_gen_dependency_override() -> Generator[int, None, None]:
        try:
            yield 1
        finally:
            sync_cleanup_mock_override()

    def use_sync_gen_dependency_override() -> int:
        return get_injected_obj(sync_gen_dependency_override)

    override_sync_cleanup_mock = Mock()

    def override_sync_gen() -> Generator[int, None, None]:
        try:
            yield 2
        finally:
            override_sync_cleanup_mock()

    loop_manager.set_loop_strategy(
        "background_thread"
    )  # To avoid affecting the FastAPI app and httpx client event loop
    app = FastAPI()

    @app.get("/")
    def read_root() -> int:
        return use_sync_gen_dependency_override()

    app.dependency_overrides[sync_gen_dependency_override] = override_sync_gen
    await register_app(app)

    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == 2

    await cleanup_all_exit_stacks()
    sync_cleanup_mock_override.assert_not_called()
    override_sync_cleanup_mock.assert_called_once()


async def test_get_injected_obj_with_dependency_override_async_generator(
    clean_exit_stack_manager: None,
) -> None:
    """Tests that get_injected_obj respects dependency_overrides for async generators."""
    async_cleanup_mock_override = Mock()

    async def async_gen_dependency_override() -> AsyncGenerator[int, None]:
        try:
            yield 1
        finally:
            async_cleanup_mock_override()

    def use_async_gen_dependency_override() -> int:
        return get_injected_obj(async_gen_dependency_override)

    override_async_cleanup_mock = Mock()

    async def override_async_gen() -> AsyncGenerator[int, None]:
        try:
            yield 2
        finally:
            override_async_cleanup_mock()

    loop_manager.set_loop_strategy(
        "background_thread"
    )  # To avoid affecting the FastAPI app and httpx client event loop
    app = FastAPI()

    @app.get("/")
    async def read_root() -> int:
        return use_async_gen_dependency_override()

    app.dependency_overrides[async_gen_dependency_override] = override_async_gen
    await register_app(app)

    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == 2

    await cleanup_all_exit_stacks()
    async_cleanup_mock_override.assert_not_called()
    override_async_cleanup_mock.assert_called_once()


def test_get_injected_obj_with_class(
    clean_exit_stack_manager: None,
) -> None:
    # Class should be supported since it's a callable
    class DummyClass:
        pass

    result = get_injected_obj(DummyClass)

    assert result is not None
    assert isinstance(result, DummyClass)


def test_get_injected_obj_with_sync_function(
    clean_exit_stack_manager: None,
) -> None:
    class DummyClass:
        pass

    def dummy_func() -> DummyClass:
        return DummyClass()

    result = get_injected_obj(dummy_func)

    assert result is not None
    assert isinstance(result, DummyClass)


def test_get_injected_obj_with_lambda_function(
    clean_exit_stack_manager: None,
) -> None:
    # Lambda function should be supported since it's a callable
    class DummyClass:
        pass

    result = get_injected_obj(lambda: DummyClass())

    assert result is not None
    assert isinstance(result, DummyClass)


def test_get_injected_obj_with_partial_function(
    clean_exit_stack_manager: None,
) -> None:
    # Partial function should be supported
    class DummyClass:
        def __init__(self, attr_1: int, attr_2: str) -> None:
            self.attr_1 = attr_1
            self.attr_2 = attr_2

    def dummy_func(attr_1: int, attr_2: str) -> DummyClass:
        return DummyClass(attr_1, attr_2)

    partial_func = partial(dummy_func, attr_1=1, attr_2="test")

    result = get_injected_obj(partial_func)

    assert result is not None
    assert isinstance(result, DummyClass)
    assert result.attr_1 == 1
    assert result.attr_2 == "test"


def test_get_injected_obj_sync_function_with_args_and_kwargs(
    clean_exit_stack_manager: None,
) -> None:
    # Partial function should be supported
    class DummyClass:
        def __init__(self, attr_1: int, attr_2: str) -> None:
            self.attr_1 = attr_1
            self.attr_2 = attr_2

    def dummy_func(attr_1: int, attr_2: str) -> DummyClass:
        return DummyClass(attr_1, attr_2)

    result = get_injected_obj(dummy_func, args=[1], kwargs={"attr_2": "test"})

    assert result is not None
    assert isinstance(result, DummyClass)
    assert result.attr_1 == 1
    assert result.attr_2 == "test"


def test_get_injected_obj_with_async_function(
    clean_exit_stack_manager: None,
) -> None:
    class DummyClass:
        pass

    async def dummy_func() -> DummyClass:
        return DummyClass()

    result = get_injected_obj(dummy_func)

    assert result is not None
    assert isinstance(result, DummyClass)


async def test_get_injected_obj_with_async_function_with_args_and_kwargs(
    clean_exit_stack_manager: None,
) -> None:
    class DummyClass:
        def __init__(self, attr_1: int, attr_2: str) -> None:
            self.attr_1 = attr_1
            self.attr_2 = attr_2

    async def dummy_func(attr_1: int, attr_2: str) -> DummyClass:
        return DummyClass(attr_1, attr_2)

    result = get_injected_obj(dummy_func, args=[1], kwargs={"attr_2": "test"})

    assert result is not None
    assert isinstance(result, DummyClass)
    assert result.attr_1 == 1
    assert result.attr_2 == "test"


def test_get_injected_obj_with_sync_generator(
    clean_exit_stack_manager: None,
) -> None:
    cleanup_mock = Mock()

    class DummyClass:
        pass

    def dummy_func() -> Generator[DummyClass, None, None]:
        try:
            yield DummyClass()
        finally:
            cleanup_mock()

    result = get_injected_obj(dummy_func)

    assert result is not None
    assert isinstance(result, DummyClass)

    run_coroutine_sync(cleanup_all_exit_stacks())
    cleanup_mock.assert_called_once()


async def test_get_injected_obj_with_async_generator(
    clean_exit_stack_manager: None,
) -> None:
    cleanup_mock = Mock()

    class DummyClass:
        pass

    async def dummy_func() -> AsyncGenerator[DummyClass, None]:
        try:
            yield DummyClass()
        finally:
            cleanup_mock()

    result = get_injected_obj(dummy_func)

    assert result is not None
    assert isinstance(result, DummyClass)

    await cleanup_all_exit_stacks()
    cleanup_mock.assert_called_once()


def test_get_injected_obj_with_sync_generator_with_args_and_kwargs(
    clean_exit_stack_manager: None,
) -> None:
    cleanup_mock = Mock()

    class DummyClass:
        def __init__(self, attr_1: int, attr_2: str) -> None:
            self.attr_1 = attr_1
            self.attr_2 = attr_2

    def dummy_func(attr_1: int, attr_2: str) -> Generator[DummyClass, None, None]:
        try:
            yield DummyClass(attr_1, attr_2)
        finally:
            cleanup_mock()

    result = get_injected_obj(dummy_func, args=[1], kwargs={"attr_2": "test"})

    assert result is not None
    assert isinstance(result, DummyClass)
    assert result.attr_1 == 1
    assert result.attr_2 == "test"

    run_coroutine_sync(cleanup_all_exit_stacks())
    cleanup_mock.assert_called_once()


async def test_get_injected_obj_with_async_generator_with_args_and_kwargs(
    clean_exit_stack_manager: None,
) -> None:
    cleanup_mock = Mock()

    class DummyClass:
        def __init__(self, attr_1: int, attr_2: str) -> None:
            self.attr_1 = attr_1
            self.attr_2 = attr_2

    async def dummy_func(attr_1: int, attr_2: str) -> AsyncGenerator[DummyClass, None]:
        try:
            yield DummyClass(attr_1, attr_2)
        finally:
            cleanup_mock()

    result = get_injected_obj(dummy_func, args=[1], kwargs={"attr_2": "test"})

    assert result is not None
    assert isinstance(result, DummyClass)
    assert result.attr_1 == 1
    assert result.attr_2 == "test"

    await cleanup_all_exit_stacks()
    cleanup_mock.assert_called_once()


async def test_async_get_injected_obj_with_async_function(
    clean_exit_stack_manager: None,
) -> None:
    class DummyClass:
        pass

    async def dummy_func() -> DummyClass:
        return DummyClass()

    result = await async_get_injected_obj(dummy_func)

    assert result is not None
    assert isinstance(result, DummyClass)


async def test_async_get_injected_obj_with_async_function_with_args_and_kwargs(
    clean_exit_stack_manager: None,
) -> None:
    class DummyClass:
        def __init__(self, attr_1: int, attr_2: str) -> None:
            self.attr_1 = attr_1
            self.attr_2 = attr_2

    async def dummy_func(attr_1: int, attr_2: str) -> DummyClass:
        return DummyClass(attr_1, attr_2)

    result = await async_get_injected_obj(dummy_func, args=[1], kwargs={"attr_2": "test"})

    assert result is not None
    assert isinstance(result, DummyClass)
    assert result.attr_1 == 1
    assert result.attr_2 == "test"


async def test_async_get_injected_obj_with_async_generator(
    clean_exit_stack_manager: None,
) -> None:
    cleanup_mock = Mock()

    class DummyClass:
        pass

    async def dummy_func() -> AsyncGenerator[DummyClass, None]:
        try:
            yield DummyClass()
        finally:
            cleanup_mock()

    result = await async_get_injected_obj(dummy_func)

    assert result is not None
    assert isinstance(result, DummyClass)

    await cleanup_all_exit_stacks()
    cleanup_mock.assert_called_once()


async def test_async_get_injected_obj_with_async_generator_with_args_and_kwargs(
    clean_exit_stack_manager: None,
) -> None:
    cleanup_mock = Mock()

    class DummyClass:
        def __init__(self, attr_1: int, attr_2: str) -> None:
            self.attr_1 = attr_1
            self.attr_2 = attr_2

    async def dummy_func(attr_1: int, attr_2: str) -> AsyncGenerator[DummyClass, None]:
        try:
            yield DummyClass(attr_1, attr_2)
        finally:
            cleanup_mock()

    result = await async_get_injected_obj(dummy_func, args=[1], kwargs={"attr_2": "test"})

    assert result is not None
    assert isinstance(result, DummyClass)
    assert result.attr_1 == 1
    assert result.attr_2 == "test"

    await cleanup_all_exit_stacks()
    cleanup_mock.assert_called_once()


async def test_async_get_injected_obj_with_dependencies(
    clean_exit_stack_manager: None,
) -> None:
    class Service:
        def __init__(self, value: str) -> None:
            self.value = value

    async def get_base_value() -> str:
        return "base"

    async def get_service(base: Annotated[str, Depends(get_base_value)]) -> Service:
        return Service(value=f"{base}_service")

    service = await async_get_injected_obj(get_service)
    assert isinstance(service, Service)
    assert service.value == "base_service"


async def test_async_get_injected_obj_in_running_loop(
    clean_exit_stack_manager: None,
) -> None:
    """Test that async_get_injected_obj works in an already running event loop.

    This is the key test case that demonstrates the fix for issue #173.
    """

    class Service:
        def __init__(self, value: str) -> None:
            self.value = value

    async def get_service() -> Service:
        return Service(value="loop_service")

    # Simulate a running loop scenario (like Kafka consumer callback)
    async def consumer_callback() -> Service:
        # This would fail with get_injected_obj() but works with async_get_injected_obj()
        return await async_get_injected_obj(get_service)

    # The test itself is running in an event loop
    service = await consumer_callback()
    assert isinstance(service, Service)
    assert service.value == "loop_service"


async def test_async_get_injected_obj_with_nested_dependencies(
    clean_exit_stack_manager: None,
) -> None:
    class Service:
        def __init__(self, value: str) -> None:
            self.value = value

    async def get_config() -> dict[str, str]:
        return {"prefix": "nested"}

    async def get_base_service(
        config: Annotated[dict[str, str], Depends(get_config)],
    ) -> Service:
        return Service(value=config["prefix"])

    async def get_dependent_service(
        base: Annotated[Service, Depends(get_base_service)],
    ) -> Service:
        return Service(value=f"{base.value}_dependent")

    service = await async_get_injected_obj(get_dependent_service)
    assert isinstance(service, Service)
    assert service.value == "nested_dependent"


async def test_async_get_injected_obj_with_cache(
    clean_exit_stack_manager: None,
) -> None:
    call_count = 0

    class Service:
        def __init__(self, value: str) -> None:
            self.value = value

    async def get_service() -> Service:
        nonlocal call_count
        call_count += 1
        return Service(value=f"cached_{call_count}")

    # First call with cache enabled
    service1 = await async_get_injected_obj(get_service, use_cache=True)
    assert service1.value == "cached_1"

    # Second call should use cache
    service2 = await async_get_injected_obj(get_service, use_cache=True)
    assert service2.value == "cached_1"
    assert call_count == 1  # Should still be 1 because of caching


async def test_async_get_injected_obj_without_cache(
    clean_exit_stack_manager: None,
) -> None:
    call_count = 0

    class Service:
        def __init__(self, value: str) -> None:
            self.value = value

    async def get_service() -> Service:
        nonlocal call_count
        call_count += 1
        return Service(value=f"uncached_{call_count}")

    # First call without cache
    service1 = await async_get_injected_obj(get_service, use_cache=False)
    assert service1.value == "uncached_1"

    # Second call should create new instance
    service2 = await async_get_injected_obj(get_service, use_cache=False)
    assert service2.value == "uncached_2"
    assert call_count == 2


def test_get_injected_obj_with_arg_override_sync(
    clean_exit_stack_manager: None,
) -> None:
    """Verify that kwargs override works for sync functions even with Depends."""

    def dep_a() -> int:
        return 1

    def dep_b() -> int:
        return 2

    def sync_func(a: Annotated[int, Depends(dep_a)], b: Annotated[int, Depends(dep_b)]) -> tuple[int, int]:
        return a, b

    # Default behavior
    assert get_injected_obj(sync_func) == (1, 2)

    # Override 'b'
    assert get_injected_obj(sync_func, kwargs={"b": 6}) == (1, 6)

    # Override 'a' and 'b'
    assert get_injected_obj(sync_func, kwargs={"a": 5, "b": 6}) == (5, 6)

    # Positional args (if supported by get_injected_obj which uses partial)
    # partial(sync_func, 5) -> a=5
    assert get_injected_obj(sync_func, args=[5]) == (5, 2)


async def test_get_injected_obj_with_arg_override_async(
    clean_exit_stack_manager: None,
) -> None:
    """Verify that kwargs override works for async functions even with Depends."""

    def dep_a() -> int:
        return 1

    def dep_b() -> int:
        return 2

    async def async_func(a: Annotated[int, Depends(dep_a)], b: Annotated[int, Depends(dep_b)]) -> tuple[int, int]:
        return a, b

    # Default behavior
    assert await async_get_injected_obj(async_func) == (1, 2)

    # Override 'b'
    assert await async_get_injected_obj(async_func, kwargs={"b": 6}) == (1, 6)

    # Override 'a' and 'b'
    assert await async_get_injected_obj(async_func, kwargs={"a": 5, "b": 6}) == (5, 6)

    # Positional args
    assert await async_get_injected_obj(async_func, args=[5]) == (5, 2)


def test_get_injected_obj_with_nested_partial(clean_exit_stack_manager: None) -> None:
    """Verify that it works even if the user passes a pre-curried partial."""

    def dep_a() -> int:
        return 1

    def dep_b() -> int:
        return 2

    def sync_func(a: Annotated[int, Depends(dep_a)], b: Annotated[int, Depends(dep_b)]) -> tuple[int, int]:
        return a, b

    # User creates a partial
    p = partial(sync_func, b=99)
    # And tries to inject it.
    # Note: get_injected_obj doesn't take 'args'/'kwargs' here, just the func.
    # The fix should detect it's a partial and fix signature.
    assert get_injected_obj(p) == (1, 99)

    # Nested partial (curry a, then b)
    p2 = partial(partial(sync_func, a=88), b=99)
    assert get_injected_obj(p2) == (88, 99)


def test_async_generators_within_async_dependencies_with_http_connection_import_be_resolved_correctly(
    clean_exit_stack_manager: None,
) -> None:
    """Ref: https://github.com/JasperSui/fastapi-injectable/issues/191"""  # noqa: D415
    from fastapi.requests import HTTPConnection  # noqa: F401

    @asynccontextmanager
    async def _get_dep_inner() -> AsyncIterator[str]:
        yield "hello, world"

    async def get_dep() -> AsyncIterator[str]:
        async with _get_dep_inner() as dep:
            yield dep

    @injectable
    async def func(dep: Annotated[str, Depends(get_dep)]) -> str:
        return dep

    result = run_coroutine_sync(func())  # type: ignore  # noqa: PGH003
    assert result == "hello, world"


def test_imbricated_async_generators_within_async_dependencies_with_http_connection_import_be_resolved_correctly(
    clean_exit_stack_manager: None,
) -> None:
    """Ref: https://github.com/JasperSui/fastapi-injectable/issues/199"""  # noqa: D415
    from fastapi.requests import HTTPConnection  # noqa: F401

    @asynccontextmanager
    async def _get_dep1_inner() -> AsyncIterator[str]:
        yield "hello, world"

    async def get_dep1() -> AsyncIterator[str]:
        async with _get_dep1_inner() as dep1:
            yield dep1

    @asynccontextmanager
    async def _get_dep2_inner(dep1: str) -> AsyncIterator[str]:
        yield f"{dep1}, lol"

    async def get_dep2(dep1: Annotated[str, Depends(get_dep1)]) -> AsyncIterator[str]:
        async with _get_dep2_inner(dep1) as dep2:
            yield dep2

    @injectable
    async def func(dep2: Annotated[str, Depends(get_dep2)]) -> str:
        return dep2

    result = run_coroutine_sync(func())  # type: ignore  # noqa: PGH003
    assert result == "hello, world, lol"
