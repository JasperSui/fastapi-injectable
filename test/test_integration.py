from collections.abc import AsyncGenerator, Generator
from typing import Annotated, Any

import pytest
from fastapi import Depends

from src.fastapi_injectable.concurrency import run_coroutine_sync
from src.fastapi_injectable.decorator import injectable
from src.fastapi_injectable.util import cleanup_all_exit_stacks, cleanup_exit_stack_of_func, get_injected_obj


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

    def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Generator[Capital, None, None]:
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

    run_coroutine_sync(cleanup_all_exit_stacks())

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

    async def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> AsyncGenerator[Capital, None]:
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

    async def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> AsyncGenerator[Capital, None]:
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

    def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Generator[Capital, None, None]:
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

    async def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> AsyncGenerator[Capital, None]:
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

    async def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> AsyncGenerator[Capital, None]:
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

    def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Generator[Capital, None, None]:
        capital = Capital(mayor)
        yield capital
        capital.cleanup()

    @injectable(use_cache=False)
    def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    @injectable(use_cache=False)
    def another_get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
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

    async def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> AsyncGenerator[Capital, None]:
        capital = Capital(mayor)
        yield capital
        capital.cleanup()

    @injectable(use_cache=False)
    async def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    @injectable(use_cache=False)
    def another_get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
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

    async def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> AsyncGenerator[Capital, None]:
        capital = Capital(mayor)
        yield capital
        capital.cleanup()

    @injectable(use_cache=False)
    async def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    @injectable(use_cache=False)
    def another_get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
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

    def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Generator[Capital, None, None]:
        capital = Capital(mayor)
        yield capital
        capital.cleanup()

    def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    def another_get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
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

    async def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> AsyncGenerator[Capital, None]:
        capital = Capital(mayor)
        yield capital
        capital.cleanup()

    async def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    def another_get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
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

    async def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> AsyncGenerator[Capital, None]:
        capital = Capital(mayor)
        yield capital
        capital.cleanup()

    async def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    def another_get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
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

    def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Generator[Capital, None, None]:
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
