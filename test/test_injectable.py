# type: ignore  # noqa: PGH003
from inspect import signature
from typing import Annotated, Generic, TypeVar

from fastapi import Depends

from src.fastapi_injectable.decorator import injectable


class Mayor:
    pass


class Capital:
    def __init__(self, mayor: Mayor) -> None:
        self.mayor = mayor


class Country:
    def __init__(self, capital: Capital) -> None:
        self.capital = capital


T = TypeVar("T", bound=Country | Capital | Mayor)


class Assembly(Generic[T]):
    def __init__(self, members: set[T]) -> None:
        self.members = members


def test_injectable_sync_only_decorator_with_cache() -> None:
    def get_mayor() -> Mayor:
        return Mayor()

    def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Capital:
        return Capital(mayor)

    @injectable
    def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    country_1 = get_country()
    country_2 = get_country()
    country_3 = get_country()
    assert country_1.capital is country_2.capital is country_3.capital
    assert country_1.capital.mayor is country_2.capital.mayor is country_3.capital.mayor


def test_injectable_sync_only_decorator_without_cache() -> None:
    def get_mayor() -> Mayor:
        return Mayor()

    def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Capital:
        return Capital(mayor)

    @injectable(use_cache=False)
    def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    country_1 = get_country()
    country_2 = get_country()
    country_3 = get_country()
    assert country_1.capital is not country_2.capital is not country_3.capital
    assert country_1.capital.mayor is not country_2.capital.mayor is not country_3.capital.mayor


def test_injectable_sync_only_wrap_function_with_cache() -> None:
    def get_mayor() -> Mayor:
        return Mayor()

    def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Capital:
        return Capital(mayor)

    def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    injectable_get_country = injectable(get_country)
    country_1 = injectable_get_country()
    country_2 = injectable_get_country()
    country_3 = injectable_get_country()
    assert country_1.capital is country_2.capital is country_3.capital
    assert country_1.capital.mayor is country_2.capital.mayor is country_3.capital.mayor


def test_injectable_sync_only_wrap_function_without_cache() -> None:
    def get_mayor() -> Mayor:
        return Mayor()

    def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Capital:
        return Capital(mayor)

    def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    injectable_get_country = injectable(get_country, use_cache=False)
    country_1 = injectable_get_country()
    country_2 = injectable_get_country()
    country_3 = injectable_get_country()
    assert country_1.capital is not country_2.capital is not country_3.capital
    assert country_1.capital.mayor is not country_2.capital.mayor is not country_3.capital.mayor


async def test_injectable_async_only_decorator_with_cache() -> None:
    async def get_mayor() -> Mayor:
        return Mayor()

    async def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Capital:
        return Capital(mayor)

    @injectable
    async def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    country_1 = await get_country()
    country_2 = await get_country()
    country_3 = await get_country()
    assert country_1.capital is country_2.capital is country_3.capital
    assert country_1.capital.mayor is country_2.capital.mayor is country_3.capital.mayor


async def test_injectable_async_only_decorator_without_cache() -> None:
    async def get_mayor() -> Mayor:
        return Mayor()

    async def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Capital:
        return Capital(mayor)

    @injectable(use_cache=False)
    async def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    country_1 = await get_country()
    country_2 = await get_country()
    country_3 = await get_country()
    assert country_1.capital is not country_2.capital is not country_3.capital
    assert country_1.capital.mayor is not country_2.capital.mayor is not country_3.capital.mayor


async def test_injectable_async_only_wrap_function_with_cache() -> None:
    async def get_mayor() -> Mayor:
        return Mayor()

    async def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Capital:
        return Capital(mayor)

    async def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    injectable_get_country = injectable(get_country)
    country_1 = await injectable_get_country()
    country_2 = await injectable_get_country()
    country_3 = await injectable_get_country()
    assert country_1.capital is country_2.capital is country_3.capital
    assert country_1.capital.mayor is country_2.capital.mayor is country_3.capital.mayor


async def test_injectable_async_only_wrap_function_without_cache() -> None:
    async def get_mayor() -> Mayor:
        return Mayor()

    async def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Capital:
        return Capital(mayor)

    async def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    injectable_get_country = injectable(get_country, use_cache=False)
    country_1 = await injectable_get_country()
    country_2 = await injectable_get_country()
    country_3 = await injectable_get_country()
    assert country_1.capital is not country_2.capital is not country_3.capital
    assert country_1.capital.mayor is not country_2.capital.mayor is not country_3.capital.mayor


async def test_injectable_async_with_sync_decorator_with_cache() -> None:
    def get_mayor() -> Mayor:
        return Mayor()

    def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Capital:
        return Capital(mayor)

    @injectable
    async def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    country_1 = await get_country()
    country_2 = await get_country()
    country_3 = await get_country()
    assert country_1.capital is country_2.capital is country_3.capital
    assert country_1.capital.mayor is country_2.capital.mayor is country_3.capital.mayor


async def test_injectable_async_with_sync_decorator_without_cache() -> None:
    def get_mayor() -> Mayor:
        return Mayor()

    def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Capital:
        return Capital(mayor)

    @injectable(use_cache=False)
    async def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    country_1 = await get_country()
    country_2 = await get_country()
    country_3 = await get_country()
    assert country_1.capital is not country_2.capital is not country_3.capital
    assert country_1.capital.mayor is not country_2.capital.mayor is not country_3.capital.mayor


async def test_injectable_async_with_sync_wrap_function_with_cache() -> None:
    def get_mayor() -> Mayor:
        return Mayor()

    def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Capital:
        return Capital(mayor)

    async def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    injectable_get_country = injectable(get_country)
    country_1 = await injectable_get_country()
    country_2 = await injectable_get_country()
    country_3 = await injectable_get_country()
    assert country_1.capital is country_2.capital is country_3.capital
    assert country_1.capital.mayor is country_2.capital.mayor is country_3.capital.mayor


async def test_injectable_async_with_sync_wrap_function_without_cache() -> None:
    def get_mayor() -> Mayor:
        return Mayor()

    def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Capital:
        return Capital(mayor)

    async def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    injectable_get_country = injectable(get_country, use_cache=False)
    country_1 = await injectable_get_country()
    country_2 = await injectable_get_country()
    country_3 = await injectable_get_country()
    assert country_1.capital is not country_2.capital is not country_3.capital
    assert country_1.capital.mayor is not country_2.capital.mayor is not country_3.capital.mayor


def test_injectable_sync_with_async_decorator_with_cache() -> None:
    async def get_mayor() -> Mayor:
        return Mayor()

    def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Capital:
        return Capital(mayor)

    @injectable
    def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    country_1 = get_country()
    country_2 = get_country()
    country_3 = get_country()
    assert country_1.capital is country_2.capital is country_3.capital
    assert country_1.capital.mayor is country_2.capital.mayor is country_3.capital.mayor


def test_injectable_sync_with_async_decorator_without_cache() -> None:
    async def get_mayor() -> Mayor:
        return Mayor()

    def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Capital:
        return Capital(mayor)

    @injectable(use_cache=False)
    def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    country_1 = get_country()
    country_2 = get_country()
    country_3 = get_country()
    assert country_1.capital is not country_2.capital is not country_3.capital
    assert country_1.capital.mayor is not country_2.capital.mayor is not country_3.capital.mayor


def test_injectable_sync_with_async_wrap_function_with_cache() -> None:
    async def get_mayor() -> Mayor:
        return Mayor()

    def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Capital:
        return Capital(mayor)

    def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    injectable_get_country = injectable(get_country)
    country_1 = injectable_get_country()
    country_2 = injectable_get_country()
    country_3 = injectable_get_country()
    assert country_1.capital is country_2.capital is country_3.capital
    assert country_1.capital.mayor is country_2.capital.mayor is country_3.capital.mayor


def test_injectable_sync_with_async_wrap_function_without_cache() -> None:
    async def get_mayor() -> Mayor:
        return Mayor()

    def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Capital:
        return Capital(mayor)

    def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    injectable_get_country = injectable(get_country, use_cache=False)
    country_1 = injectable_get_country()
    country_2 = injectable_get_country()
    country_3 = injectable_get_country()
    assert country_1.capital is not country_2.capital is not country_3.capital
    assert country_1.capital.mayor is not country_2.capital.mayor is not country_3.capital.mayor


def test_injectable_converts_depends_to_dynamic_types() -> None:
    async def get_mayor() -> Mayor:
        return Mayor()

    @injectable(use_cache=True)
    def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Capital:
        return Capital(mayor)

    sig = signature(get_capital)
    param = next(iter(sig.parameters.values()))

    assert type(param.default).__name__ == "Injected_Mayor"


def test_injectable_converts_annotated_generic_depends_to_dynamic_types() -> None:
    def get_mayor() -> Mayor:
        return Mayor()

    def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Capital:
        return Capital(mayor)

    def get_country(capital: Annotated[Capital, Depends(get_capital, use_cache=False)]) -> Country:
        return Country(capital)

    def get_country_assembly(
        country_a: Annotated[Country, Depends(get_country, use_cache=False)],
        country_b: Annotated[Country, Depends(get_country, use_cache=False)],
    ) -> Assembly[Country]:
        return Assembly(members=[country_a, country_b])

    @injectable
    def injectable_function(assembly: Annotated[Assembly[Country], Depends(get_country_assembly)]) -> Assembly[Country]:
        return assembly

    sig = signature(injectable_function)
    param = next(iter(sig.parameters.values()))

    assert type(param.default).__name__ == "Injected_Assembly"


async def test_injectable_async_generator_and_decorator_with_cache() -> None:
    async def get_mayor() -> Mayor:
        return Mayor()

    async def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Capital:
        return Capital(mayor)

    @injectable(use_cache=True)
    async def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        yield Country(capital)

    async for country in get_country():
        country_1 = country

    async for country in get_country():
        country_2 = country
    assert country_1.capital is country_2.capital
    assert country_1.capital.mayor is country_2.capital.mayor


def test_injectable_sync_generator_and_decorator_with_cache() -> None:
    def get_mayor() -> Mayor:
        return Mayor()

    def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Capital:
        return Capital(mayor)

    @injectable(use_cache=True)
    def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        yield Country(capital)

    for country in get_country():
        country_1 = country

    for country in get_country():
        country_2 = country
    assert country_1.capital is country_2.capital
    assert country_1.capital.mayor is country_2.capital.mayor
