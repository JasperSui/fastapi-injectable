# type: ignore  # noqa: PGH003
from inspect import signature
from typing import Annotated, Generic, TypeVar

from fastapi import Depends

from src.fastapi_injectable.decorator import injectable
from src.fastapi_injectable.main import _build_dependency_only_callable, _has_depends


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


def test_injectable_converts_depends_with_optional_uniontype_types() -> None:
    async def get_mayor() -> Mayor | None:
        return None

    @injectable(use_cache=True)
    def get_capital(mayor: Annotated[Mayor | None, Depends(get_mayor)]) -> Capital | None:
        return Capital(mayor) if mayor else None

    sig = signature(get_capital)
    param = next(iter(sig.parameters.values()))

    assert type(param.default).__name__.startswith("Injected")


def test_injectable_sync_override_country() -> None:
    def get_mayor() -> Mayor:
        return Mayor()

    def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Capital:
        return Capital(mayor)

    @injectable
    def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    # automatic DI
    country_injected = get_country()
    assert isinstance(country_injected, Country)
    assert isinstance(country_injected.capital, Capital)
    assert isinstance(country_injected.capital.mayor, Mayor)

    # manual override
    capital = Capital(Mayor())
    country_manual = get_country(capital=capital)
    assert country_manual.capital is capital
    assert country_manual.capital.mayor is capital.mayor


async def test_injectable_async_override_country() -> None:
    async def get_mayor() -> Mayor:
        return Mayor()

    async def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Capital:
        return Capital(mayor)

    @injectable
    async def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    country_injected = await get_country()
    assert isinstance(country_injected, Country)
    assert isinstance(country_injected.capital, Capital)
    assert isinstance(country_injected.capital.mayor, Mayor)

    capital = Capital(Mayor())
    country_manual = await get_country(capital=capital)
    assert country_manual.capital is capital
    assert country_manual.capital.mayor is capital.mayor


async def test_injectable_async_gen_override_country() -> None:
    async def get_mayor() -> Mayor:
        return Mayor()

    async def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Capital:
        return Capital(mayor)

    @injectable
    async def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        yield Country(capital)

    country_injected: Country | None = None
    async for c in get_country():
        country_injected = c
        break
    assert isinstance(country_injected, Country)
    assert isinstance(country_injected.capital, Capital)
    assert isinstance(country_injected.capital.mayor, Mayor)

    capital = Capital(Mayor())
    country_manual: Country | None = None
    async for c in get_country(capital=capital):
        country_manual = c
        break
    assert country_manual.capital is capital
    assert country_manual.capital.mayor is capital.mayor


class NonPydanticType:
    """A class that is NOT a valid Pydantic field type, simulating Celery's Task class."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass


def test_injectable_with_non_pydantic_positional_param() -> None:
    """Regression test for #214: non-Pydantic params (e.g. Celery bound task) should not break injection."""

    def get_mayor() -> Mayor:
        return Mayor()

    def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Capital:
        return Capital(mayor)

    @injectable
    def my_task(self: NonPydanticType, capital: Annotated[Capital, Depends(get_capital)]) -> Country:
        return Country(capital)

    task_instance = NonPydanticType()
    country = my_task(task_instance)
    assert isinstance(country, Country)
    assert isinstance(country.capital, Capital)
    assert isinstance(country.capital.mayor, Mayor)


def test_injectable_with_non_pydantic_positional_param_and_regular_args() -> None:
    """Ensure non-dependency params of mixed types work alongside Depends()."""

    def get_mayor() -> Mayor:
        return Mayor()

    @injectable
    def my_task(self: NonPydanticType, arg: int, mayor: Annotated[Mayor, Depends(get_mayor)]) -> Mayor:
        assert isinstance(arg, int)
        return mayor

    task_instance = NonPydanticType()
    mayor = my_task(task_instance, 42)
    assert isinstance(mayor, Mayor)


def test_build_dependency_only_callable_filters_non_depends_params() -> None:
    """_build_dependency_only_callable should strip non-Depends parameters."""
    import inspect

    def get_mayor() -> Mayor:
        return Mayor()

    def func(self: NonPydanticType, arg: int, dep: Annotated[Mayor, Depends(get_mayor)]) -> None:
        pass

    stub = _build_dependency_only_callable(func)
    assert stub is not func
    sig = inspect.signature(stub)
    assert list(sig.parameters.keys()) == ["dep"]


def test_build_dependency_only_callable_returns_func_when_all_are_depends() -> None:
    """When all params are Depends, the original function is returned as-is."""

    def get_mayor() -> Mayor:
        return Mayor()

    def func(dep: Annotated[Mayor, Depends(get_mayor)]) -> None:
        pass

    result = _build_dependency_only_callable(func)
    assert result is func


def test_has_depends_with_annotated_depends() -> None:
    import inspect

    def get_mayor() -> Mayor:
        return Mayor()

    def func(dep: Annotated[Mayor, Depends(get_mayor)]) -> None:
        pass

    sig = inspect.signature(func)
    param = next(iter(sig.parameters.values()))
    assert _has_depends(param) is True


def test_has_depends_with_plain_param() -> None:
    import inspect

    def func(x: int) -> None:
        pass

    sig = inspect.signature(func)
    param = next(iter(sig.parameters.values()))
    assert _has_depends(param) is False


def test_has_depends_with_annotated_non_depends_metadata() -> None:
    import inspect

    def func(x: Annotated[int, "some_metadata"]) -> None:
        pass

    sig = inspect.signature(func)
    param = next(iter(sig.parameters.values()))
    assert _has_depends(param) is False


def test_has_depends_with_default_depends() -> None:
    import inspect

    def get_mayor() -> Mayor:
        return Mayor()

    def func(dep: Mayor = Depends(get_mayor)) -> None:
        pass

    sig = inspect.signature(func)
    param = next(iter(sig.parameters.values()))
    assert _has_depends(param) is True
