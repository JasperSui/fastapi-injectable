# type: ignore  # noqa: PGH003
"""Regression tests for https://github.com/JasperSui/fastapi-injectable/issues/268.

This module deliberately uses ``from __future__ import annotations`` so every
annotation below is a string until it is evaluated.
"""

from __future__ import annotations

import inspect
from functools import partial
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends

from src.fastapi_injectable.decorator import injectable
from src.fastapi_injectable.main import _get_signature, resolve_dependencies

if TYPE_CHECKING:
    from collections.abc import Sequence


class Mayor:
    pass


def get_mayor() -> Mayor:
    return Mayor()


def get_number() -> int:
    return 41


async def test_injectable_async_resolves_string_annotated_depends() -> None:
    @injectable
    async def job(number: Annotated[int, Depends(get_number)]) -> int:
        return number + 1

    assert await job() == 42


def test_injectable_sync_resolves_string_annotated_depends() -> None:
    @injectable
    def job(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Mayor:
        return mayor

    assert isinstance(job(), Mayor)


def test_injectable_keeps_non_dependency_parameters() -> None:
    @injectable
    def job(name: str, number: Annotated[int, Depends(get_number)]) -> str:
        return f"{name}-{number}"

    assert job("job") == "job-41"


def test_injectable_with_type_checking_only_annotation() -> None:
    # ``Sequence`` exists only under TYPE_CHECKING, so evaluating every
    # annotation at once raises NameError; the Depends parameter must still be found.
    @injectable
    def job(items: Sequence[int], number: Annotated[int, Depends(get_number)]) -> Sequence[int]:
        return [*items, number]

    assert job([1]) == [1, 41]


async def test_resolve_dependencies_with_string_annotations() -> None:
    def func(name: str, number: Annotated[int, Depends(get_number)]) -> None:
        return None

    assert await resolve_dependencies(func) == {"number": 41}


def test_get_signature_leaves_unresolvable_annotations_as_strings() -> None:
    def func(items: Sequence[int], number: Annotated[int, Depends(get_number)]) -> Sequence[int]:
        return items

    params = _get_signature(func).parameters

    assert params["items"].annotation == "Sequence[int]"
    assert params["number"].annotation == Annotated[int, Depends(get_number)]
    assert inspect.signature(func).parameters["number"].annotation == "Annotated[int, Depends(get_number)]"


def test_get_signature_evaluates_annotations_of_partial() -> None:
    def func(bound: int, items: Sequence[int], number: Annotated[int, Depends(get_number)], untyped) -> None:  # noqa: ANN001
        return None

    params = _get_signature(partial(func, 1)).parameters

    assert list(params) == ["items", "number", "untyped"]
    assert params["items"].annotation == "Sequence[int]"
    assert params["number"].annotation == Annotated[int, Depends(get_number)]
