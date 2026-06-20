"""Tests for the fastapi-injectable mypy plugin.

These drive the real type checker (via ``mypy.api.run``) over small fixture
modules and assert the plugin's effect on the *caller's* view of an
``@injectable``-decorated function. The plugin executes inside mypy's own run, so
it is never exercised by coverage (and is omitted in ``pyproject.toml``); this
file is the plugin's only safety net.

mypy ships ``mypyc``-compiled wheels that do not target free-threaded builds, so
it is not installed there -- ``importorskip`` makes those interpreters skip
cleanly instead of erroring at collection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

api = pytest.importorskip("mypy.api")


_PREAMBLE = (
    "from typing import Annotated\n"
    "\n"
    "from fastapi import Depends\n"
    "\n"
    "from fastapi_injectable import injectable\n"
    "\n"
    "\n"
    "class Capital:\n"
    "    pass\n"
    "\n"
    "\n"
    "class Country:\n"
    "    pass\n"
    "\n"
    "\n"
    "def get_capital() -> Capital:\n"
    "    return Capital()\n"
)

_MYPY_CONFIG = "[mypy]\nplugins = fastapi_injectable.mypy\n"

_VALID_CALLS = 'make_country("Taiwan")\nmake_country("Taiwan", capital=Capital())\n'
_POSITIONAL_DEPENDENCY_CALL = 'make_country("Taiwan", Capital())  # positional-dependency\n'
_MISSING_REQUIRED_CALL = "make_country()  # missing-required-arg\n"


def _source(decorator: str, calls: str) -> str:
    """Build a fixture module decorating ``make_country`` with ``decorator``."""
    return (
        _PREAMBLE
        + "\n"
        + f"@{decorator}\n"
        + "def make_country(name: str, capital: Annotated[Capital, Depends(get_capital)]) -> Country:\n"
        + "    return Country()\n"
        + "\n\n"
        + calls
    )


def _run_mypy(tmp_path: Path, source: str) -> tuple[int, str]:
    """Type-check ``source`` with the plugin enabled; return (exit_status, stdout)."""
    case = tmp_path / "case.py"
    case.write_text(source)
    config = tmp_path / "mypy.ini"
    config.write_text(_MYPY_CONFIG)

    stdout, stderr, status = api.run(
        [
            "--config-file",
            str(config),
            "--no-incremental",
            "--cache-dir",
            str(tmp_path / ".mypy_cache"),
            "--no-error-summary",
            "--no-color-output",
            str(case),
        ]
    )
    assert "Traceback" not in stderr, f"the plugin crashed:\n{stderr}\n{stdout}"
    return status, stdout


def _errors(stdout: str) -> list[tuple[int, str]]:
    """Parse ``case.py:<line>: error: ...`` lines into (line_number, text) pairs."""
    parsed: list[tuple[int, str]] = []
    for line in stdout.splitlines():
        if ".py:" not in line or ": error:" not in line:
            continue
        lineno = int(line.split(".py:", 1)[1].split(":", 1)[0])
        parsed.append((lineno, line))
    return parsed


def _line_of(source: str, needle: str) -> int:
    """Return the 1-based line number of the first line containing ``needle``."""
    return next(lineno for lineno, line in enumerate(source.splitlines(), start=1) if needle in line)


def _assert_clean(tmp_path: Path, source: str) -> None:
    status, stdout = _run_mypy(tmp_path, source)
    assert status == 0, f"expected a clean type-check, got:\n{stdout}"
    assert _errors(stdout) == [], f"expected no errors, got:\n{stdout}"


def _assert_rejects(tmp_path: Path, source: str, marker: str) -> None:
    status, stdout = _run_mypy(tmp_path, source)
    bad_line = _line_of(source, marker)
    assert status == 1, f"expected a type error at {marker!r}, got status {status}:\n{stdout}"
    assert any(
        lineno == bad_line for lineno, _ in _errors(stdout)
    ), f"expected an error on line {bad_line} ({marker!r}), got:\n{stdout}"


def test_bare_injectable_allows_omitting_or_keywording_dependency(tmp_path: Path) -> None:
    _assert_clean(tmp_path, _source("injectable", _VALID_CALLS))


def test_parametrized_injectable_allows_omitting_or_keywording_dependency(tmp_path: Path) -> None:
    _assert_clean(tmp_path, _source("injectable(use_cache=False)", _VALID_CALLS))


def test_bare_injectable_rejects_positional_dependency(tmp_path: Path) -> None:
    _assert_rejects(tmp_path, _source("injectable", _POSITIONAL_DEPENDENCY_CALL), "# positional-dependency")


def test_parametrized_injectable_rejects_positional_dependency(tmp_path: Path) -> None:
    _assert_rejects(
        tmp_path,
        _source("injectable(use_cache=False)", _POSITIONAL_DEPENDENCY_CALL),
        "# positional-dependency",
    )


def test_injectable_still_requires_non_dependency_arguments(tmp_path: Path) -> None:
    _assert_rejects(tmp_path, _source("injectable", _MISSING_REQUIRED_CALL), "# missing-required-arg")
