import importlib.metadata

import pytest
from fastapi.dependencies.utils import solve_dependencies

from src.fastapi_injectable.exception import FastAPICompatibilityError
from src.fastapi_injectable.main import _verify_fastapi_compatibility


def test_verify_fastapi_compatibility_passes_for_installed_fastapi() -> None:
    # The installed FastAPI is within the supported range, so the probe is a no-op.
    _verify_fastapi_compatibility(solve_dependencies)


def test_verify_fastapi_compatibility_passes_when_all_params_present() -> None:
    def fake_solve(
        *,
        dependency_overrides_provider: object = None,
        dependency_cache: object = None,
        async_exit_stack: object = None,
        embed_body_fields: bool = False,
    ) -> None:
        """A stub exposing every keyword param we rely on."""

    _verify_fastapi_compatibility(fake_solve)


def test_verify_fastapi_compatibility_reports_only_missing_params() -> None:
    def fake_solve(
        *,
        dependency_overrides_provider: object = None,
        dependency_cache: object = None,
        async_exit_stack: object = None,
    ) -> None:
        """A stub missing only ``embed_body_fields``."""

    with pytest.raises(FastAPICompatibilityError) as exc_info:
        _verify_fastapi_compatibility(fake_solve)

    message = str(exc_info.value)
    assert "embed_body_fields" in message
    assert "async_exit_stack" not in message.split("expected parameter(s):")[1].split(".")[0]


def test_verify_fastapi_compatibility_raises_when_param_missing() -> None:
    def fake_solve(request: object, dependant: object) -> None:
        """A solve_dependencies stub missing every keyword param we rely on."""

    with pytest.raises(FastAPICompatibilityError) as exc_info:
        _verify_fastapi_compatibility(fake_solve)

    message = str(exc_info.value)
    # Names the missing parameter(s) so the failure is actionable.
    assert "embed_body_fields" in message
    assert "async_exit_stack" in message
    assert "dependency_overrides_provider" in message
    # Names the installed FastAPI version and the supported range.
    assert importlib.metadata.version("fastapi") in message
    assert ">=0.112.4,<1.0.0" in message
