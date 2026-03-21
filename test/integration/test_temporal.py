# type: ignore  # noqa: PGH003
"""Integration tests for Temporal activity + @injectable interop."""

from typing import Annotated, Any

import pytest
from fastapi import Depends
from temporalio import activity
from temporalio.testing import ActivityEnvironment

from src.fastapi_injectable.decorator import injectable

from .conftest import DbSession, get_db


@pytest.fixture
def activity_env() -> ActivityEnvironment:
    return ActivityEnvironment()


async def test_temporal_activity_mixed_params(activity_env: ActivityEnvironment) -> None:
    """@activity.defn + @injectable with float, str, and Depends(get_db)."""

    @activity.defn
    @injectable
    async def process_payment(
        amount: float, currency: str, db: Annotated[DbSession, Depends(get_db)]
    ) -> dict[str, Any]:
        return {"amount": amount, "currency": currency, "connected": db.connected}

    result = await activity_env.run(process_payment, 99.99, "USD")
    assert result["amount"] == 99.99
    assert result["currency"] == "USD"
    assert result["connected"] is True


async def test_temporal_activity_only_depends(activity_env: ActivityEnvironment) -> None:
    """@activity.defn + @injectable with only Depends params."""

    @activity.defn
    @injectable
    async def health_check(db: Annotated[DbSession, Depends(get_db)]) -> dict[str, Any]:
        return {"connected": db.connected}

    result = await activity_env.run(health_check)
    assert result["connected"] is True


async def test_temporal_activity_async(activity_env: ActivityEnvironment) -> None:
    """Async activity variant."""

    @activity.defn
    @injectable
    async def async_process(amount: float, db: Annotated[DbSession, Depends(get_db)]) -> dict[str, Any]:
        return {"amount": amount, "connected": db.connected}

    result = await activity_env.run(async_process, 50.0)
    assert result["amount"] == 50.0
    assert result["connected"] is True


async def test_temporal_activity_kwarg_override(activity_env: ActivityEnvironment) -> None:
    """Verify that caller-supplied kwargs override DI resolution."""

    @activity.defn
    @injectable
    async def check(db: Annotated[DbSession, Depends(get_db)]) -> dict[str, Any]:
        return {"connected": db.connected, "custom": getattr(db, "custom", False)}

    custom_db = DbSession()
    custom_db.custom = True  # type: ignore[attr-defined]
    result = await activity_env.run(check, db=custom_db)
    assert result["connected"] is True
    assert result["custom"] is True
