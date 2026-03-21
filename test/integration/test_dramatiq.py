# type: ignore  # noqa: PGH003
"""Integration tests for Dramatiq actor + @injectable interop."""

from collections.abc import Generator
from typing import Annotated, Any

import dramatiq
import pytest
from dramatiq.brokers.stub import StubBroker
from fastapi import Depends

from src.fastapi_injectable.decorator import injectable

from .conftest import DbSession, get_db


@pytest.fixture
def stub_broker() -> Generator[StubBroker, None, None]:
    broker = StubBroker()
    broker.emit_after("process_boot")
    dramatiq.set_broker(broker)
    yield broker
    broker.flush_all()
    broker.close()


def test_dramatiq_actor_mixed_params(stub_broker: StubBroker) -> None:
    """@dramatiq.actor + @injectable with int, dict, and Depends(get_db) — the #74 scenario."""

    @dramatiq.actor(broker=stub_broker)
    @injectable
    def ingest(batch_id: int, metadata: dict[str, Any], db: Annotated[DbSession, Depends(get_db)]) -> dict[str, Any]:
        return {"batch_id": batch_id, "metadata": metadata, "connected": db.connected}

    result = ingest(42, {"source": "api"})
    assert result["batch_id"] == 42
    assert result["metadata"] == {"source": "api"}
    assert result["connected"] is True


def test_dramatiq_actor_only_depends(stub_broker: StubBroker) -> None:
    """@dramatiq.actor + @injectable with only Depends params."""

    @dramatiq.actor(broker=stub_broker)
    @injectable
    def ping(db: Annotated[DbSession, Depends(get_db)]) -> dict[str, Any]:
        return {"connected": db.connected}

    result = ping()
    assert result["connected"] is True


def test_dramatiq_actor_kwarg_override(stub_broker: StubBroker) -> None:
    """Verify that caller-supplied kwargs override DI resolution."""

    @dramatiq.actor(broker=stub_broker)
    @injectable
    def process(batch_id: int, db: Annotated[DbSession, Depends(get_db)]) -> dict[str, Any]:
        return {"batch_id": batch_id, "connected": db.connected, "custom": getattr(db, "custom", False)}

    custom_db = DbSession()
    custom_db.custom = True  # type: ignore[attr-defined]
    result = process(1, db=custom_db)
    assert result["batch_id"] == 1
    assert result["custom"] is True
