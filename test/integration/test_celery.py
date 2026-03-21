# type: ignore  # noqa: PGH003
"""Integration tests for Celery task + @injectable interop."""

import pytest

pytest.importorskip("celery", reason="celery not installable on this Python")

from typing import Annotated, Any  # noqa: E402

from celery import Celery, Task  # noqa: E402
from fastapi import Depends  # noqa: E402

from src.fastapi_injectable.decorator import injectable  # noqa: E402

from .conftest import DbSession, EmailService, get_db, get_email_service  # noqa: E402

app = Celery("test")
app.conf.task_always_eager = True
app.conf.task_store_eager_result = True


def test_celery_unbound_task() -> None:
    """@app.task + @injectable with a positional int arg and Depends(get_db)."""

    @app.task
    @injectable
    def add_record(record_id: int, db: Annotated[DbSession, Depends(get_db)]) -> dict[str, Any]:
        return {"record_id": record_id, "connected": db.connected}

    result = add_record.apply(args=[42])
    value = result.get()
    assert value["record_id"] == 42
    assert value["connected"] is True


def test_celery_bound_task() -> None:
    """@app.task(bind=True) + @injectable with self, str arg, and Depends — the #214 scenario."""

    @app.task(bind=True)
    @injectable
    def send_email(
        self: Task, recipient: str, email_svc: Annotated[EmailService, Depends(get_email_service)]
    ) -> dict[str, Any]:
        return {
            "is_task": isinstance(self, Task),
            "recipient": recipient,
            "ready": email_svc.ready,
        }

    result = send_email.apply(args=["user@test.com"])
    value = result.get()
    assert value["is_task"] is True
    assert value["recipient"] == "user@test.com"
    assert value["ready"] is True


def test_celery_bound_task_only_depends() -> None:
    """@app.task(bind=True) + @injectable with self and only Depends params."""

    @app.task(bind=True)
    @injectable
    def health_check(self: Task, db: Annotated[DbSession, Depends(get_db)]) -> dict[str, Any]:
        return {"is_task": isinstance(self, Task), "connected": db.connected}

    result = health_check.apply()
    value = result.get()
    assert value["is_task"] is True
    assert value["connected"] is True


def test_celery_task_kwarg_override() -> None:
    """Verify that caller-supplied kwargs override DI resolution."""

    @app.task
    @injectable
    def process(record_id: int, db: Annotated[DbSession, Depends(get_db)]) -> dict[str, Any]:
        return {"record_id": record_id, "connected": db.connected, "custom": getattr(db, "custom", False)}

    custom_db = DbSession()
    custom_db.custom = True  # type: ignore[attr-defined]
    result = process.apply(args=[7], kwargs={"db": custom_db})
    value = result.get()
    assert value["record_id"] == 7
    assert value["custom"] is True
