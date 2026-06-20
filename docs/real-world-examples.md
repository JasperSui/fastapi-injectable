# Real-world Examples

## 1. Processing messages by background worker with `Depends()`

Here's a practical example of using `fastapi-injectable` in a background worker that processes messages.

This example demonstrates several key patterns for using dependency injection in background workers:

1. **Fresh Dependencies per Message**:
   - Each message gets a fresh set of dependencies through `_init_as_consumer()`
   - This ensures clean state for each message, similar to how FastAPI handles HTTP requests

2. **Proper Resource Management**:
   - Dependencies with cleanup needs (like database connections) are properly handled
   - Cleanup code in generators runs when `cleanup_exit_stack_of_func()` is called, or automatically at the end of an `injectable_scope()` block
   - Cache is cleared between messages to prevent memory leaks

3. **Graceful Shutdown**:
   - `setup_graceful_shutdown()` ensures resources are cleaned up on program termination
   - Handles both SIGTERM and SIGINT signals

```{literalinclude} ../example/worker/main.py
---
language: python
---
```

You can extend the example to re-using the business logic in your:
- Message queue consumers
- Batch processing jobs
- Long-running background tasks
- Any scenario where you need FastAPI-style dependency injection in a worker process

## 2. Framework integrations

`@injectable` composes with most task/worker frameworks. The rule is the same everywhere: stack the framework's decorator on the **outside** and `@injectable` on the **inside** (closest to the function). The framework then sees a plain callable, while `@injectable` resolves every `Depends(...)` parameter — the caller supplies the non-dependency arguments, and caller-supplied keyword arguments still override injection.

These recipes are distilled from the integration tests in [`test/integration/`](https://github.com/JasperSui/fastapi-injectable/tree/main/test/integration), which run in CI, so they stay verified.

All recipes reuse the same ordinary FastAPI dependencies — that's the whole point: define them once, use them everywhere.

```python
class DbSession:
    connected = True


class EmailService:
    ready = True


def get_db() -> DbSession:
    return DbSession()


def get_email_service() -> EmailService:
    return EmailService()
```

> **Worker lifecycle:** if your dependencies use generators that need cleanup, reset per-task state after each task — `cleanup_exit_stack_of_func(...)` + `clear_dependency_cache()`, or an `injectable_scope()` block. See *Long-running workers: reset state per iteration* in the [usage guide](https://fastapi-injectable.readthedocs.io/en/latest/usage.html).

### Celery

```python
from typing import Annotated, Any

from celery import Celery, Task
from fastapi import Depends
from fastapi_injectable import injectable

celery_app = Celery("myapp", broker="redis://localhost:6379/0")


@celery_app.task   # framework decorator: outermost
@injectable        # injectable: innermost (closest to the function)
def add_record(record_id: int, db: Annotated[DbSession, Depends(get_db)]) -> dict[str, Any]:
    # `record_id` comes from the caller; `db` is injected from get_db.
    return {"record_id": record_id, "connected": db.connected}


@celery_app.task(bind=True)   # bound tasks keep `self` as the first parameter
@injectable
def send_email(
    self: Task, recipient: str, svc: Annotated[EmailService, Depends(get_email_service)]
) -> dict[str, Any]:
    return {"task_id": self.request.id, "recipient": recipient, "ready": svc.ready}


# Enqueue as usual — dependencies resolve inside the worker:
add_record.delay(42)
send_email.delay("user@example.com")
```

### Dramatiq

```python
from typing import Annotated, Any

import dramatiq
from fastapi import Depends
from fastapi_injectable import injectable


@dramatiq.actor   # framework decorator: outermost
@injectable       # injectable: innermost
def ingest(
    batch_id: int, metadata: dict[str, Any], db: Annotated[DbSession, Depends(get_db)]
) -> dict[str, Any]:
    # `batch_id` / `metadata` from the caller; `db` injected.
    return {"batch_id": batch_id, "connected": db.connected}


# Send a message — dependencies resolve inside the worker:
ingest.send(42, {"source": "api"})
```

### Temporal

```python
from typing import Annotated, Any

from fastapi import Depends
from fastapi_injectable import injectable
from temporalio import activity


@activity.defn   # framework decorator: outermost
@injectable      # injectable: innermost
async def process_payment(
    amount: float, currency: str, db: Annotated[DbSession, Depends(get_db)]
) -> dict[str, Any]:
    return {"amount": amount, "currency": currency, "connected": db.connected}


# Register `process_payment` with your Worker; Temporal calls it with (amount, currency)
# and `db` is injected. Both sync and async activities are supported.
```

### Typer / Click (sync CLI)

CLI frameworks inspect the command's signature to build `--options`, so an injected parameter would leak into the CLI. Resolve dependencies **inside** the command body with `get_injected_obj()` instead of decorating with `@injectable`:

```python
import typer
from fastapi_injectable import (
    cleanup_all_exit_stacks,
    clear_dependency_cache,
    get_injected_obj,
    run_coroutine_sync,
)

cli = typer.Typer()


@cli.command()
def migrate(target: str) -> None:
    db: DbSession = get_injected_obj(get_db)  # resolve inside the body
    typer.echo(f"migrating to {target} (connected={db.connected})")

    # Clean up before the process exits (generators, cache):
    run_coroutine_sync(cleanup_all_exit_stacks())
    run_coroutine_sync(clear_dependency_cache())


if __name__ == "__main__":
    cli()
```
