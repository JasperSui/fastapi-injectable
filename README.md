<!-- homepage-begin -->
<p align="center">
  <img src="https://raw.githubusercontent.com/JasperSui/fastapi-injectable/main/static/image/logo.png" alt="FastAPI Injectable" height="200">
</p>
<p align="center">
    <em>Use FastAPI's <code>Depends()</code> anywhere — in CLI tools, background workers, scheduled jobs, and more.</em>
</p>
<p align="center">
    <strong>Stop rewriting your dependency logic. Start reusing it.</strong>
</p>
<p align="center">
<a href="https://pypi.org/project/fastapi-injectable/" target="_blank">
    <img src="https://img.shields.io/pypi/v/fastapi-injectable.svg?color=009688&label=PyPI" alt="PyPI">
</a>
<a href="https://pypi.org/project/fastapi-injectable/" target="_blank">
    <img src="https://img.shields.io/pypi/dm/fastapi-injectable?color=009688&label=Downloads" alt="PyPI Downloads">
</a>
<a href="https://pypi.org/project/fastapi-injectable" target="_blank">
    <img src="https://img.shields.io/pypi/pyversions/fastapi-injectable?color=009688&label=Python" alt="Python Version">
</a>
<a href="https://github.com/JasperSui/fastapi-injectable/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/pypi/l/fastapi-injectable?color=009688&label=License" alt="License">
</a>
<a href="https://github.com/JasperSui/fastapi-injectable/stargazers" target="_blank">
    <img src="https://img.shields.io/github/stars/JasperSui/fastapi-injectable?style=social" alt="GitHub Stars">
</a>
</p>
<p align="center">
<a href="https://github.com/JasperSui/fastapi-injectable/actions?workflow=Tests" target="_blank">
    <img src="https://img.shields.io/github/actions/workflow/status/JasperSui/fastapi-injectable/tests.yml?branch=main&color=009688&label=CI" alt="CI">
</a>
<a href="https://app.codecov.io/gh/JasperSui/fastapi-injectable" target="_blank">
    <img src="https://img.shields.io/codecov/c/github/JasperSui/fastapi-injectable?color=009688&label=Test%20Coverage" alt="Codecov">
</a>
<a href="https://fastapi-injectable.readthedocs.io/" target="_blank">
    <img src="https://img.shields.io/readthedocs/fastapi-injectable/latest.svg?label=Docs&color=009688" alt="Read the Docs">
</a>
</p>

---

**Installation**: `pip install fastapi-injectable`

> **Using `mypy`?** Enable the bundled plugin so `@injectable` calls type-check correctly — add `plugins = ["fastapi_injectable.mypy"]` to your `mypy` config (see [Type Hinting](#type-hinting)). The plugin is `mypy`-only; `pyright`/Pylance users still see `call-arg` errors.

**Documentation**: <a href="https://fastapi-injectable.readthedocs.io/en/latest/" target="_blank">https://fastapi-injectable.readthedocs.io/en/latest/</a>

---

## Why fastapi-injectable?

If you use FastAPI, you've built your app around `Depends()`. But the moment you need those same dependencies in a **CLI command**, **background worker**, **Celery task**, or **scheduled job** — you're stuck. You end up:

- **Duplicating** dependency logic outside of routes
- **Introducing** a second DI framework alongside FastAPI's
- **Refactoring** hundreds of existing dependency functions

**fastapi-injectable** fixes this with a single decorator. Your existing `Depends()` functions just work — everywhere.

> Born from a real need: This project solves [FastAPI#1105](https://github.com/fastapi/fastapi/issues/1105) — a 4+ year old issue requesting `Depends()` outside routes.

## Quick Start

```python
from typing import Annotated

from fastapi import Depends
from fastapi_injectable import injectable

class Database:
    def query(self) -> str:
        return "data"

def get_db() -> Database:
    return Database()

@injectable
def process_data(db: Annotated[Database, Depends(get_db)]) -> str:
    return db.query()

# Use it anywhere!
result = process_data()
print(result) # Output: 'data'
```

> **⚠️ Calling this in a loop (worker, consumer, cron)?** Resolved dependencies are cached in a **process-global** cache that persists across calls — a naïve loop reuses the *same* instances for the whole process and never runs generator cleanup, leaking resources. Reset state per iteration with [`injectable_scope()`](#3-isolated-dependency-scopes-for-parallel-work), or with `clear_dependency_cache()` + `cleanup_all_exit_stacks()`. See [Long-running workers: reset state per iteration](#long-running-workers-reset-state-per-iteration).

## Key Features

| Feature | Description |
|---------|-------------|
| **Drop-in decorator** | Add `@injectable` to any function using `Depends()` |
| **Full async support** | Works with sync, async, and mixed dependency chains |
| **Test-friendly** | Manual overrides let you swap in mocks instantly |
| **Resource cleanup** | Built-in lifecycle management for generator deps |
| **Dependency caching** | Optional caching for better performance |
| **App state access** | Register your FastAPI app to access `app.state` in deps |
| **Mypy plugin** | Opt-in `mypy` plugin for full type-checking — enable `plugins = ["fastapi_injectable.mypy"]` (mypy only) |
| **Graceful shutdown** | Automatic cleanup on program exit via signal handling |

## Overview

`fastapi-injectable` is a lightweight package that enables seamless use of FastAPI's dependency injection system outside of route handlers. It solves a common pain point where developers need to reuse FastAPI dependencies in non-FastAPI contexts like CLI tools, background tasks, or scheduled jobs, allowing you to use FastAPI's dependency injection system **anywhere**!

### Requirements

- Python `3.10` or higher (including `3.13t`, `3.14t` free-threaded builds)
- FastAPI `0.112.4` or higher

<!-- homepage-end -->

## Usage
<!-- usage-begin -->

`fastapi-injectable` provides several powerful ways to use FastAPI's dependency injection outside of route handlers. Let's explore the key usage patterns with practical examples.

### Basic Injection

The most basic way to use dependency injection is through the `@injectable` decorator. This allows you to use FastAPI's `Depends` in any function, not just route handlers.

```python
from typing import Annotated

from fastapi import Depends
from fastapi_injectable import injectable

class Database:
    def __init__(self) -> None:
        pass

    def query(self) -> str:
        return "data"

# Define your dependencies
def get_database():
    return Database()

# Use dependencies in any function
@injectable
def process_data(db: Annotated[Database, Depends(get_database)]):
    return db.query()

# Call it like a normal function
result = process_data()
print(result) # Output: 'data'
```

### Function-based Approach

The function-based approach provides an alternative way to use dependency injection without decorators. This can be useful when you need more flexibility or want to avoid modifying the original function.

Here's how to use it:


```python
from typing import Annotated

from fastapi import Depends
from fastapi_injectable import get_injected_obj

class Database:
    def __init__(self) -> None:
        pass

    def query(self) -> str:
        return "data"

def get_database() -> Database:
    return Database()

def process_data(db: Annotated[Database, Depends(get_database)]):
    return db.query()

# Get injected instance without decorator
result = get_injected_obj(process_data)
print(result) # Output: 'data'
```

### Async Function-based Approach (For Running Event Loops)

When you're working in an async context where an event loop is already running (like Kafka consumers, streaming frameworks, or async callbacks), you need to use `async_get_injected_obj()` instead of `get_injected_obj()`.

**Why?** The regular `get_injected_obj()` uses `loop.run_until_complete()` internally, which fails with `RuntimeError: This event loop is already running` when called from within an already-running event loop. The async version directly awaits coroutines instead, making it safe to use in these scenarios.

**When to use `async_get_injected_obj()`:**
- Inside async callbacks (e.g., Kafka/kstreams consumers)
- In async background tasks that are already running in an event loop
- Within async functions where an event loop is active
- Any scenario where you get "This event loop is already running" errors

**When to use `get_injected_obj()`:**
- In synchronous code
- In scripts or CLI tools without a running loop
- In situations where you need to block and wait for async dependencies

Here's how to use it:

```python
from fastapi_injectable import async_get_injected_obj

class MessageProcessor:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def process(self, message: str) -> str:
        return f"Processed: {message}"

async def get_processor(db: Annotated[Database, Depends(get_database)]) -> MessageProcessor:
    return MessageProcessor(db)

# In a Kafka consumer or async callback
async def consume(message: str):
    # This works in a running event loop!
    processor = await async_get_injected_obj(get_processor)
    result = await processor.process(message)
    print(result)

# In an async streaming framework
from kstreams import ConsumerRecord, Stream

stream = Stream("my-topic")

@stream.consume
async def process_stream(record: ConsumerRecord):
    # Event loop is already running here
    processor = await async_get_injected_obj(get_processor)
    await processor.process(record.value)
```

**Key differences:**

| Feature    | `get_injected_obj()`                         | `async_get_injected_obj()` |
| ---------- | -------------------------------------------- | -------------------------- |
| Usage      | Synchronous code                             | Async contexts             |
| Returns    | Direct value (blocks if async)               | Must be awaited            |
| Event loop | Creates/uses loop via `run_until_complete()` | Works with running loop    |
| Use case   | Scripts, CLI, sync functions                 | Async callbacks, consumers |

### Manual Overrides

Sometimes you want to use FastAPI’s dependency injection system, but still explicitly pass certain arguments yourself.

For example, in tests you may want to supply a mock instead of the default dependency, or in CLI tools you may want to provide a value directly.

`fastapi-injectable` makes this possible by allowing manual overrides: any arguments you pass will take priority over injected dependencies.

```python
from typing import Annotated
from fastapi import Depends
from fastapi_injectable import get_injected_obj, injectable

class Database:
    def query(self) -> str:
        return "real data"

def get_db() -> Database:
    return Database()

@injectable
def process_data(db: Annotated[Database, Depends(get_db)]) -> str:
    return db.query()

# Normal usage – resolved through DI
print(process_data())
# Output: "real data"

# Override dependency manually (great for tests)
mock_db = Database()
mock_db.query = lambda: "mock data"

print(process_data(db=mock_db)) # Explicitly pass the mock dependency
# Output: "mock data"
```

### Testing with `dependency_overrides`

The [Manual Overrides](#manual-overrides) above only replace **top-level** parameters of the entry-point function — they can't reach a **nested** dependency resolved deeper in the chain. To swap one out (the usual case in tests), use FastAPI's native `app.dependency_overrides`, which `fastapi-injectable` honors once you `register_app()`:

```python
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi_injectable import (
    clear_dependency_cache,
    injectable,
    register_app,
    run_coroutine_sync,
)


class Database:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn


class Repo:
    def __init__(self, db: Database) -> None:
        self.db = db


def get_db() -> Database:
    return Database("real")


def get_repo(db: Annotated[Database, Depends(get_db)]) -> Repo:
    return Repo(db)


@injectable
def handle_request(repo: Annotated[Repo, Depends(get_repo)]) -> str:
    return repo.db.dsn


# Register the app so fastapi-injectable consults app.dependency_overrides.
app = FastAPI()
run_coroutine_sync(register_app(app))

# Override the *nested* get_db — a kwarg override on handle_request() can't reach it,
# but dependency_overrides resolves through the whole chain (Repo -> Database).
app.dependency_overrides[get_db] = lambda: Database("mock")
try:
    assert handle_request() == "mock"
finally:
    # Reset in teardown so the next test sees the real dependencies again.
    app.dependency_overrides.clear()
    run_coroutine_sync(clear_dependency_cache())

assert handle_request() == "real"
```

This is the same mechanism you already use for FastAPI route tests, so existing test helpers carry over. It works for sync, async, and generator dependencies alike — see [`test/test_integration.py`](https://github.com/JasperSui/fastapi-injectable/tree/main/test/test_integration.py) for the full matrix.

### Generator Dependencies with Cleanup

When working with generator dependencies that require cleanup (like database connections or file handles), `fastapi-injectable` provides built-in support for controlling dependency lifecycles and proper resource management with error handling.

Here's an example showing how to work with generator dependencies:

```python
from collections.abc import Generator
from typing import Annotated

from fastapi import Depends
from fastapi_injectable import (
    cleanup_all_exit_stacks,
    cleanup_exit_stack_of_func,
    clear_dependency_cache,
    injectable,
    run_coroutine_sync,
)
from fastapi_injectable.exception import DependencyCleanupError

class Database:
    def __init__(self) -> None:
        self.closed = False

    def query(self) -> str:
        return "data"

    def close(self) -> None:
        self.closed = True

class Machine:
    def __init__(self, db: Database) -> None:
        self.db = db

def get_database() -> Generator[Database, None, None]:
    db = Database()
    yield db
    db.close()

@injectable
def get_machine(db: Annotated[Database, Depends(get_database)]):
    machine = Machine(db)
    return machine

# Use the function
machine = get_machine()

# The cleanup helpers are async. In sync code (CLI tools, workers, cron) wrap them
# with run_coroutine_sync(); in async code you can `await` them directly instead.

# Option #1: Silent cleanup when done for a single decorated function (logs errors but doesn't raise)
assert machine.db.closed is False
run_coroutine_sync(cleanup_exit_stack_of_func(get_machine))
assert machine.db.closed is True

# Option #2: Strict cleanup with error handling
try:
    run_coroutine_sync(cleanup_exit_stack_of_func(get_machine, raise_exception=True))
except DependencyCleanupError as e:
    print(f"Cleanup failed: {e}")

# Option #3: If you don't care about the other injectable functions,
#            just use cleanup_all_exit_stacks() to clean up everything at once.
#            (Reset the process-global cache first so there's a fresh machine to clean.)
run_coroutine_sync(clear_dependency_cache())
machine = get_machine()
assert machine.db.closed is False
run_coroutine_sync(cleanup_all_exit_stacks()) # accepts raise_exception=True too
assert machine.db.closed is True
```

### Async Support

`fastapi-injectable` provides full support for both synchronous and asynchronous dependencies, allowing you to mix and match them as needed. You can freely use async dependencies in sync functions and vice versa. For cases where you need to run async code in a synchronous context, we provide the `run_coroutine_sync` utility function.

```python
import asyncio
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends
from fastapi_injectable import injectable, run_coroutine_sync

class AsyncDatabase:
    def __init__(self) -> None:
        self.closed = False

    async def query(self) -> str:
        return "data"

    async def close(self) -> None:
        self.closed = True

async def get_async_database() -> AsyncGenerator[AsyncDatabase, None]:
    db = AsyncDatabase()
    yield db
    await db.close()

@injectable
async def async_process_data(db: Annotated[AsyncDatabase, Depends(get_async_database)]):
    return await db.query()

# In sync code (no running loop), block on it with run_coroutine_sync():
result = run_coroutine_sync(async_process_data())
print(result) # Output: 'data'

# In async code, await the injectable directly:
async def main() -> None:
    result = await async_process_data()
    print(result) # Output: 'data'

asyncio.run(main())
```

### Dependency Caching Control

By default, `fastapi-injectable` caches dependency instances to improve performance and maintain consistency. This means when you request a dependency multiple times, you'll get the same instance back.

You can control this behavior using the `use_cache` parameter in the `@injectable` decorator:
- `use_cache=True` (default): Dependencies are cached and reused
- `use_cache=False`: New instances are created for each dependency request

Using `use_cache=False` is particularly useful when:
- You need fresh instances for each request
- You want to avoid sharing state between different parts of your application
- You're dealing with stateful dependencies that shouldn't be reused

```python
from typing import Annotated

from fastapi import Depends

from fastapi_injectable import injectable

class Mayor:
    pass

class Capital:
    def __init__(self, mayor: Mayor) -> None:
        self.mayor = mayor

class Country:
    def __init__(self, capital: Capital) -> None:
        self.capital = capital

def get_mayor() -> Mayor:
    return Mayor()

def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Capital:
    return Capital(mayor)

@injectable
def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
    return Country(capital)

# With caching (default), all instances share the same dependencies
country_1 = get_country()
country_2 = get_country()
country_3 = get_country()
assert country_1.capital is country_2.capital is country_3.capital
assert country_1.capital.mayor is country_2.capital.mayor is country_3.capital.mayor

# Without caching, new instances are created each time
@injectable(use_cache=False)
def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
    return Country(capital)

country_1 = get_country()
country_2 = get_country()
country_3 = get_country()
assert country_1.capital is not country_2.capital is not country_3.capital
assert country_1.capital.mayor is not country_2.capital.mayor is not country_3.capital.mayor
```

### Long-running workers: reset state per iteration

By default, resolved dependencies live in a **process-global** cache, and the cleanup for generator dependencies is held in a **process-global** exit stack — neither is reset automatically. In a one-shot script that's fine, but in a **worker / consumer / cron loop** it means every iteration silently reuses the *same* instances, and the generators' cleanup (closing connections, files, sessions) never runs until the process exits. The fix is to resolve fresh dependencies per message and reset after each one:

```python
from typing import Annotated

from fastapi import Depends
from fastapi_injectable import (
    cleanup_exit_stack_of_func,
    clear_dependency_cache,
    injectable,
    run_coroutine_sync,
)


class Connection:
    def __init__(self) -> None:
        self.open = True

    def close(self) -> None:
        self.open = False


def get_connection():  # a generator dependency that must be cleaned up
    conn = Connection()
    print("  opened connection")
    try:
        yield conn
    finally:
        conn.close()
        print("  closed connection")


@injectable
def handle(conn: Annotated[Connection, Depends(get_connection)]) -> str:
    return f"open={conn.open}"


def process(messages: list[str]) -> None:
    for message in messages:
        result = handle()  # fresh resolution for this message
        print(message, "->", result)

        # Reset *after each message*: run the generator cleanup for this call tree,
        # then drop the cache — so the next iteration starts from a clean slate.
        # The cleanup helpers are async, so wrap them with run_coroutine_sync() in sync code.
        run_coroutine_sync(cleanup_exit_stack_of_func(handle))
        run_coroutine_sync(clear_dependency_cache())


process(["msg-1", "msg-2"])
```

Each iteration prints `opened connection` / `closed connection`, proving resources are released per message instead of leaking for the lifetime of the process.

**Prefer a context manager?** In async workers, [`injectable_scope()`](#3-isolated-dependency-scopes-for-parallel-work) wraps this whole pattern — each `async with injectable_scope():` block gets its own exit stack and cache and cleans up on exit, so you can't forget the reset (and it's safe under `asyncio.TaskGroup` fan-out). To also clean up when the **process** exits (SIGTERM/SIGINT), call [`setup_graceful_shutdown()`](#graceful-shutdown). For full, copy-pasteable worker recipes — Celery, Dramatiq, Temporal, and a Typer CLI — see [Real-world Examples](https://fastapi-injectable.readthedocs.io/en/latest/real-world-examples.html).

### Type Hinting

`fastapi-injectable` will prepare the dependency objects of injected functions for you, but static type checkers like `mypy` haven't known about the dependency object existence since they are normally injected via `Annotated[Type, Depends(get_dependency_func)]`, when using this kind of expression, static type checkers will complain if you don't explicitly provide the dependency object when using the function, example error codes ([call-arg](https://mypy.readthedocs.io/en/stable/error_code_list.html#check-arguments-in-calls-call-arg)).

```python

@injectable
def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
    return Country(capital)

country = get_country() # mypy will complain here, error: Missing positional arguments or Too few arguments.
```

To make the `mypy` happy, you can enable the `fastapi-injectable.mypy` plugin in your `mypy.ini` file, or add `fastapi_injectable.mypy` to your `pyproject.toml` file.

```toml
[tool.mypy]
# ... your mypy config
plugins = ["fastapi_injectable.mypy"]
```

```python
@injectable
def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
    return Country(capital)

country = get_country() # Now it's happy!
```

> **Note:** The plugin is a `mypy` plugin, so only `mypy` benefits from it. `pyright`/Pylance (the default type checker in VS Code) cannot load it and will still report `call-arg` errors on `@injectable` calls.

### Event Loop Management

`fastapi-injectable` includes a powerful loop management system to handle asynchronous code execution in different contexts. This is particularly useful when working with async code in synchronous environments or when you need controlled event loop execution.

```python
from fastapi_injectable import loop_manager, run_coroutine_sync

# Configure loop strategy
# Options: "current" (default), "isolated", or "background_thread"
loop_manager.set_loop_strategy("isolated")

loop = loop_manager.get_loop() # This is useful if you have to aware of the loop, so that you can make sure the objects created by fastapi-injectable are executed in the right loop.
# asyncio.set_event_loop(loop)
# loop.run_until_complete(your_coro)

# The run_coroutine_sync function uses loop_manager internally
# This works regardless of what thread or context you're in
result = run_coroutine_sync(async_process_data())
```


Loop strategies explained:

1. **`current`** (default): Uses the current thread's event loop. This is the simplest option and meets most needs.
   - Limitation: Fails if no loop is running in the current thread.
   - Perfect when your code runs in synchronous functions within the main thread with a runnable event loop.

    ```python
    # Default strategy - uses the current thread's event loop
    # Simple and efficient for most use cases

    import asyncio

    my_loop = asyncio.get_event_loop()
    # Or
    # my_loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(my_loop)

    loop_manager.set_loop_strategy("current")

    assert my_loop is loop_manager.get_loop()

    # This will work if you're in the main thread with a running event loop
    result = run_coroutine_sync(async_process_data())
    ```

2. **`isolated`**: Creates a separate isolated loop.
   - Benefit: Works even when no loop is running in the current thread.
   - Ideal when you need control over the loop lifecycle or need to ensure all injected objects come from the same event loop (important for objects like `aiohttp.ClientSession` that must execute in the same loop where they were instantiated).

    ```python
    # Isolated strategy - creates a dedicated event loop
    # Great for scripts, CLI tools, or when you need loop lifecycle control

    import asyncio

    from fastapi_injectable import get_injected_obj

    async def get_aiohttp_session():
        return aiohttp.ClientSession()

    # Make sure the loop strategy is set to "isolated" before any injected objects are created
    loop_manager.set_loop_strategy("isolated")

    aiohttp_session = get_injected_obj(get_aiohttp_session)

    original_loop = asyncio.get_event_loop()
    loop = loop_manager.get_loop()

    assert original_loop is not loop

    original_loop.run_until_complete(aiohttp_session.get("https://www.google.com")) # This will raise an error because the aiohttp_session is created in the loop_manager's loop, not the original_loop.

    loop.run_until_complete(aiohttp_session.get("https://www.google.com")) # This will work since the aiohttp_session is created in the loop_manager's loop and executed in the same loop.
    ```

3. **`background_thread`**: Runs a dedicated background thread with its own event loop.
   - Best for: Long-running applications where you need to run async code from sync contexts.
   - Benefit: Allows async code to run from any thread without blocking.
   - Perfect when you're uncertain about your environment's event loop availability and don't use objects that assume they run in the same event loop.

    ```python
    # Background thread strategy - runs a daemon thread with a dedicated loop
    # Ideal for long-running applications or uncertain environments
    loop_manager.set_loop_strategy("background_thread")

    # This will work from any thread, even without a running event loop
    # The background thread handles all async operations
    result = run_coroutine_sync(async_process_data())
    ```

### Logging Configuration

`fastapi-injectable` provides a simple way to configure logging for the package. This is useful for debugging or monitoring the package's behavior.

```python
import logging
from fastapi_injectable import configure_logging

# Basic configuration with default format
configure_logging(level=logging.DEBUG)

# Custom format
configure_logging(
    level=logging.INFO,
    format_="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Custom handler
file_handler = logging.FileHandler("fastapi_injectable.log")
configure_logging(level=logging.WARNING, handler=file_handler)
```

### Graceful Shutdown

If you want to ensure proper cleanup when the program exits, you can register cleanup functions with error handling:

```python
import signal

from fastapi_injectable import setup_graceful_shutdown
from fastapi_injectable.exception import DependencyCleanupError

# Option #1: Silent cleanup (default)
# it handles SIGTERM and SIGINT, and will logs errors if any exceptions are raised during cleanup
setup_graceful_shutdown()

# Option #2: Strict cleanup that raises errors
# it handles SIGTERM and SIGINT, and will raise DependencyCleanupError if any exceptions are raised during cleanup
setup_graceful_shutdown(raise_exception=True)

# Option #3: Pass custom signals to handle
# it handles the custom signals, and will raise DependencyCleanupError if any exceptions are raised during cleanup
setup_graceful_shutdown(
    signals=[signal.SIGTERM],
    raise_exception=True
)

# Option #4: Only register the atexit cleanup, without installing OS signal handlers
# (useful when a framework already owns SIGTERM/SIGINT, or when calling off the main thread)
setup_graceful_shutdown(install_signal_handlers=False)
```

**How the signal handler behaves:**

- **It still terminates the process.** After running cleanup, the handler re-raises termination so
  your process exits as usual: `SIGINT` raises `KeyboardInterrupt` and any other signal (e.g. `SIGTERM`)
  raises `SystemExit(0)`. This means an orchestrator's `SIGTERM` (Kubernetes, systemd) and `Ctrl-C` keep
  working — cleanup runs *and then* the process exits, instead of hanging until `SIGKILL`.
- **It chains your existing handler.** If you already installed a handler for one of these signals,
  it is captured and invoked after cleanup, so your own shutdown logic (flush metrics, finish in-flight
  work, close connections) is preserved rather than silently overwritten. `SIG_DFL`/`SIG_IGN` are left
  alone.
- **It must run on the main thread.** Python only allows `signal.signal()` to be called from the main
  thread of the main interpreter. If you call `setup_graceful_shutdown()` from another thread, signal
  registration is skipped with a warning and only the `atexit` cleanup is registered — it will **not**
  raise `ValueError`. Pass `install_signal_handlers=False` to opt out of signal handling explicitly.


### App Registration for State Access

If your dependencies need access to the FastAPI app state (like database connections or other services), you can register your app with `fastapi-injectable`:

```python
import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi_injectable import injectable, register_app
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# Define your dependencies that need app state access
def get_db_engine(*, request: Request) -> AsyncEngine:
    return request.app.state.db_engine

DBEngine = Annotated[AsyncEngine, Depends(get_db_engine)]

async def get_db(*, db_engine: DBEngine) -> AsyncIterator[AsyncSession]:
    session = async_sessionmaker(db_engine)
    async with session.begin() as session:
        yield session

DB = Annotated[AsyncSession, Depends(get_db)]

# Register your app during startup
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Register the app so dependencies can access app.state
    await register_app(app)

    # Setup your app state
    app.state.db_engine = create_async_engine("postgresql+asyncpg://...")
    yield
    await app.state.db_engine.dispose()

app = FastAPI(lifespan=lifespan)

# Now you can use dependencies that need app state anywhere!
@injectable
async def process_data(db: DB) -> str:
    result = await db.execute(...)
    return result

# Use it in background tasks, CLI tools, etc. (once `register_app` has run via the app lifecycle)
async def main() -> None:
    result = await process_data()


asyncio.run(main())
```

This is particularly useful when:
- Your dependencies need access to shared services in `app.state`
- You're using third-party libraries that call your code internally
- You want to maintain a single source of truth for long-running services

### Test Isolation

`fastapi-injectable` keeps its dependency cache and exit stacks in **process-global** singletons, so a value cached — or a generator left open — by one test can leak into the next. To make "tests are isolated" the default, the package ships a pytest plugin that resets both around every test.

The plugin is registered automatically when you install `fastapi-injectable` (via a `pytest11` entry point) — there's nothing to import or enable. By default it resets the global dependency cache and exit stacks around **every** test, so leakage simply can't happen:

```python
from fastapi_injectable import get_injected_obj

def test_a() -> None:
    service = get_injected_obj(get_service)  # populates the process-global cache
    ...

def test_b() -> None:
    # Starts from a clean slate — nothing from test_a leaks in.
    service = get_injected_obj(get_service)
```

Prefer to manage isolation yourself? Turn the autouse behaviour off in your pytest config and request the `injectable_cleanup` fixture only where you need it:

```toml
# pyproject.toml
[tool.pytest.ini_options]
injectable_autouse_cleanup = false
```

```python
import pytest

# Reset around a single test:
def test_one(injectable_cleanup: None) -> None:
    ...

# ...or for a whole module:
pytestmark = pytest.mark.usefixtures("injectable_cleanup")
```

Both the autouse reset and the `injectable_cleanup` fixture work in sync and async tests alike, and honour whatever loop strategy you've configured via `loop_manager`.

<!-- usage-end -->

## Advanced Scenarios
<!-- advanced-scenarios-begin -->

If the basic examples don't cover your needs, check out our test files - they're basically a cookbook of real-world scenarios:

### 1. [`test_injectable.py`](https://github.com/JasperSui/fastapi-injectable/tree/main/test/test_injectable.py) - Shows all possible combinations of:

- Sync/async functions
- Decorator vs function wrapping
- Caching vs no caching

### 2. [`test_integration.py`](https://github.com/JasperSui/fastapi-injectable/tree/main/test/test_integration.py) - Demonstrates:

- Resource cleanup
- Generator dependencies
- Mixed sync/async dependencies
- Multiple dependency chains

These test cases mirror common development patterns you'll encounter. They show how to handle complex dependency trees, resource management, and mixing sync/async code - stuff you'll actually use in production.

The test files are written to be self-documenting, so browsing through them will give you practical examples for most scenarios you'll face in your codebase.

### 3. Isolated dependency scopes for parallel work

By default, the exit stack that holds a dependency's cleanup is keyed by the **function**, so every concurrent call of the same injectable shares one stack. When you process events in parallel (for example, an async consumer fanning work out through `asyncio.TaskGroup`), that shared stack makes per-event cleanup unsafe — cleaning up one event can tear down another's in-flight resources.

`injectable_scope()` gives each unit of work its **own** exit stack *and* cache — the same request-scoped model FastAPI uses internally. Enter the scope inside each task; `contextvars` keeps the tasks isolated:

```python
import asyncio
from collections.abc import AsyncGenerator

from fastapi_injectable import async_get_injected_obj, injectable_scope


async def get_connection() -> AsyncGenerator[Connection, None]:
    conn = await open_connection()
    try:
        yield conn
    finally:
        await conn.close()


async def process_event(event) -> None:
    async with injectable_scope():
        conn = await async_get_injected_obj(get_connection)
        await handle(event, conn)
    # only this event's connection is closed here — siblings are untouched


async def consume(events) -> None:
    async with asyncio.TaskGroup() as tg:  # Python 3.11+; use asyncio.gather() on 3.10
        for event in events:
            tg.create_task(process_event(event))
```

#### Sharing one scope explicitly

`InjectableScope` is also usable directly when you want to manage a scope's lifecycle yourself and inject into it from several places. Pass it with `scope=`; this routes resolution into that scope without taking ownership of it (you close it):

```python
import asyncio

from fastapi_injectable import InjectableScope, async_get_injected_obj


async def main() -> None:
    scope = InjectableScope()
    async with scope:
        a = await async_get_injected_obj(get_a, scope=scope)
        scope.exit_stack.push_async_callback(my_cleanup)  # rides the same lifecycle
    # a and my_cleanup are torn down together when the block exits


asyncio.run(main())
```

#### Notes & limitations

- Inside a scope, `cleanup_exit_stack_of_func()` / `cleanup_all_exit_stacks()` are no-ops for the scope's resources — the `async with` owns cleanup.
- With no active scope, behavior is unchanged (the global, function-keyed manager and the `cleanup_*` helpers work exactly as before).
- `injectable_scope` is **async-first**. Under the `background_thread` loop strategy, `contextvars` are not propagated across threads, so a scope is not visible inside the background loop. Use the async API for scoped resolution.

<!-- advanced-scenarios-end -->

## Real-world Examples

We've collected some real-world examples of using `fastapi-injectable` in various scenarios:

### 1. [Processing messages by background worker with `Depends()`](https://fastapi-injectable.readthedocs.io/en/latest/real-world-examples.html#1-processing-messages-by-background-worker-with-depends)

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

### 2. [Framework integrations: Celery, Dramatiq, Temporal, Typer/Click](https://fastapi-injectable.readthedocs.io/en/latest/real-world-examples.html#2-framework-integrations)

Copy-pasteable recipes for wiring `@injectable` into the task/worker frameworks the package targets — distilled from the CI integration tests so they stay verified. The rule is the same for all: framework decorator on the outside, `@injectable` on the inside.

Please refer to the [Real-world Examples](https://fastapi-injectable.readthedocs.io/en/latest/real-world-examples.html) for more details.

## Frequently Asked Questions

<!-- faq-begin -->

<details>
<summary><strong>Click to expand FAQ</strong></summary>

- [Why would I need this package?](#why-would-i-need-this-package)
- [Why not directly use other DI packages like Dependency Injector or FastDepends?](#why-not-directly-use-other-di-packages-like-dependency-injector-or-fastdepends)
- [Can I use it with existing FastAPI dependencies?](#can-i-use-it-with-existing-fastapi-dependencies)
- [Does it work with all FastAPI dependency types?](#does-it-work-with-all-fastapi-dependency-types)
- [What happens to dependency cleanup in long-running processes?](#what-happens-to-dependency-cleanup-in-long-running-processes)
- [Can I mix sync and async dependencies?](#can-i-mix-sync-and-async-dependencies)
- [When should I use `async_get_injected_obj()` vs `get_injected_obj()`?](#when-should-i-use-async_get_injected_obj-vs-get_injected_obj)
- [Are type hints fully supported for `injectable()` and `get_injected_obj()`?](#are-type-hints-fully-supported-for-injectable-and-get_injected_obj)
- [How does caching work?](#how-does-caching-work)
- [Is it production-ready?](#is-it-production-ready)


### Why would I need this package?

A: If your project heavily relies on FastAPI's `Depends()` as the sole DI system and you don't want to introduce additional DI packages (like [Dependency Injector](https://python-dependency-injector.ets-labs.org/) or [FastDepends](https://github.com/Lancetnik/FastDepends)), `fastapi-injectable` is your friend.

It allows you to reuse your existing FastAPI built-in DI system anywhere, without the need to **refactor your entire codebase** or **maintain multiple DI systems**.

Life is short, keep it simple!

<hr>

### Why not directly use other DI packages like Dependency Injector or FastDepends?

A: You absolutely can if your situation allows you to:
1. Modify large amounts of existing code that uses `Depends()`
2. Maintain multiple DI systems in your project

`fastapi-injectable` focuses solely on extending FastAPI's built-in `Depends()` beyond routes. We're not trying to be another DI system - **we're making the existing one more useful!**

For projects with hundreds of dependency functions (especially with nested dependencies), this approach is more intuitive and requires minimal changes to your existing code.

Choose what works best for you!

<hr>

### Can I use it with existing FastAPI dependencies?

A: Absolutely! That's exactly what this package was built for! `fastapi-injectable` was created to seamlessly work with FastAPI's dependency injection system, allowing you to reuse your existing `Depends()` code **anywhere** - not just in routes.

Focus on what matters instead of worrying about how to get your existing dependencies outside of FastAPI routes!

<hr>

### Does it work with all FastAPI dependency types?

A: Yes! It supports:
- Regular dependencies
- Generator dependencies (with cleanup utility functions)
- Async dependencies
- Sync dependencies
- Nested dependencies (dependencies with sub-dependencies)

<hr>

### What happens to dependency cleanup in long-running processes?

A: You have a few options:
1. Manual cleanup per function: `await cleanup_exit_stack_of_func(your_func)`
2. Cleanup everything: `await cleanup_all_exit_stacks()`
3. Reset the (process-global) dependency cache: `await clear_dependency_cache()`
4. Automatic cleanup on shutdown: `setup_graceful_shutdown()`

In a worker/consumer loop, combine 1 (or 2) with 3 after each message — or wrap each unit of work in `injectable_scope()`, which does both for you. See [Long-running workers: reset state per iteration](#long-running-workers-reset-state-per-iteration). In sync code, wrap the async helpers with `run_coroutine_sync(...)`.

<hr>

### Can I mix sync and async dependencies?

A: Yes! You can freely mix them. For running async code in sync contexts, use the provided `run_coroutine_sync()` utility.

<hr>

### When should I use `async_get_injected_obj()` vs `get_injected_obj()`?

A: Use `async_get_injected_obj()` when you're in an async context with a **running event loop** (like Kafka consumers, async callbacks, or streaming frameworks). Use `get_injected_obj()` in **synchronous code** or when no event loop is running.

If you see `RuntimeError: This event loop is already running`, switch to `async_get_injected_obj()`.

**Quick rule of thumb:**
- Already in an `async` function with a running loop? → Use `async_get_injected_obj()`
- In sync code or scripts? → Use `get_injected_obj()`

See [Async Function-based Approach](#async-function-based-approach-for-running-event-loops) for detailed examples.

<hr>

### Are type hints fully supported for `injectable()` and `get_injected_obj()`?

A: Currently, type hint support is available if you are using `mypy` as your static type checker, you can enable the `fastapi-injectable.mypy` plugin in your `mypy.ini` file, or add `fastapi_injectable.mypy` to your `pyproject.toml` file, see [Type Hinting](#type-hinting) for more details.

<hr>

### How does caching work?

A: By default, resolved dependencies are cached in a **process-global** cache that lives for the lifetime of the process. This differs from FastAPI routes, where the cache is request-scoped and discarded after each request — here nothing is auto-cleared, so a long-running worker reuses the same instances across iterations until you reset it. To control it: disable per-function caching with `@injectable(use_cache=False)`, scope it per unit of work with `injectable_scope()`, or reset it manually with `await clear_dependency_cache()`. See [Long-running workers: reset state per iteration](#long-running-workers-reset-state-per-iteration).

<hr>

### Is it production-ready?

A: Yes! The package has:
- **100%** test coverage
- Type checking with `mypy`
- Comprehensive error handling
- Production use cases documented

<hr>

</details>

<!-- faq-end -->

## Star History

<a href="https://star-history.com/#JasperSui/fastapi-injectable&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=JasperSui/fastapi-injectable&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=JasperSui/fastapi-injectable&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=JasperSui/fastapi-injectable&type=Date" />
 </picture>
</a>

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license],
`fastapi-injectable` is free and open source software.

<!-- info-begin -->
## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

1. This project was generated from [@cjolowicz]'s [Hypermodern Python Cookiecutter] template.
2. Thanks to [@barapa]'s initiation, [his work] inspires me to create this project.

[@cjolowicz]: https://github.com/cjolowicz
[@barapa]: https://github.com/barapa
[pypi]: https://pypi.org/
[hypermodern python cookiecutter]: https://github.com/cjolowicz/cookiecutter-hypermodern-python
[file an issue]: https://github.com/JasperSui/fastapi-injectable/issues
[pip]: https://pip.pypa.io/
[his work]: https://github.com/fastapi/fastapi/discussions/7720#discussioncomment-8661497

## Related Issue & Discussion

- [[Issue] Using Depends() in other functions, outside the endpoint path operation!](https://github.com/fastapi/fastapi/issues/1105)
- [[Discussion] Using Depends() in other functions, outside the endpoint path operation!](https://github.com/fastapi/fastapi/discussions/7720)

## Bonus

My blog posts about the prototype of this project:

1. [Easily Reusing Depends Outside FastAPI Routes](https://j-sui.com/2024/10/26/use-fastapi-depends-outside-fastapi-routes-en/)
2. [在 FastAPI Routes 以外無痛複用 Depends 的方法](https://j-sui.com/2024/10/26/use-fastapi-depends-outside-fastapi-routes/)

<!-- info-end -->

<!-- github-only -->

[license]: https://github.com/JasperSui/fastapi-injectable/blob/main/LICENSE
[contributor guide]: https://github.com/JasperSui/fastapi-injectable/blob/main/CONTRIBUTING.md
