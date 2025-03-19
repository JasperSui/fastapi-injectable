<!-- homepage-begin -->
<p align="center">
  <img src="https://raw.githubusercontent.com/JasperSui/fastapi-injectable/main/static/image/logo.png" alt="FastAPI Injectable" height="200">
</p>
<p align="center">
    <em>Use FastAPI's Depends() anywhere - even outside FastAPI routes</em>
</p>
<p align="center">
<a href="https://pypi.org/project/fastapi-injectable/" target="_blank">
    <img src="https://img.shields.io/pypi/v/fastapi-injectable.svg?color=009688&label=PyPI" alt="PyPI">
</a>
<a href="https://pypi.org/project/fastapi-injectable" target="_blank">
    <img src="https://img.shields.io/pypi/pyversions/fastapi-injectable?color=009688&label=Python" alt="Python Version">
</a>
<a href="https://github.com/JasperSui/fastapi-injectable/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/pypi/l/fastapi-injectable?color=009688&label=License" alt="License">
</a>
<a href="https://fastapi-injectable.readthedocs.io/" target="_blank">
    <img src="https://img.shields.io/readthedocs/fastapi-injectable/latest.svg?label=Read%20the%20Docs&color=009688" alt="Read the documentation">
</a>
</p>
<p align="center">
<a href="https://github.com/JasperSui/fastapi-injectable/actions?workflow=Tests" target="_blank">
    <img src="https://img.shields.io/github/actions/workflow/status/JasperSui/fastapi-injectable/tests.yml?branch=main&color=009688&label=CI" alt="CI">
</a>
<a href="https://app.codecov.io/gh/JasperSui/fastapi-injectable" target="_blank">
    <img src="https://img.shields.io/codecov/c/github/JasperSui/fastapi-injectable?color=009688&label=Test%20Coverage" alt="Codecov">
</a>
<a href="https://github.com/astral-sh/ruff" target="_blank">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
</a>
<img src="https://img.shields.io/badge/type--checked-mypy-blue?style=flat-square&logo=python&label=type-checked&color=009688" alt="Mypy">
<a href="https://github.com/pre-commit/pre-commit" target="_blank">
    <img src="https://img.shields.io/badge/pre--commit-blue?logo=pre-commit&logoColor=FAB040&color=009688" alt="pre-commit">
</a>
</p>

---

**Installation**: `pip install fastapi-injectable`

**Documentation**: <a href="https://fastapi-injectable.readthedocs.io/en/latest/" target="_blank">https://fastapi-injectable.readthedocs.io/en/latest/</a>

---

## Basic Example

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

## Key Features

1. **Flexible Injection**: Use decorators, function wrappers, or utility functions.
2. **Full Async Support**: Works with both sync and async code.
3. **Resource Management**: Built-in cleanup for dependencies.
4. **Dependency Caching**: Optional caching for better performance.
5. **Graceful Shutdown**: Automatic cleanup on program exit.
6. **Event Loop Management**: Control the event loop to ensure the objects created by `fastapi-injectable` are executed in the right loop.

## Overview

`fastapi-injectable` is a lightweight package that enables seamless use of FastAPI's dependency injection system outside of route handlers. It solves a common pain point where developers need to reuse FastAPI dependencies in non-FastAPI contexts like CLI tools, background tasks, or scheduled jobs, allowing you to use FastAPI's dependency injection system **anywhere**!

### Requirements

- Python `3.10` or higher
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
from fastapi_injectable.decorator import injectable

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
from fastapi_injectable.util import get_injected_obj

class Database:
    def __init__(self) -> None:
        pass

    def query(self) -> str:
        return "data"

def process_data(db: Annotated[Database, Depends(get_database)]):
    return db.query()

# Get injected instance without decorator
result = get_injected_obj(process_data)
print(result) # Output: 'data'
```

### Generator Dependencies with Cleanup

When working with generator dependencies that require cleanup (like database connections or file handles), `fastapi-injectable` provides built-in support for controlling dependency lifecycles and proper resource management with error handling.

Here's an example showing how to work with generator dependencies:

```python
from collections.abc import Generator

from fastapi_injectable.util import cleanup_all_exit_stacks, cleanup_exit_stack_of_func
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

# Option #1: Silent cleanup when done for a single decorated function (logs errors but doesn't raise)
assert machine.db.closed is False
await cleanup_exit_stack_of_func(get_machine)
assert machine.db.closed is True

# Option #2: Strict cleanup with error handling
try:
    await cleanup_exit_stack_of_func(get_machine, raise_exception=True)
except DependencyCleanupError as e:
    print(f"Cleanup failed: {e}")

# Option #3: If you don't care about the other injectable functions,
#              just use the cleanup_all_exit_stacks() to cleanup all at once.
assert machine.db.closed is False
await cleanup_all_exit_stacks() # can still pass the raise_exception=True to raise the error if you want
assert machine.db.closed is True
```

### Async Support

`fastapi-injectable` provides full support for both synchronous and asynchronous dependencies, allowing you to mix and match them as needed. You can freely use async dependencies in sync functions and vice versa. For cases where you need to run async code in a synchronous context, we provide the `run_coroutine_sync` utility function.

```python
from collections.abc import AsyncGenerator

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

# Use it with async/await
result = await async_process_data()
print(result) # Output: 'data'

# In sync func, you can still get the result by using `run_coroutine_sync()`
from fastapi_injectable.concurrency import run_coroutine_sync

result = run_coroutine_sync(async_process_data())
print(result) # Output: 'data'
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

from fastapi_injectable.decorator import injectable

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

### Event Loop Management

`fastapi-injectable` includes a powerful loop management system to handle asynchronous code execution in different contexts. This is particularly useful when working with async code in synchronous environments or when you need controlled event loop execution.

```python
from fastapi_injectable.concurrency import loop_manager, run_coroutine_sync

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
```


### App Registration for State Access

If your dependencies need access to the FastAPI app state (like database connections or other services), you can register your app with `fastapi-injectable`:

```python
from fastapi import FastAPI, Request, Depends
from fastapi_injectable import injectable, register_app
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

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

# Use it in background tasks, CLI tools, etc.
result = await process_data()
```

This is particularly useful when:
- Your dependencies need access to shared services in `app.state`
- You're using third-party libraries that call your code internally
- You want to maintain a single source of truth for long-running services

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
   - Cleanup code in generators runs when `cleanup_exit_stack_of_func()` is called
   - Cache is cleared between messages to prevent memory leaks

3. **Graceful Shutdown**:
   - `setup_graceful_shutdown()` ensures resources are cleaned up on program termination
   - Handles both SIGTERM and SIGINT signals

Please refer to the [Real-world Examples](https://fastapi-injectable.readthedocs.io/en/latest/real-world-examples.html) for more details.

## Frequently Asked Questions

<!-- faq-begin -->

- [Why would I need this package?](#why-would-i-need-this-package)
- [Why not directly use other DI packages like Dependency Injector or FastDepends?](#why-not-directly-use-other-di-packages-like-dependency-injector-or-fastdepends)
- [Can I use it with existing FastAPI dependencies?](#can-i-use-it-with-existing-fastapi-dependencies)
- [Does it work with all FastAPI dependency types?](#does-it-work-with-all-fastapi-dependency-types)
- [What happens to dependency cleanup in long-running processes?](#what-happens-to-dependency-cleanup-in-long-running-processes)
- [Can I mix sync and async dependencies?](#can-i-mix-sync-and-async-dependencies)
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

A: You have three options:
1. Manual cleanup per function: `await cleanup_exit_stack_of_func(your_func)`
2. Cleanup everything: `await cleanup_all_exit_stacks()`
3. Automatic cleanup on shutdown: `setup_graceful_shutdown()`

<hr>

### Can I mix sync and async dependencies?

A: Yes! You can freely mix them. For running async code in sync contexts, use the provided `run_coroutine_sync()` utility.

<hr>

### Are type hints fully supported for `injectable()` and `get_injected_obj()`?

A: Currently, type hint support is a work in progress. However, this doesn't affect the core benefits of the package (seamlessly reusing and maintaining consistency in your FastAPI DI system).

We're actively working on improving type hint support, and we'll have good news on this front soon! In the meantime, enjoy the elegant and clean solution that `fastapi-injectable` provides.

<hr>

### How does caching work?

A: By default, dependencies are cached like in FastAPI routes. You can disable caching with `@injectable(use_cache=False)` if you need fresh instances.

<hr>

### Is it production-ready?

A: Yes! The package has:
- **100%** test coverage
- Type checking with `mypy`
- Comprehensive error handling
- Production use cases documented

<hr>

<!-- faq-end -->

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
