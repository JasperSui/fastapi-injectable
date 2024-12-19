# FastAPI Injectable

[![PyPI](https://img.shields.io/pypi/v/fastapi-injectable.svg)][pypi_]
[![Status](https://img.shields.io/pypi/status/fastapi-injectable.svg)][status]
[![Python Version](https://img.shields.io/pypi/pyversions/fastapi-injectable)][python version]
[![License](https://img.shields.io/pypi/l/fastapi-injectable)][license]

[![Read the documentation at https://fastapi-injectable.readthedocs.io/](https://img.shields.io/readthedocs/fastapi-injectable/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/JasperSui/fastapi-injectable/workflows/Tests/badge.svg)][tests]


[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]

[pypi_]: https://pypi.org/project/fastapi-injectable/
[status]: https://pypi.org/project/fastapi-injectable/
[python version]: https://pypi.org/project/fastapi-injectable
[read the docs]: https://fastapi-injectable.readthedocs.io/
[tests]: https://github.com/JasperSui/fastapi-injectable/actions?workflow=Tests
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

A lightweight package that lets you use FastAPI's dependency injection anywhere - not just in route handlers. Perfect for CLI tools, background tasks, and more.

## Overview

`fastapi-injectable` is a lightweight package that enables seamless use of FastAPI's dependency injection system outside of route handlers. It solves a common pain point where developers need to reuse FastAPI dependencies in non-HTTP contexts like CLI tools, background tasks, or scheduled jobs.

## Quick Start

Key features:

- **Flexible Injection APIs**: Choose between decorator or function-based approaches
  ```python
  from fastapi_injectable.decorator import injectable
  from fastapi_injectable.util import get_injected_obj

  # 1. Decorator approach
  @injectable
  def process_data(db: Annotated[Database, Depends(get_db)]):
      return db.query()
  result1 = process_data()

  # 2. Function-wrapper approach
  def process_data(db: Annotated[Database, Depends(get_db)]):
      return db.query()

  injectable_process_data = injectable(process_data)
  result2 = injectable_process_data()

  # 3. Use the utility
  def process_data(db: Annotated[Database, Depends(get_db)]):
      return db.query()

  result3 = get_injected_obj(process_data)

  # They are all the same!
  assert result1 == result2 == result3
  ```

- **Support for Both Sync and Async**: Works seamlessly with both synchronous and asynchronous code
  ```python
  from fastapi_injectable.decorator import injectable

  def get_service():
      return Service()

  @injectable
  async def async_task(service: Annotated[Service, Depends(get_service)]):
      await service.process()
  ```

- **Controlled Resource Management**: Explicit cleanup of dependencies through utility functions
  ```python
  from fastapi_injectable.decorator import injectable
  from fastapi_injectable.util import cleanup_all_exit_stacks, cleanup_exit_stack_of_func

  # Define a dependency with cleanup
  def get_db() -> Generator[Database, None, None]:
      db = Database()
      yield db
      db.cleanup()  # Called when cleanup functions are invoked

  # Use the dependency
  @injectable
  def process_data(db: Annotated[Database, Depends(get_db)]):
      return db.query()

  # Cleanup options
  await cleanup_exit_stack_of_func(process_data)  # Option #1: Cleanup specific function's resources
  await cleanup_all_exit_stacks()  # Option #2: Cleanup all resources
  ```

- **Dependency Caching**: Optional caching of resolved dependencies for better performance
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

- **Graceful Shutdown**: Built-in utilities for proper cleanup during application shutdown
  ```python
  from fastapi_injectable import setup_graceful_shutdown

  setup_graceful_shutdown()  # Handles SIGTERM and SIGINT
  ```

This package is particularly useful for:
- Background task workers
- CLI applications
- Scheduled jobs
- Test fixtures
- Any non-HTTP context where you want to leverage FastAPI's dependency injection

## Table of Content
- [FastAPI Injectable](#fastapi-injectable)
   * [Overview](#overview)
   * [Quick Start](#quick-start)
   * [Table of Content](#table-of-content)
   * [Requirements](#requirements)
   * [Installation](#installation)
   * [Usage](#usage)
      + [Basic Dependency Injection](#basic-dependency-injection)
      + [Function-based Approach](#function-based-approach)
      + [Generator Dependencies with Cleanup](#generator-dependencies-with-cleanup)
      + [Async Support](#async-support)
      + [Dependency Caching Control](#dependency-caching-control)
      + [Graceful Shutdown](#graceful-shutdown)
   * [Advanced Scenarios](#advanced-scenarios)
      + [1. `test_injectable.py` - Shows all possible combinations of:](#1-test_injectablepy-shows-all-possible-combinations-of)
      + [2. `test_integration.py` - Demonstrates:](#2-test_integrationpy-demonstrates)
   * [Real-world Examples](#real-world-examples)
      + [1. Using `Depends` in in-house background worker](#1-using-depends-in-in-house-background-worker)
   * [Contributing](#contributing)
   * [License](#license)
   * [Issues](#issues)
   * [Credits](#credits)
   * [Bonus](#bonus)

## Requirements

- Python `3.10` or higher
- FastAPI `0.112.4` or higher

## Installation

You can install `fastapi-injectable` via [pip] from [PyPI]:

```console
$ pip install fastapi-injectable
```

## Usage

`fastapi-injectable` provides several powerful ways to use FastAPI's dependency injection outside of route handlers. Let's explore the key usage patterns with practical examples.

### Basic Dependency Injection

The most basic way to use dependency injection is through the `@injectable` decorator. This allows you to use FastAPI's `Depends` in any function, not just route handlers.

Here's a simple example:

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

When working with generator dependencies that require cleanup (like database connections or file handles), `fastapi-injectable` provides built-in support for controlling dependency lifecycles and proper resource management by using cleanup functions.

Here's an example showing how to work with generator dependencies:

```python
from collections.abc import Generator

from fastapi_injectable.util import cleanup_all_exit_stacks, cleanup_exit_stack_of_func

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

# Option #1: Cleanup when done for a single decorated function
assert machine.db.closed is False
await cleanup_exit_stack_of_func(get_machine)
assert machine.db.closed is True

# Option #2: If you don't care about the other injectable functions,
#              just use the cleanup_all_exit_stacks() to cleanup all at once.
assert machine.db.closed is False
await cleanup_all_exit_stacks()
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

### Graceful Shutdown

If you don't care about the generator's lifecycle and just want to ensure proper cleanup when the program exits, you can register cleanup functions anywhere:

```python
import signal

from fastapi_injectable import setup_graceful_shutdown

# Setup automatic cleanup on shutdown
setup_graceful_shutdown()  # Handles SIGTERM and SIGINT by signal, and also atexit

# Or specify custom signals
setup_graceful_shutdown(signals=[signal.SIGTERM])
```

## Advanced Scenarios

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

## Real-world Examples

### 1. Using `Depends` in in-house background worker

Here's a practical example of using `fastapi-injectable` in a background worker that processes messages.

You can find the complete example with more details in the [examples/worker/main.py](https://github.com/JasperSui/fastapi-injectable/tree/main/example/worker/main.py) file.

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

You can extend the example to re-using the business logic in your:
- Message queue consumers
- Batch processing jobs
- Long-running background tasks
- Any scenario where you need FastAPI-style dependency injection in a worker process

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license],
`fastapi-injectable` is free and open source software.

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

## Bonus

My blog posts about the prototype of this project:

1. [Easily Reusing Depends Outside FastAPI Routes](https://j-sui.com/2024/10/26/use-fastapi-depends-outside-fastapi-routes-en/)
2. [在 FastAPI Routes 以外無痛複用 Depends 的方法](https://j-sui.com/2024/10/26/use-fastapi-depends-outside-fastapi-routes/)

<!-- github-only -->

[license]: https://github.com/JasperSui/fastapi-injectable/blob/main/LICENSE
[contributor guide]: https://github.com/JasperSui/fastapi-injectable/blob/main/CONTRIBUTING.md
