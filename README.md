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
  await cleanup_exit_stack_of_func(process_data)  # Cleanup specific function's resources
  await cleanup_all_exit_stacks()  # Cleanup all resources
  ```

- **Dependency Caching**: Optional caching of resolved dependencies for better performance
  ```python
  from fastapi_injectable.decorator import injectable

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
   * [Requirements](#requirements)
   * [Installation](#installation)
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

## Real-world Examples

### 1. Using `Depends` in in-house background worker

Here's a practical example of using `fastapi-injectable` in a background worker that processes messages.

You can find the complete example with more details in the [examples/worker](https://github.com/JasperSui/fastapi-injectable/tree/main/example/worker) directory.

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

This pattern is particularly useful for:
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
[his work]: https://github.com/fastapi/fastapi/discussions/7720#discussioncomment-11581187

## Bonus

My blog posts about the prototype of this project:

1. [Easily Reusing Depends Outside FastAPI Routes](https://j-sui.com/2024/10/26/use-fastapi-depends-outside-fastapi-routes-en/)
2. [在 FastAPI Routes 以外無痛複用 Depends 的方法](https://j-sui.com/2024/10/26/use-fastapi-depends-outside-fastapi-routes/)

<!-- github-only -->

[license]: https://github.com/JasperSui/fastapi-injectable/blob/main/LICENSE
[contributor guide]: https://github.com/JasperSui/fastapi-injectable/blob/main/CONTRIBUTING.md
