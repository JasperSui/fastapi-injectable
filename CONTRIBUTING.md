# Contributor Guide

Thank you for your interest in improving this project.
This project is open-source under the [MIT license] and
welcomes contributions in the form of bug reports, feature requests, and pull requests.

Here is a list of important resources for contributors:

- [Source Code]
- [Documentation]
- [Issue Tracker]
- [Code of Conduct]

[mit license]: https://opensource.org/licenses/MIT
[source code]: https://github.com/JasperSui/fastapi-injectable
[documentation]: https://fastapi-injectable.readthedocs.io/
[issue tracker]: https://github.com/JasperSui/fastapi-injectable/issues

## How to report a bug

Report bugs on the [Issue Tracker].

When filing an issue, make sure to answer these questions:

- Which operating system and Python version are you using?
- Which version of this project are you using?
- What did you do?
- What did you expect to see?
- What did you see instead?

The best way to get your bug fixed is to provide a test case,
and/or steps to reproduce the issue.

## How to request a feature

Request features on the [Issue Tracker].

## How to set up your development environment

You need to have Python 3.10, 3.11, 3.12, 3.13, 3.13t, 3.14, 3.14t installed. With uv, you can easily install and manage multiple Python versions:

```console
$ uv python install 3.10 3.11 3.12 3.13 3.13t 3.14 3.14t
```

**Note:** uv will be installed automatically when you run the commands above. Alternatively, you can install uv manually by following the instructions on the [official website](https://astral.sh/uv/install.sh). This project requires uv version 0.9.2 or higher. You can verify your version with `uv --version`.

Install dependencies:

```console
$ uv sync
```

**About dependency management:**
- Dependencies are defined in `pyproject.toml` under `[dependency-groups]`
- `uv.lock` is the lockfile (similar to what `poetry.lock` was)
- After modifying dependencies in `pyproject.toml`, run `uv lock` to update the lockfile
- Run `uv sync` to install dependencies from the lockfile
- All development tools (including [Nox]) are managed by uv

You can now run an interactive Python session or use development tools:

```console
$ uv run python
$ uv run nox
```

[uv]: https://astral.sh/uv
[nox]: https://nox.thea.codes/

## How to test the project

Run the full test suite:

```console
$ uv run nox
```

List the available Nox sessions:

```console
$ uv run nox --list-sessions
```

You can also run a specific Nox session.
For example, invoke the unit test suite like this:

```console
$ uv run nox --session=tests
```

Unit tests are located in the _test_ directory,
and are written using the [pytest] testing framework.

[pytest]: https://pytest.readthedocs.io/

## How to submit changes

Open a [pull request] to submit changes to this project.

Your pull request needs to meet the following guidelines for acceptance:

- The Nox test suite must pass without errors and warnings.
- Include unit tests. This project maintains 100% code coverage.
- If your changes add functionality, update the documentation accordingly.

Feel free to submit early, though—we can always iterate on this.

To run linting and code formatting checks before committing your change, you can install pre-commit as a Git hook by running the following command:

```console
$ uv run nox --session=pre-commit -- install
```

It is recommended to open an issue before starting work on anything.
This will allow a chance to talk it over with the owners and validate your approach.

## Troubleshootings

If you get errors about `_sqlite3` module not found:

```
    import datetime
    import time
    import collections.abc

>   from _sqlite3 import *
E   ModuleNotFoundError: No module named '_sqlite3
```

You may follow this [StackOverflow solution](https://stackoverflow.com/a/76266406) to fix it by install `sqlite3-dev` in your os first, then run `uv python install <version>` again.

[pull request]: https://github.com/JasperSui/fastapi-injectable/pulls

<!-- github-only -->

[code of conduct]: CODE_OF_CONDUCT.md
