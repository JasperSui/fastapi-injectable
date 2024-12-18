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

You need to have Python 3.10, 3.11, 3.12 at the same time, you can use [pyenv] to do so, once you have installed [pyenv], you can use the following command to install the required python versions:

```console
$ pyenv install 3.10 3.11 3.12
```

And you also need the following packages:

- [Poetry]
- [Nox]
- [nox-poetry]


If you don't have any of these, you can use pip to install them directly:
```console
$ pip install -r requirements/dev.txt # Optional if you already have them
```

And then install the package with development requirements:

```console
$ poetry install
```

You can now run an interactive Python session

```console
$ poetry run python
```

[poetry]: https://python-poetry.org/
[nox]: https://nox.thea.codes/
[nox-poetry]: https://nox-poetry.readthedocs.io/
[pyenv]: https://github.com/pyenv/pyenv

## How to test the project

Run the full test suite:

```console
$ nox
```

List the available Nox sessions:

```console
$ nox --list-sessions
```

You can also run a specific Nox session.
For example, invoke the unit test suite like this:

```console
$ nox --session=tests
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
$ nox --session=pre-commit -- install
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

You may follow this [StackOverflow solution](https://stackoverflow.com/a/76266406) to fix it by install `sqlite3-dev` in your os first, then `pyenv install` again.

[pull request]: https://github.com/JasperSui/fastapi-injectable/pulls

<!-- github-only -->

[code of conduct]: CODE_OF_CONDUCT.md
