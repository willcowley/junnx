# Contributing

JUNNX is still a work in progress, if you'd like to be involved in its development, please get in touch!

Here are some pointers:

---
## Getting started

JUNNX uses [poetry](https://python-poetry.org/) to manage its dependencies. Installation
instructions for poetry can be found [here](https://python-poetry.org/docs/#installation).
```bash
git clone git@github.com:willcowley/junnx.git
cd junnx
poetry install
```

This will install JUNNX and all its dependencies, including those required for
development, into a virtual environment.

---
## Development

Some helpful development tasks: formatting, linting and typechecking can be run with the
following commands:

```bash
poetry run task format
poetry run task lint
poetry run task mypy
```

JUNNX currently uses `black` and `isort` for formatting, `ruff` for linting and `mypy` for
typechecking.

The tests can be run with `pytest` via:

```bash
JAX_ENABLE_X64=1 poetry run task test
```

This will also create a coverage report in the `htmlcov` directory.

---
## Documentation

The documentation is built using `mkdocs` and can be run with:

```bash
JAX_ENABLE_X64=1 poetry run mkdocs serve
```

Note that this will execute the notebooks in `docs/examples`, so may take a few minutes.