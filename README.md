# Open Pulse Graph Classifier

> [!WARNING]
> This is a WIP repository, it is not yet stable.

This tool is a graph classifier developped for the EPFL Open Pulse project. The graph is composed of nodes representing Github Organizations, Users and Repositories. The classifier aims to classify if a Repository nodes belongs to EPFL Github community or not.

## Important modelling choices

During graph extraction for creating the Pygeometric HeteroData object, the indices of the nodes start at 1 (not zero).

## Installation

Install pinned development dependencies using:

```
pip install -r requirements.txt
```

## Development tools

In order to use [pre-commit](https://pre-commit.com/) hooks, they need to be registered:

```
pre-commit install
```

It is a good practice to manually invoke hooks after installation, just in case:

```
pre-commit run --all-files
```

Unit tests (using [pytest](https://pytest.org/)) are not executed as a pre-commit hook, to keep the overhead to a minimum. Instead, a CI/CD pipeline is configured to run tests after each commit. You can also execute them locally, manually:

```
pytest
```

By default, [mypy](https://mypy-lang.org/) is not executed automatically. You can however run them manually:

```
mypy src
```
