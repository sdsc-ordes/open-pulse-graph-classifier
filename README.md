# Open Pulse Graph Classifier

> [!WARNING]
> This is a WIP repository, it is not yet stable.

This tool is a graph classifier developped for the EPFL Open Pulse project. The graph is composed of nodes representing Github Organizations, Users and Repositories. The classifier aims to classify if a Repository nodes belongs to EPFL Github community or not.

## Important modelling choices

During graph extraction for creating the Pygeometric HeteroData object, the indices of the nodes start at 1 (not zero).

## Installation

Docker image building:

```
docker build . --build-arg PLATFORM=cpu --platform linux/amd64 -t openpulse
```

Docker image run:

```
docker run -it openpulse
```

Inside docker:
```
uv sync
source .venv/bin/activate
python open-pulse-graph-classifier/main.py
```
