# Open Pulse Graph Classifier

> [!WARNING]
> This is a WIP repository, it is not yet stable.

This tool is a graph classifier developped for the EPFL Open Pulse project. The graph is composed of nodes representing Github Organizations, Users and Repositories. The classifier aims to classify if a Repository nodes belongs to EPFL Github community or not.

## Important modelling choices

During graph extraction for creating the Pygeometric HeteroData object, the indices of the nodes start at 1 (not zero).

## Installation

### Docker build (Linux/Windows)

Docker image building :

```
docker build . --platform linux/amd64 -t openpulse
```

Docker image run:

```
docker run -it openpulse
```

Inside docker:
```
python open-pulse-graph-classifier/main.py
```

### Local dev

```
source .venv/bin/activate
uv pip install -r requirements.txt
python open-pulse-graph-classifier/main.py
```

### RunAI

```
runai config project <your project>
runai submit openpulse -i ghcr.io/sdsc-ordes/open-pulse-graph-classifier:latest --gpu 0.02  --attach
```

(where `openpulse` is the job name, change according to your needs.)
