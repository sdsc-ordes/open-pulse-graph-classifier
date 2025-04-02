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

### How to prepare the docker image on Mac-Apple Sillicon (Temporal solution)

#### Emulating the linux/amd64 paltform

1. Build the docker image with `--platform` flag.

``` bash
docker build -t open-pulse-graph-classifier --platform linux/amd64 .
```

2. Run the docker image using `--platform linux/amd64`

``` bash
docker run --rm -it --platform linux/amd64 open-pulse-graph-classifier
```

3. Install the dependencies using the script provided.

``` bash
bash /app/tools/cpu-amd64-script.sh
```

This script will install the correct versions of all dependencies and run a little test. 

4. In order to use the environment please activate the enviroment

```bash
source /amd64/.venv/bin/activate
```

