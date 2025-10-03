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

for training

```
python open-pulse-graph-classifier/inference/api.py
```

for inference

```
python open-pulse-graph-classifier/train/main.py
```

### Local dev

```
source .venv/bin/activate
uv pip install -r requirements.txt
cd src
python training.py
```

### RunAI

#### Training

```
runai project set <your project>
```

```
runai workspace submit openpulse-training2 \
  -i ghcr.io/sdsc-ordes/open-pulse-graph-classifier-train:latest \
  --gpu-request-type portion --gpu-portion-request 0.9 \
  --image-pull-policy Always \
  --attach
```

Then interactively run training by :

1. entering into the container: `runai workspace bash openpulse-training`
2. going to the right repository: `cd ../app/`
3. running training script: `python training.py`

(remember to set any env variables needed beforehand)

#### Inference

To do: this needs update to new runai cli

```
runai inference submit openpulse-inference \
  -p openpulse-laure \
  -i ghcr.io/sdsc-ordes/open-pulse-graph-classifier-inference:latest \
  --serving-port=8000 \
  --min-replicas=0 \
  --max-replicas=1 \
  --scale-down-delay-seconds 3600 \
  --image-pull-policy Always \
  -c -- python inference_api.py
```

(where `openpulse` is the job name, change according to your needs.)
