# How to prepare the docker image on Mac-Apple Sillicon (Temporal solution)

## Running on native arm64

1. Build the docker image

``` bash
docker build -t open-pulse-graph-classifier .
```

2. Run the docker image

``` bash
docker run --rm -it open-pulse-graph-classifier
```

3. Install the dependencies using the script provided.

``` bash
bash /app/tools/cpu-arm64-script.sh
```

This script will install the correct versions of all dependencies and run a little test.

4. In order to use the environment please activate the enviroment

```bash
source /arm64/.venv/bin/activate
```

## Emulating the linux/amd64 platform

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
