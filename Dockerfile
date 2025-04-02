ARG PLATFORM
ARG UV_ARGUMENTS=""

# We select the image based on the platform argument

# Define GPU image
FROM nvidia/cuda:12.4.1-base-ubuntu22.04 AS gpu
ARG PLATFORM
RUN echo "Using GPU image"

# Define cpu image
FROM ubuntu:22.04 AS cpu
ARG PLATFORM
ARG UV_ARGUMENTS="--extra cpu"
RUN echo "Using CPU-only image"

# Select image
FROM ${PLATFORM:-gpu}
ARG PLATFORM

COPY --from=ghcr.io/astral-sh/uv:0.5.8 /uv /uvx /bin/

RUN apt-get update && \
   apt-get install -y  --no-install-recommends  \
      nano curl git ca-certificates


# Copy the project into the image
ADD . /app

# Sync the project into a new environment, using the frozen lockfile
WORKDIR /app


# Create a virtual environment with UV
#RUN uv venv
#RUN uv pip install --no-build-isolation setuptools wheel

# Define the build argument with a default value of an empty string (optional)
#RUN uv sync --no-build-isolation ${UV_ARGUMENTS}


# make uv's python the default python for the image
ENV PATH="/app/.venv/bin:$PATH"

ENV DASK_DISTRIBUTED__WORKER__DAEMON=False

ENTRYPOINT ["/bin/bash"]
#RUN python /open-pulse-graph-classifier/main.py


# FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# RUN apt-get update && apt-get install -y apt-utils

# RUN apt-get install -y python3.11 python3.11-venv python3-pip

# # Create directories and set permissions before switching to the non-root user
# # RUN mkdir -p /open-pulse-classifier \
# #     /home/user && \
# #     chown -R 1000:1000 /odtp /home/user

# COPY . /open-pulse-classifier
# RUN pip install -r /open-pulse-classifier/requirements.txt --no-build-isolation

# # Adjust permissions so user 1000 can access /usr/local/bin
# # RUN chown -R 1000:1000 /usr/local/bin/
# # Switch back to the "user" user
# # USER 1000

# # Fix for end of the line issue on Windows. Avoid error when building on windows
# RUN python /open-pulse-classifier/open-pulse-classifier/main.py
