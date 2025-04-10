ARG PLATFORM
ARG UV_ARGUMENTS=""

# We select the image based on the platform argument

# Define GPU image
FROM nvidia/cuda:12.4.1-base-ubuntu22.04 AS cu124
ARG PLATFORM
RUN echo "Using GPU image"
ARG UV_ARGUMENTS="--extra cu124"


# Define cpu image
FROM ubuntu:22.04 AS cpu
ARG PLATFORM
ARG UV_ARGUMENTS="--extra cpu"
RUN echo "Using CPU-only image"

# Select image
FROM ${PLATFORM:-cu124}
ARG PLATFORM


ARG TARGETPLATFORM
RUN echo "Building for platform: ${TARGETPLATFORM}"


COPY --from=ghcr.io/astral-sh/uv:0.6.12 /uv /uvx /bin/


# Copy the project into the image
ADD . /app

# Sync the project into a new environment, using the frozen lockfile
WORKDIR /app

# Define the build argument with a default value of an empty string (optional)
# RUN uv sync ${UV_ARGUMENTS}
RUN uv sync


# make uv's python the default python for the image
ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["/bin/bash"]
