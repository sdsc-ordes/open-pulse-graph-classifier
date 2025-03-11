FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y apt-utils

RUN apt-get install -y python3.11 python3.11-venv python3-pip

# Create directories and set permissions before switching to the non-root user
RUN mkdir -p /open-pulse-classifier \
    /home/user && \
    chown -R 1000:1000 /odtp /home/user


COPY . /open-pulse-classifier
RUN pip install -r /open-pulse-classifier/requirements.txt

# Adjust permissions so user 1000 can access /usr/local/bin
RUN chown -R 1000:1000 /usr/local/bin/

# Switch back to the "user" user
USER 1000
# Fix for end of the line issue on Windows. Avoid error when building on windows
RUN python /open-pulse-classifier/open-pulse-classifier/main.py
