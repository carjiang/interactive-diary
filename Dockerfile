FROM python:3.12-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies (audio + ffmpeg)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        portaudio19-dev \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python setup
RUN pip install --upgrade pip

# Install PyTorch (CPU build by default)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install app dependencies first (better caching)
COPY docker_requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# App directory
WORKDIR /app
ENV PYTHONPATH=/app

# Copy source last (so code changes don’t bust dependency cache)
COPY . /app
