FROM python:3.12

# Install ffmpeg and protaudio for audio processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Update pip
RUN pip install --upgrade pip

# Install PyTorch
RUN pip install --no-cache-dir torch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html

# Install the application dependencies
COPY docker_requirements.txt ./
RUN pip install --no-cache-dir -r docker_requirements.txt

# Set working directory and Python path so local packages are importable
WORKDIR /app
ENV PYTHONPATH=/app

# Copy in the source code
COPY . /app

