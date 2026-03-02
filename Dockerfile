FROM python:3.9

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
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


# Copy in the source code
COPY . .

