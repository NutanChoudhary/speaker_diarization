FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Basic environment setup
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VENV_PATH=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Install system dependencies and Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        build-essential ffmpeg libsndfile1 curl git cmake && \
    rm -rf /var/lib/apt/lists/*

# Create venv and upgrade pip
RUN python3 -m venv $VENV_PATH && \
    pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements first (layer cache)
COPY requirements.txt .

# Install PyTorch (cu124 to match CUDA 12.4 base image)
RUN pip install --no-cache-dir \
        torch==2.5.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu124

# Install onnxruntime-gpu — latest available on PyPI is 1.23.2
RUN pip install --no-cache-dir \
        onnxruntime-gpu==1.23.2

# Install remaining project requirements

#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir \
        librosa \
        soundfile \
        numpy\
        scipy\
        packaging \
        psutil

# Copy the rest of the application

RUN mkdir -p /app/output

COPY . .

# Create non-root user and adjust ownership
RUN useradd -m appuser && \
    chown -R appuser:appuser /app /opt/venv
USER appuser

# Usage: docker run --gpus all -v /path/to/audio:/audio diarization:latest /audio/file.wav
ENTRYPOINT ["python3", "diarize.py"]
CMD ["--help"]
