# NVIDIA NGC PyTorch container — arm64-sbsa build with Blackwell / GB10 support.
# The 25.03 tag ships PyTorch 2.7 + CUDA 12.8 + cuDNN 9 and officially supports
# compute capability sm_120 (Blackwell consumer + GB10).
FROM nvcr.io/nvidia/pytorch:25.03-py3

WORKDIR /app

# HF cache lives on a mounted volume (see docker-compose.yml). We still set
# the env var so diffusers/transformers know where to look inside the container.
ENV HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Base Python deps first (for layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r requirements.txt \
 && pip install hf_transfer \
 && pip install "git+https://github.com/huggingface/diffusers"

# App code
COPY server.py /app/server.py
COPY loaders /app/loaders
COPY web /app/web

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
