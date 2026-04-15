# NVIDIA NGC PyTorch container — arm64-sbsa build with Blackwell / GB10 support.
# Bumped from 25.03 → 26.03 to pick up proper sm_120 kernels for SDPA and
# attention paths. On 25.03 we saw edit generations take ~290s for 4 steps
# because several ops fell back to reference implementations on sm_120.
# Keep the Blackwell nvrtc shim in loaders/_blackwell_shims.py until this
# image is proven to not need it.
FROM nvcr.io/nvidia/pytorch:26.03-py3

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
