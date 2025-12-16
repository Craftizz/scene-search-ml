FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential git libgl1 libglib2.0-0 libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Optionally pre-download smaller model snapshots at build-time to bake them into the image.
# Set PRELOAD_MODELS=0 to skip download. If your models are private, pass
# --build-arg HUGGINGFACE_HUB_TOKEN=<token>
ARG PRELOAD_MODELS=1
ARG HUGGINGFACE_HUB_TOKEN=""
RUN if [ "$PRELOAD_MODELS" = "1" ]; then \
      python -c "from huggingface_hub import snapshot_download; import os; token=os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN') or None; snapshot_download('google/siglip-base-patch16-224', local_dir='/app/models/siglip', local_dir_use_symlinks=False, token=token)"; \
    else echo 'Skipping model preload'; fi

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
