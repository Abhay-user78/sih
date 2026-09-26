# Stage-6 deployment image: Spatio-Temporal cyclone early-warning pipeline.
# CPU-oriented image -- the API and dashboard services do not need a GPU.
# Scheduled ERA5 cycles also run from this image (see README, Stage-2 cron).
FROM python:3.13-slim

# libgomp1 = OpenMP runtime for CPU torch. Avoids heavy geo system libs;
# Cartopy/cfgrib are only used by offline scripts, not the API/dashboard.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# 1) CPU torch first (skips the ~3 GB CUDA bundle).
# 2) The rest from PyPI. PyG companion binaries (torch-scatter/sparse) are
#    best-effort from the PyG wheel index; PyG 2.8 has a pure-torch fallback
#    for the scatter ops used by the GNN, so failure there is non-fatal.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0 \
 && pip install -r requirements.txt \
 && (pip install torch-scatter torch-sparse \
        -f https://data.pyg.org/whl/torch-2.11.0.html \
     || echo "PyG companion wheels unavailable -- using pure-torch fallback")

# Application code + frozen weights (weights load purely from state dicts/
# configs; no runtime internet needed for inference).
COPY weather_pipeline ./weather_pipeline
COPY scripts ./scripts
COPY app ./app
COPY models ./models
# Streamlit base theme (app/theme.py injects the rest of the visual system).
COPY .streamlit ./.streamlit

RUN mkdir -p /app/data /app/outputs \
 && useradd -m -u 10001 appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8501

# Default = API; docker-compose overrides per service.
CMD ["uvicorn", "weather_pipeline.api:app", "--host", "0.0.0.0", "--port", "8000"]