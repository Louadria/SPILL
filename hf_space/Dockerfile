FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# Force CUDA for HF Spaces (local dev defaults to CPU for safety)
ENV SPILL_FORCE_CUDA=1

EXPOSE 7860
CMD ["python", "app.py"]
