# Single stage build for CPU-only environment
FROM python:3.11-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    HOME="/home/appuser"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create appuser with home directory and set up directories
RUN useradd -m -d /home/appuser -s /bin/bash appuser && \
    mkdir -p /app/test_data/input /app/test_data/output && \
    chown -R appuser:appuser /app /home/appuser

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Switch to non-root user and set working directory
USER appuser
WORKDIR $HOME

# Copy application code with correct permissions
COPY --chown=appuser:appuser . /app/

# Set working directory back to app
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD ["python", "-c", "print('Service is healthy')"]

# Default command
ENTRYPOINT ["python", "app/main.py"]

# Metadata
LABEL org.opencontainers.image.title="PACE 2025 Breast Cancer Detection" \
      org.opencontainers.image.description="Multi-task model for breast cancer detection in ultrasound images" \
      org.opencontainers.image.vendor="MedViewPro Team" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.created="2025-08-22T00:00:00Z" \
      org.opencontainers.image.licenses="MIT"

# Set default command (can be overridden when running the container)
CMD ["--help"]