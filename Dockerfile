# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=12.2"

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --chown=nobody:nogroup . .

# Create non-root user and set permissions
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app \
    && mkdir -p /input /output \
    && chown -R appuser:appuser /input /output

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('CUDA available:', torch.cuda.is_available())" || exit 1

# Default command (can be overridden)
ENTRYPOINT ["python", "main.py"]

# Metadata
LABEL org.opencontainers.image.title="PACE 2025 Breast Cancer Detection" \
      org.opencontainers.image.description="Multi-task model for breast cancer detection in ultrasound images" \
      org.opencontainers.image.vendor="MedViewPro Team" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.created="2025-08-22T00:00:00Z" \
      org.opencontainers.image.licenses="MIT"

# Set working directory
WORKDIR /app

# Set default command (can be overridden when running the container)
CMD ["--help"]