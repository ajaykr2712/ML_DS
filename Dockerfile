# ML Arsenal - Production Dockerfile
# ===================================
# Multi-stage build for optimized production deployment
# Supports both CPU and GPU environments

# Build arguments
ARG PYTHON_VERSION=3.9
ARG BASE_IMAGE=python:${PYTHON_VERSION}-slim

# =============================================================================
# Stage 1: Build Dependencies
# =============================================================================
FROM ${BASE_IMAGE} as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libc6-dev \
    libffi-dev \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
COPY requirements-prod.txt /tmp/requirements-prod.txt

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /tmp/requirements.txt && \
    pip install -r /tmp/requirements-prod.txt

# =============================================================================
# Stage 2: Runtime Image
# =============================================================================
FROM ${BASE_IMAGE} as runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN groupadd -r ml_user && useradd -r -g ml_user ml_user

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create application directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY setup.py .
COPY README.md .
COPY LICENSE .

# Install the application
RUN pip install -e .

# Create necessary directories
RUN mkdir -p \
    data/samples \
    models/production \
    logs \
    experiments \
    reports \
    && chown -R ml_user:ml_user /app

# Copy entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && chown ml_user:ml_user /entrypoint.sh

# Switch to non-root user
USER ml_user

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]

# =============================================================================
# Stage 3: Development Image (optional)
# =============================================================================
FROM runtime as development

# Switch back to root for development tools installation
USER root

# Install development dependencies
COPY requirements-dev.txt /tmp/requirements-dev.txt
RUN pip install -r /tmp/requirements-dev.txt

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    nano \
    htop \
    tree \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Jupyter Lab
RUN pip install jupyterlab

# Copy development configuration
COPY .dev/ ./.dev/
COPY notebooks/ ./notebooks/
COPY tests/ ./tests/

# Switch back to non-root user
USER ml_user

# Expose Jupyter port
EXPOSE 8888

# Development entrypoint
ENTRYPOINT ["/entrypoint.sh"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# =============================================================================
# Build Information
# =============================================================================

# Add build information as labels
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

LABEL maintainer="ML Arsenal Team" \
      org.opencontainers.image.title="ML Arsenal" \
      org.opencontainers.image.description="Production-ready machine learning platform" \
      org.opencontainers.image.version=${VERSION} \
      org.opencontainers.image.created=${BUILD_DATE} \
      org.opencontainers.image.revision=${VCS_REF} \
      org.opencontainers.image.vendor="ML Arsenal" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.url="https://github.com/your-org/ml-arsenal" \
      org.opencontainers.image.source="https://github.com/your-org/ml-arsenal"
