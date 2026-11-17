FROM python:3.11-slim

# Install system libraries for OpenCV, RapidOCR, and OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up user with UID 1000 (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /home/user/app

# Copy dependency definition and install
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application source code and assets
COPY --chown=user:user . .

# Expose port 7860 (default for Hugging Face Spaces)
EXPOSE 7860

# Launch server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
