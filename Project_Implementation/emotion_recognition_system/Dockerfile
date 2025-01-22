# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt \
    && pip3 install --no-cache-dir transformers[torch] \
    && python3 -c "from transformers import BertTokenizer; BertTokenizer.from_pretrained('bert-base-uncased')"

# Copy source code
COPY . .

# Download NLTK data
RUN python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

CMD ["python3", "src/train.py"] 