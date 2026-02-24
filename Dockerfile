FROM python:3.11-slim

# Install system dependencies
# git is often needed for installing dependencies from git or diffusers components
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' /usr/local/lib/python3.11/site-packages/basicsr/data/degradations.py

# Copy application code
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Command to run the bot
CMD ["python", "-m", "bot.main"]
