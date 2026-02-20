#!/usr/bin/env bash
# ============================================================================
# Athena — Model Download Script
#
# Downloads:
#   1. LLM model via Ollama (gemma3:12b-it-qat)
#   2. Stable Diffusion v1.5 via diffusers
#
# Requirements:
#   - Docker (for Ollama) or Ollama installed locally
#   - Python 3.11+ with diffusers & torch
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${PROJECT_DIR}/models"

OLLAMA_MODEL="${OLLAMA_MODEL:-gemma3:12b-it-qat}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"

echo "============================================"
echo "  Athena — Model Downloader"
echo "============================================"
echo ""

mkdir -p "$MODELS_DIR"

# ── 1. LLM Model (via Ollama) ─────────────────────────────────────────────
echo "── LLM Model ──────────────────────────────"
echo "  Model: ${OLLAMA_MODEL}"
echo ""

# Check if Ollama service is running in Docker
if docker compose ps ollama | grep -q "Up"; then
    echo "✓ Ollama container is running"

    # Pull model using docker exec
    echo "⬇ Pulling model '${OLLAMA_MODEL}' via Ollama container …"
    docker compose exec ollama ollama pull "${OLLAMA_MODEL}"
    echo ""
    echo "✓ Model pulled successfully"
else
    echo "⚠ Ollama container is not running"
    echo ""
    echo "  Start Ollama first with:"
    echo "    docker compose up -d ollama"
    echo "  Then re-run this script."
fi

echo ""

# ── 2. Stable Diffusion v1.5 ──────────────────────────────────────────────
SD_DIR="${MODELS_DIR}/stable-diffusion-v1-5"
echo "── Stable Diffusion ───────────────────────"

if [ -d "$SD_DIR" ] && [ "$(ls -A "$SD_DIR" 2>/dev/null)" ]; then
    echo "✓ SD model already exists: stable-diffusion-v1-5/"
else
    echo "⬇ Downloading Stable Diffusion v1.5 (~4 GB) …"
    echo "  (Using Athena container to download)"
    echo ""

    # Create a temporary python script to download model
    cat <<EOF > "${PROJECT_DIR}/_download_sd.py"
from diffusers import StableDiffusionPipeline
import torch
import os

sd_dir = '/app/models/stable-diffusion-v1-5'
print(f'Downloading SD v1.5 to {sd_dir} ...')

try:
    pipe = StableDiffusionPipeline.from_pretrained(
        'stable-diffusion-v1-5/stable-diffusion-v1-5',
        torch_dtype=torch.float16,
        cache_dir=sd_dir,
    )
    print('✓ SD model downloaded successfully')
except Exception as e:
    print(f'✗ Failed to download SD model: {e}')
    exit(1)
EOF

    # Run the script inside the container
    # We use 'athena' service but verify it can run python. 
    # Since athena likely has the environment, we can use 'run' or 'exec' if it's up.
    # Ideally 'run --rm' to not leave containers, and mount the volume.
    # However, 'athena' service mounts './models:/app/models'.
    
    echo "  Running download script inside Docker container..."
    docker compose run --rm --entrypoint python3 athena /app/_download_sd.py # Assuming the script is mounted or we can copy it? 
    # Wait, the script is created in PROJECT_DIR which is mounted as root via "." in build context? No.
    # The athena service context is ".", but volumes are specific.
    # docker-compose.yml has:
    # volumes:
    #   - ./models:/app/models
    
    # We need to make the script available to the container. 
    # Since we can't easily mount a single file dynamically in 'docker compose run' without overriding volumes cleanly (though we can),
    # let's just pass the code via stdin if possible or mount the current dir.
    
    # Alternative: Mount the current directory to /tmp and run it.
    docker compose run --rm -v "${PROJECT_DIR}/_download_sd.py:/tmp/download_sd.py" --entrypoint python3 athena /tmp/download_sd.py
    
    # Cleanup
    rm "${PROJECT_DIR}/_download_sd.py"

    echo "✓ Download process completed"
fi

echo ""
echo "============================================"
echo "  ✅ Setup complete!"
echo ""
echo "  LLM:  Ollama (${OLLAMA_MODEL})"
echo "  SD:   ${SD_DIR}"
echo ""
echo "  Start with:  docker compose up -d"
echo "============================================"
