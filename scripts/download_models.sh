#!/usr/bin/env bash
# ============================================================================
# Athena — Model Download Script
#
# Downloads:
#   1. LLM model via Ollama (gemma3:12b-it-qat)
#   2. FLUX.1-schnell via diffusers (auto-downloaded at first boot)
#
# Requirements:
#   - Docker (for Ollama)
#   - HF_TOKEN in .env (FLUX.1 is gated)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${PROJECT_DIR}/models"

OLLAMA_MODEL="${OLLAMA_MODEL:-gemma3:12b-it-qat}"

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

# ── 2. FLUX.1-schnell ─────────────────────────────────────────────────────
echo "── FLUX.1-schnell ─────────────────────────"
echo ""
echo "  FLUX.1-schnell は初回起動時に自動ダウンロードされます。"
echo "  ダウンロードには HF_TOKEN が必要です（.env に設定済みか確認）。"
echo ""
echo "  事前に手動ダウンロードする場合は以下を実行:"
echo "    docker compose run --rm athena python3 -c \\"
echo "      \"from diffusers import AutoPipelineForText2Image; \\"
echo "       AutoPipelineForText2Image.from_pretrained( \\"
echo "         'black-forest-labs/FLUX.1-schnell', \\"
echo "         cache_dir='/app/models')\""
echo ""

echo "============================================"
echo "  ✅ Setup complete!"
echo ""
echo "  LLM:     Ollama (${OLLAMA_MODEL})"
echo "  Image:   FLUX.1-schnell (auto-download on first boot)"
echo ""
echo "  Start with:  docker compose up -d"
echo "============================================"
