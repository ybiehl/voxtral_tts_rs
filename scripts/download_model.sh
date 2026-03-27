#!/bin/bash
# Download Voxtral-4B-TTS model files (no Python required)
# Usage: bash scripts/download_model.sh [output_dir]

set -e

MODEL_DIR="${1:-models/voxtral-4b-tts}"
BASE_URL="https://huggingface.co/mistralai/Voxtral-4B-TTS-2603/resolve/main"

echo "Downloading Voxtral-4B-TTS to ${MODEL_DIR}..."
mkdir -p "${MODEL_DIR}/voice_embedding"

# Model weights (8GB, BF16)
echo "Downloading consolidated.safetensors (8GB)..."
curl -L -o "${MODEL_DIR}/consolidated.safetensors" \
    "${BASE_URL}/consolidated.safetensors"

# Model configuration
echo "Downloading params.json..."
curl -L -o "${MODEL_DIR}/params.json" \
    "${BASE_URL}/params.json"

# Tekken tokenizer
echo "Downloading tekken.json..."
curl -L -o "${MODEL_DIR}/tekken.json" \
    "${BASE_URL}/tekken.json"

# Voice embeddings (20 preset voices)
echo "Downloading voice embeddings..."
VOICES=(
    casual_female casual_male cheerful_female
    neutral_female neutral_male
    pt_male pt_female
    nl_male nl_female
    it_male it_female
    fr_male fr_female
    es_male es_female
    de_male de_female
    ar_male
    hi_male hi_female
)

for voice in "${VOICES[@]}"; do
    echo "  Downloading ${voice}.pt..."
    curl -L -o "${MODEL_DIR}/voice_embedding/${voice}.pt" \
        "${BASE_URL}/voice_embedding/${voice}.pt"
done

echo ""
echo "Download complete! Model files are in: ${MODEL_DIR}"
echo ""
echo "Files:"
ls -lh "${MODEL_DIR}/"
echo ""
echo "Voice embeddings:"
ls -lh "${MODEL_DIR}/voice_embedding/"
