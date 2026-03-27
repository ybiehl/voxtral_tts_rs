#!/bin/bash
# Installer for Voxtral TTS Rust — downloads binaries, models, and voice embeddings
# Usage: curl -sSf https://raw.githubusercontent.com/second-state/voxtral_tts_rs/main/install.sh | bash

set -e

REPO="second-state/voxtral_tts_rs"
INSTALL_DIR="./voxtral_tts_rs"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[info]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ok]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
err()   { echo -e "${RED}[error]${NC} $*" >&2; }

# ---------------------------------------------------------------------------
# 1. Detect platform
# ---------------------------------------------------------------------------
detect_platform() {
    case "$(uname -s)" in
        Linux*)  OS="linux" ;;
        Darwin*) OS="darwin" ;;
        *)
            err "Unsupported operating system: $(uname -s)"
            exit 1
            ;;
    esac

    case "$(uname -m)" in
        x86_64|amd64)    ARCH="x86_64" ;;
        aarch64|arm64)   ARCH="aarch64" ;;
        *)
            err "Unsupported architecture: $(uname -m)"
            exit 1
            ;;
    esac

    # CUDA detection (Linux x86_64 only)
    USE_CUDA=""
    if [ "$OS" = "linux" ] && [ "$ARCH" = "x86_64" ]; then
        if command -v nvidia-smi &>/dev/null && nvidia-smi --query-gpu=driver_version --format=csv,noheader &>/dev/null; then
            CUDA_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
            info "NVIDIA GPU detected (driver ${CUDA_DRIVER})."
            echo "  1) CUDA 12.8 (Recommended)"
            echo "  2) CPU only"
            printf "Select variant [1]: "
            read -r variant </dev/tty
            variant="${variant:-1}"
            if [ "$variant" = "1" ]; then
                USE_CUDA="1"
            fi
        fi
    fi

    info "Platform: ${OS} ${ARCH}${USE_CUDA:+ (CUDA)}"
}

# ---------------------------------------------------------------------------
# 2. Resolve release asset name
# ---------------------------------------------------------------------------
resolve_asset() {
    case "${OS}-${ARCH}" in
        darwin-aarch64)  ASSET_NAME="voxtral-tts-macos-aarch64" ;;
        linux-x86_64)
            if [ -n "$USE_CUDA" ]; then
                ASSET_NAME="voxtral-tts-linux-x86_64-cuda"
            else
                ASSET_NAME="voxtral-tts-linux-x86_64"
            fi
            ;;
        linux-aarch64)   ASSET_NAME="voxtral-tts-linux-aarch64" ;;
        *)
            err "Unsupported platform: ${OS}-${ARCH}"
            exit 1
            ;;
    esac
    info "Release asset: ${ASSET_NAME}"
}

# ---------------------------------------------------------------------------
# 3. Download & extract release
# ---------------------------------------------------------------------------
download_release() {
    local zip_name="${ASSET_NAME}.zip"
    local url="https://github.com/${REPO}/releases/latest/download/${zip_name}"

    info "Downloading release..."
    mkdir -p "${INSTALL_DIR}"

    local temp_dir
    temp_dir=$(mktemp -d)

    curl -fSL -o "${temp_dir}/${zip_name}" "$url"
    info "Extracting release..."
    unzip -q "${temp_dir}/${zip_name}" -d "${temp_dir}"

    # Copy everything from the extracted asset directory into INSTALL_DIR
    cp -r "${temp_dir}/${ASSET_NAME}/"* "${INSTALL_DIR}/"
    chmod +x "${INSTALL_DIR}/voxtral-tts" "${INSTALL_DIR}/voxtral-tts-server"

    rm -rf "$temp_dir"
    ok "Binaries installed to ${INSTALL_DIR}/"
}

# ---------------------------------------------------------------------------
# 4. Download CUDA libtorch (Linux x86_64 CUDA only)
# ---------------------------------------------------------------------------
download_cuda_libtorch() {
    if [ -z "$USE_CUDA" ]; then
        return
    fi

    info "Downloading CUDA libtorch (this may take a while)..."
    local url="https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcu128.zip"
    local temp_dir
    temp_dir=$(mktemp -d)

    curl -fSL -o "${temp_dir}/libtorch.zip" "$url"
    info "Extracting libtorch..."
    unzip -q "${temp_dir}/libtorch.zip" -d "${temp_dir}"

    rm -rf "${INSTALL_DIR}/libtorch"
    mv "${temp_dir}/libtorch" "${INSTALL_DIR}/libtorch"

    rm -rf "$temp_dir"
    ok "CUDA libtorch installed to ${INSTALL_DIR}/libtorch/"
}

# ---------------------------------------------------------------------------
# 5. Download model (curl only, no Python)
# ---------------------------------------------------------------------------
download_model() {
    local model_dir="${INSTALL_DIR}/models/voxtral-4b-tts"

    if [ -d "$model_dir" ] && [ -f "${model_dir}/consolidated.safetensors" ]; then
        ok "Model already downloaded, skipping."
        return
    fi

    mkdir -p "${model_dir}/voice_embedding"
    local hf_url="https://huggingface.co/mistralai/Voxtral-4B-TTS-2603/resolve/main"

    info "Downloading model weights (8GB, this may take a while)..."
    curl -fSL -o "${model_dir}/consolidated.safetensors" "${hf_url}/consolidated.safetensors"

    info "Downloading config and tokenizer..."
    curl -fSL -o "${model_dir}/params.json" "${hf_url}/params.json"
    curl -fSL -o "${model_dir}/tekken.json" "${hf_url}/tekken.json"

    info "Downloading voice embeddings (20 voices)..."
    for v in casual_female casual_male cheerful_female neutral_female neutral_male \
             pt_male pt_female nl_male nl_female it_male it_female \
             fr_male fr_female es_male es_female de_male de_female \
             ar_male hi_male hi_female; do
        curl -fSL -o "${model_dir}/voice_embedding/${v}.pt" \
            "${hf_url}/voice_embedding/${v}.pt"
    done

    ok "Model downloaded to ${model_dir}/"
}

# ---------------------------------------------------------------------------
# 6. Done — print sample commands
# ---------------------------------------------------------------------------
print_usage() {
    echo ""
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN} Installation complete!${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo ""

    echo "Text-to-Speech (preset voice):"
    echo ""
    echo "  cd ${INSTALL_DIR}"
    echo "  ./voxtral-tts models/voxtral-4b-tts \\"
    echo "    --text \"Hello! This is a test of Voxtral TTS.\" \\"
    echo "    --voice neutral_female --output output.wav"
    echo ""

    echo "Voice Cloning (from reference audio):"
    echo ""
    echo "  cd ${INSTALL_DIR}"
    echo "  ./voxtral-tts models/voxtral-4b-tts \\"
    echo "    --text \"Hello from a cloned voice!\" \\"
    echo "    --reference-audio reference.wav --output cloned.wav"
    echo ""

    echo "API Server:"
    echo ""
    echo "  cd ${INSTALL_DIR}"
    echo "  ./voxtral-tts-server models/voxtral-4b-tts --port 8080"
    echo ""
    echo "  curl -X POST http://localhost:8080/v1/audio/speech \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"input\":\"Hello world\",\"voice\":\"alloy\",\"model\":\"voxtral-4b-tts\"}' \\"
    echo "    -o output.wav"
    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    echo ""
    info "Voxtral TTS Rust Installer"
    echo ""

    detect_platform
    resolve_asset
    download_release
    download_cuda_libtorch
    download_model
    print_usage
}

main "$@"
