#!/bin/bash
# Download libtorch (no Python required)
# Usage: bash scripts/download_libtorch.sh [cpu|cu128|macos]

set -e

VARIANT="${1:-cpu}"
LIBTORCH_VERSION="2.7.1"

case "${VARIANT}" in
    cpu)
        URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip"
        ARCHIVE="libtorch.zip"
        ;;
    cu128)
        URL="https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcu128.zip"
        ARCHIVE="libtorch.zip"
        ;;
    macos)
        URL="https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${LIBTORCH_VERSION}.zip"
        ARCHIVE="libtorch.zip"
        ;;
    *)
        echo "Usage: $0 [cpu|cu128|macos]"
        echo "  cpu   - Linux x86_64 CPU only (default)"
        echo "  cu128 - Linux x86_64 CUDA 12.8"
        echo "  macos - macOS ARM64 (Apple Silicon)"
        exit 1
        ;;
esac

echo "Downloading libtorch ${LIBTORCH_VERSION} (${VARIANT})..."
curl -Lo "${ARCHIVE}" "${URL}"

echo "Extracting..."
if [[ "${ARCHIVE}" == *.zip ]]; then
    unzip -q "${ARCHIVE}"
else
    tar xzf "${ARCHIVE}"
fi
rm "${ARCHIVE}"

echo ""
echo "libtorch downloaded to: $(pwd)/libtorch"
echo ""
echo "Set the environment variables before building:"
echo "  export LIBTORCH=$(pwd)/libtorch"
echo "  export LIBTORCH_BYPASS_VERSION_CHECK=1"

if [ "$(uname -s)" = "Linux" ]; then
    echo "  export LD_LIBRARY_PATH=\${LIBTORCH}/lib:\${LD_LIBRARY_PATH}"
fi

echo ""
echo "Then build with:"
echo "  cargo build --release"
