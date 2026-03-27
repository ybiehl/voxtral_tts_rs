# Voxtral TTS Rust

Rust port of [Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) -- Mistral AI's 4B-parameter text-to-speech model. Runs on macOS (Apple Silicon via MLX) and Linux (CUDA via libtorch). No Python required.

## Features

- **Dual backend**: libtorch (Linux/CUDA) and MLX (macOS/Metal)
- **CLI tool**: Generate speech from text with 20 preset voices
- **API server**: OpenAI-compatible `/v1/audio/speech` endpoint (Axum)
- **Pure Rust inference**: Tekken BPE tokenizer, safetensors loader, full pipeline
- **No Python**: Model downloaded with curl, weights loaded via `safetensors` crate

## Prerequisites

| Platform | Requirements |
|----------|-------------|
| macOS (Apple Silicon) | Xcode Command Line Tools, CMake, Rust 1.75+ |
| Linux (CPU) | GCC/Clang, Rust 1.75+ |
| Linux (CUDA) | NVIDIA driver 535+, CUDA 12.8, Rust 1.75+ |

Disk: ~10GB for model weights + ~2GB for libtorch (Linux only).

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/example/voxtral_tts_rs.git
cd voxtral_tts_rs
```

### 2. Download the model (curl only, no Python)

```bash
bash scripts/download_model.sh
```

This downloads to `models/voxtral-4b-tts/`:

| File | Size | Description |
|------|------|-------------|
| `consolidated.safetensors` | 8 GB | Model weights (BF16) |
| `params.json` | 4 KB | Model configuration |
| `tekken.json` | 15 MB | Tokenizer vocabulary |
| `voice_embedding/*.pt` | ~50 MB | 20 preset voice embeddings |

### 3. Build

**macOS (MLX backend -- recommended for Apple Silicon):**

```bash
git submodule update --init --recursive
cargo build --release --no-default-features --features mlx
```

**Linux (libtorch backend):**

```bash
# Download libtorch (pick one)
bash scripts/download_libtorch.sh cpu      # CPU only
bash scripts/download_libtorch.sh cu128    # CUDA 12.8

# Set environment
export LIBTORCH=$(pwd)/libtorch
export LIBTORCH_BYPASS_VERSION_CHECK=1
export LD_LIBRARY_PATH=${LIBTORCH}/lib:${LD_LIBRARY_PATH}

cargo build --release
```

### 4. Convert voice embeddings (MLX only)

The voice embeddings ship as PyTorch `.pt` files. The MLX backend needs `.safetensors` format. Convert them once:

```bash
pip install torch safetensors   # one-time dependency
python3 -c "
import torch, os
from safetensors.torch import save_file
d = 'models/voxtral-4b-tts/voice_embedding'
for f in os.listdir(d):
    if f.endswith('.pt'):
        t = torch.load(os.path.join(d, f), map_location='cpu', weights_only=True)
        save_file({'embedding': t}, os.path.join(d, f.replace('.pt', '.safetensors')))
        print(f'Converted {f}')
"
```

## Usage

### CLI

```bash
# Generate speech with a preset voice
./target/release/voxtral-tts models/voxtral-4b-tts \
    --text "Hello, this is Voxtral TTS!" \
    --voice neutral_female \
    --output output.wav

# Use a different voice
./target/release/voxtral-tts models/voxtral-4b-tts \
    --text "Bonjour le monde!" \
    --voice fr_female \
    --output bonjour.wav

# List all available voices
./target/release/voxtral-tts models/voxtral-4b-tts --list-voices --text ""
```

**CLI options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--text` | (required) | Text to synthesize |
| `--voice` | `neutral_female` | Voice name or OpenAI alias |
| `--output` | `output.wav` | Output WAV file path |
| `--temperature` | `0.7` | Sampling temperature |
| `--max-tokens` | `4096` | Maximum generation tokens |
| `--list-voices` | | Print available voices and exit |

### API Server

```bash
./target/release/voxtral-tts-server models/voxtral-4b-tts --port 8080
```

**Generate speech (OpenAI-compatible):**

```bash
curl -X POST http://localhost:8080/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"input":"Hello world","voice":"alloy","model":"voxtral-4b-tts"}' \
    -o output.wav
```

**List models:**

```bash
curl http://localhost:8080/v1/models
```

**Health check:**

```bash
curl http://localhost:8080/health
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Returns `{"status":"ok"}` |
| `/v1/models` | GET | List available models |
| `/v1/audio/speech` | POST | Generate speech |

**POST /v1/audio/speech** request body:

```json
{
    "model": "voxtral-4b-tts",
    "input": "Text to synthesize",
    "voice": "neutral_female",
    "response_format": "wav",
    "speed": 1.0,
    "stream": false
}
```

## Voices

### 20 Preset Voices

| Voice | Language | Gender |
|-------|----------|--------|
| `casual_female`, `casual_male` | English | F, M |
| `cheerful_female` | English | F |
| `neutral_female`, `neutral_male` | English | F, M |
| `fr_male`, `fr_female` | French | M, F |
| `es_male`, `es_female` | Spanish | M, F |
| `de_male`, `de_female` | German | M, F |
| `pt_male`, `pt_female` | Portuguese | M, F |
| `it_male`, `it_female` | Italian | M, F |
| `nl_male`, `nl_female` | Dutch | M, F |
| `ar_male` | Arabic | M |
| `hi_male`, `hi_female` | Hindi | M, F |

### OpenAI Voice Aliases

| Alias | Maps to |
|-------|---------|
| `alloy` | `neutral_female` |
| `echo` | `casual_male` |
| `fable` | `cheerful_female` |
| `onyx` | `neutral_male` |
| `nova` | `casual_female` |
| `shimmer` | `fr_female` |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `RUST_LOG` | Log verbosity: `error`, `warn`, `info` (default), `debug`, `trace` |
| `LIBTORCH` | Path to libtorch directory (Linux/tch backend only) |
| `LIBTORCH_BYPASS_VERSION_CHECK` | Set to `1` to skip libtorch version check |
| `LD_LIBRARY_PATH` | Include `$LIBTORCH/lib` (Linux only) |

## Architecture Overview

The model has three components totalling 4B parameters:

```
Text ──> Tekken Tokenizer ──> Token IDs
                                  │
                                  v
Voice Embedding ──> Backbone Decoder (3.4B, 26 layers) ──> Hidden States
                                                               │
                                                               v
                    Flow-Matching Transformer (390M) ──> 37 Audio Codes/Frame
                                                               │
                                                               v
                    Voxtral Codec Decoder (300M) ──> 24kHz Mono Waveform
```

| Component | Parameters | Architecture |
|-----------|-----------|--------------|
| Backbone Decoder | 3.4B | 26-layer Mistral transformer, dim=3072, 32 heads (8 KV), SwiGLU, RoPE |
| Flow-Matching Transformer | 390M | 3-layer bidirectional transformer, Euler ODE (7 steps), CFG |
| Voxtral Codec Decoder | 300M | 4 conv+transformer blocks, strides [1,2,2,2], 240-channel output |

## Project Structure

```
src/
├── lib.rs              # Library root, feature gates, constants
├── tensor.rs           # Unified tensor abstraction (tch / MLX)
├── config.rs           # params.json config parsing
├── tokenizer.rs        # Pure Rust Tekken BPE tokenizer
├── audio.rs            # WAV I/O, resampling, PCM encoding
├── voice.rs            # Voice embedding loading
├── inference.rs        # High-level TTS pipeline
├── error.rs            # Error types
├── model/
│   ├── layers.rs       # RMSNorm, Linear, GQA Attention, SwiGLU MLP, RoPE
│   ├── backbone.rs     # 26-layer Mistral decoder (3.4B)
│   ├── flow_matching.rs # Flow-matching acoustic transformer (390M)
│   ├── codec.rs        # Voxtral neural audio codec decoder (300M)
│   ├── kv_cache.rs     # KV cache for autoregressive generation
│   ├── sampling.rs     # Top-k, top-p, temperature sampling
│   └── weights.rs      # Safetensors weight loading and partitioning
├── backend/
│   └── mlx/            # Apple MLX C FFI bindings
│       ├── ffi.rs      # Raw C function declarations
│       ├── array.rs    # MlxArray RAII wrapper
│       ├── ops.rs      # Safe operation wrappers
│       ├── stream.rs   # Device/stream initialization
│       ├── io.rs       # Safetensors loading
│       └── signal.rs   # Conv/STFT operations
├── bin/
│   ├── tts.rs          # CLI binary
│   └── tts_server/     # Axum API server
│       ├── main.rs
│       ├── state.rs
│       └── routes/
│           ├── health.rs
│           ├── models.rs
│           └── speech.rs
scripts/
├── download_model.sh       # curl-only model download
└── download_libtorch.sh    # curl-only libtorch download
```

## License

Apache-2.0

The Voxtral model weights are licensed under [Mistral AI Non-Production License](https://mistral.ai/licenses/MNPL-0.1.md).
