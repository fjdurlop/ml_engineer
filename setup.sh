#!/bin/bash

# ======================================================
# Environment Setup Script for ML Engineer Assessment
# Using uv as the package manager
# ======================================================

set -e

echo "🔍 Checking system requirements..."

# Check NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA GPU not detected (nvidia-smi missing)."
    exit 1
else
    echo "✅ NVIDIA GPU detected."
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

# Check GPU memory (at least 4GB)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
if [ "$GPU_MEM" -lt 4000 ]; then
    echo "❌ GPU memory is less than 4GB ($GPU_MEM MB)."
    exit 1
else
    echo "✅ GPU memory sufficient: $GPU_MEM MB"
fi

# Check CUDA toolkit
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA toolkit (nvcc) not found. Please install CUDA."
    exit 1
else
    echo "✅ CUDA toolkit detected: $(nvcc --version | grep release)"
fi

echo "🚀 Setting up Python environment with uv..."

# Create venv if not exists
if [ ! -d ".venv" ]; then
    uv venv
fi

# Activate venv
source .venv/bin/activate

# Sync dependencies
uv pip install torch torchaudio
uv pip install numpy matplotlib seaborn
uv pip install python-Levenshtein scipy GPUtil psutil
uv pip install jupyter notebook  # Optional for analysis

echo "🔍 Verifying PyTorch GPU access..."
python - <<'EOF'
import torch
if torch.cuda.is_available():
    print("✅ PyTorch CUDA available")
    print("   GPU:", torch.cuda.get_device_name(0))
    print("   Memory:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "GB")
else:
    print("❌ PyTorch cannot access CUDA. Please check driver/toolkit compatibility.")
    exit(1)
EOF

echo "✅ Environment setup complete!"
