# Installation Guide

## Quick Install

### Option 1: Install All Dependencies
```bash
pip install -r requirements.txt
```

### Option 2: Install Minimal (Essential Only)
```bash
# Core fine-tuning dependencies only
pip install -U \
    torch>=2.1.0 \
    transformers>=4.50.0 \
    peft>=0.14.0 \
    trl>=0.12.0 \
    accelerate>=1.2.0 \
    bitsandbytes>=0.46.1 \
    datasets>=2.14.0 \
    scipy \
    tensorboard
```

### Option 3: Install in Colab/Jupyter Notebook
```python
# Run this in a notebook cell
!pip install -q -U \
    transformers>=4.50.0 \
    peft>=0.14.0 \
    trl>=0.12.0 \
    accelerate>=1.2.0 \
    bitsandbytes>=0.46.1 \
    scipy \
    tensorboard
```

## Environment Setup

### 1. Create Virtual Environment (Recommended)
```bash
# Using venv
python3 -m venv llm-env
source llm-env/bin/activate  # On Windows: llm-env\Scripts\activate

# Using conda
conda create -n llm-env python=3.10
conda activate llm-env
```

### 2. Install CUDA Toolkit (If Not Installed)
```bash
# Check CUDA version
nvidia-smi

# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```python
import torch
import transformers
import peft
import trl

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Transformers: {transformers.__version__}")
print(f"PEFT: {peft.__version__}")
print(f"TRL: {trl.__version__}")
```

## Troubleshooting

### Issue: bitsandbytes not compatible with Windows
```bash
# Use pre-built Windows version
pip install bitsandbytes-windows
```

### Issue: CUDA out of memory during installation
```bash
# Clear cache
pip cache purge
# Install one by one
pip install torch
pip install transformers
# ... continue
```

### Issue: Version conflicts
```bash
# Create fresh environment
conda create -n llm-fresh python=3.10
conda activate llm-fresh
pip install -r requirements.txt
```

## Dependencies Explained

| Package | Purpose | Required |
|---------|---------|----------|
| torch | Deep learning framework | ✅ Yes |
| transformers | HuggingFace model library | ✅ Yes |
| peft | Parameter-efficient fine-tuning (LoRA) | ✅ Yes |
| trl | Transformer Reinforcement Learning (SFTTrainer) | ✅ Yes |
| accelerate | Multi-GPU training support | ✅ Yes |
| bitsandbytes | 4-bit quantization | ✅ Yes |
| datasets | Dataset loading and processing | ✅ Yes |
| scipy | Scientific computing | ✅ Yes |
| tensorboard | Training monitoring | ✅ Yes |
| wandb | Experiment tracking | ❌ Optional |
| ipython/jupyter | Notebook support | ❌ Optional |

## System Requirements

- **Python**: 3.9 - 3.11 (3.10 recommended)
- **CUDA**: 11.8+ or 12.1+ (for GPU training)
- **GPU**: 8GB+ VRAM minimum (32GB recommended)
- **RAM**: 16GB+ system memory
- **Disk**: 20GB+ free space

## Post-Installation

After installation, you're ready to:
1. Open `LLM.ipynb` in Jupyter
2. Run the fine-tuning cells
3. Monitor training with TensorBoard

## Need Help?

- Check package versions: `pip list | grep -E "torch|transformers|peft|trl"`
- Update packages: `pip install -U <package-name>`
- See full documentation in `FINETUNING_GUIDE.md`
