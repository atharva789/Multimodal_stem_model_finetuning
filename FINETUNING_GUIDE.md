# Fine-Tuning DeepSeek-R1-Distill-Qwen-1.5B - Implementation Guide

## Overview

This guide explains the fine-tuning implementation that has been added to `LLM.ipynb`. The notebook now includes complete functionality to fine-tune DeepSeek-R1-Distill-Qwen-1.5B on a mixed math/science dataset using QLoRA (4-bit quantization + LoRA adapters).

## What Was Added

**14 new cells** have been inserted after cell-6 (the dataset mixing code) in the notebook:

1. **Markdown**: Fine-Tuning Setup introduction
2. **Code**: Install dependencies (transformers, peft, trl, accelerate, scipy, tensorboard)
3. **Code**: Format MathCodeInstruct and SciBench datasets to chat format with `<think>` tags
4. **Code**: Create 90/10 train/eval split (~22.5k train, ~2.5k eval)
5. **Code**: Load model for fine-tuning with QLoRA (4-bit quantization)
6. **Code**: Configure LoRA adapters (r=16, alpha=32, all-linear targets)
7. **Code**: Set training hyperparameters (3 epochs, batch size 2, grad accum 8)
8. **Code**: Initialize SFTTrainer from TRL library
9. **Code**: Start training with trainer.train()
10. **Code**: Save fine-tuned LoRA adapters (~100MB)
11. **Markdown**: Testing section introduction
12. **Code**: Test fine-tuned model on sample problems
13. **Markdown**: Optional merging section
14. **Code**: Optional - merge LoRA with base model for deployment

## Execution Order

Run the notebook cells in this sequence:

### Phase 1: Initial Setup (Existing Cells 0-6)
- Cell 0: Install bitsandbytes
- Cell 1-4: Load base model and test inference
- Cell 5: Dataset planning markdown
- Cell 6: Load and mix datasets (MathCodeInstruct + SciBench)

### Phase 2: Fine-Tuning Preparation (New Cells 7-11)
- Cell 7: Markdown intro
- Cell 8: Install fine-tuning dependencies
- Cell 9: Format datasets to chat format with reasoning tags
- Cell 10: Split into train/eval sets
- Cell 11: Load model with QLoRA configuration

### Phase 3: Fine-Tuning Execution (New Cells 12-16)
- Cell 12: Configure LoRA adapters
- Cell 13: Set training hyperparameters
- Cell 14: Initialize SFTTrainer
- Cell 15: **START TRAINING** (takes several hours)
- Cell 16: Save LoRA adapters

### Phase 4: Testing & Deployment (New Cells 17-20)
- Cell 17-18: Test fine-tuned model
- Cell 19-20: Optional - merge adapters for deployment

## Dataset Formatting

The implementation formats both datasets into DeepSeek-R1's expected chat format with `<think>` reasoning tags:

### MathCodeInstruct Format
```python
{
  "messages": [
    {"role": "user", "content": "Solve: 2x + 5 = 13"},
    {"role": "assistant", "content": "<think>\n[step-by-step solution]\n</think>\n\nThe solution is x = 4"}
  ]
}
```

### SciBench Format
```python
{
  "messages": [
    {"role": "user", "content": "Calculate the force..."},
    {"role": "assistant", "content": "<think>\n[detailed solution]\n</think>\n\nFinal Answer: 50 N"}
  ]
}
```

## Training Configuration

### Memory Optimization (Fits in 32GB VRAM)
- **4-bit NF4 quantization**: Reduces model size ~75%
- **LoRA adapters**: Only ~100MB trainable parameters
- **Gradient checkpointing**: Saves ~40% memory
- **8-bit paged AdamW**: Saves ~6GB VRAM
- **Expected VRAM usage**: ~8-9GB (comfortable headroom)

### Hyperparameters
- **Learning rate**: 2e-4 (QLoRA standard)
- **Batch size**: 2 per device × 8 gradient accumulation = 16 effective
- **Epochs**: 3
- **Max sequence length**: 2048 tokens
- **LoRA rank**: 16 (higher for reasoning tasks)
- **LoRA alpha**: 32 (2× rank scaling)

### Disk Management (16GB Constraint)
- **save_total_limit=3**: Keeps only 3 checkpoints
- **Checkpoint size**: ~100MB each (300MB total)
- **Final model**: ~100MB LoRA adapters
- **Total disk usage**: <500MB during training

## Monitoring Training

### TensorBoard (Recommended)
```python
%load_ext tensorboard
%tensorboard --logdir ./deepseek-r1-math-science/logs
```

**Key metrics to watch:**
- **Training loss**: Should decrease smoothly
- **Eval loss**: Should track training loss (gap <0.5 = good)
- **Learning rate**: Follows cosine decay
- **Gradient norms**: Should be stable

### Success Indicators
✅ Training loss decreases steadily
✅ Eval loss < 1.0 (perplexity < 2.72)
✅ Model generates `<think>` formatted responses
✅ Improved math/science reasoning vs base model
✅ No OOM errors
✅ Disk usage stays under 500MB

## Output Files

After training completes:

```
./deepseek-r1-math-science/           # Training directory
├── checkpoint-250/                    # Checkpoint 1
├── checkpoint-500/                    # Checkpoint 2
├── checkpoint-750/                    # Checkpoint 3 (or best)
├── logs/                              # TensorBoard logs
└── runs/                              # Training runs

./deepseek-r1-math-science-final/     # Final LoRA adapters (~100MB)
├── adapter_config.json
├── adapter_model.safetensors
└── tokenizer files

./deepseek-r1-math-science-merged/    # Optional merged model (~3GB)
├── model.safetensors
├── config.json
└── tokenizer files
```

## Testing the Fine-Tuned Model

Cell 18 includes test prompts:
```python
test_prompts = [
    "Solve for x: 3x + 5 = 20",
    "What is the derivative of f(x) = x^3 + 2x^2 - 5x + 3?",
    "Calculate the force required to accelerate a 10 kg object at 5 m/s^2.",
]
```

**Expected improvements:**
- Step-by-step reasoning in `<think>` tags
- Proper use of mathematical notation
- Code/tool usage where appropriate (from MathCodeInstruct)
- Structured problem-solving approach

## Deployment Options

### Option 1: Use LoRA Adapters (Recommended for Testing)
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model = PeftModel.from_pretrained(base_model, "./deepseek-r1-math-science-final")
```
- **Size**: ~1.5GB base + ~100MB adapters
- **Pros**: Smallest footprint, easy to swap adapters
- **Cons**: Slightly slower inference

### Option 2: Merge Adapters (For Production)
```python
# Run cell 20 to merge
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./deepseek-r1-math-science-merged")
```
- **Size**: ~3GB merged model
- **Pros**: Faster inference, standalone model
- **Cons**: Larger file size

### Option 3: Convert to GGUF (For On-Device)
After merging, use llama.cpp:
```bash
python convert.py ./deepseek-r1-math-science-merged --outfile model.gguf
./quantize model.gguf model-q8_0.gguf q8_0
```
- **Q8_0**: ~1.5GB (high quality)
- **Q4_K_M**: ~900MB (balanced)
- **Q4_0**: ~800MB (smaller, lower quality)

## Troubleshooting

### Out of Memory (OOM) Error
**Symptoms**: CUDA OOM during training

**Solutions**:
1. Reduce batch size: `per_device_train_batch_size=1`
2. Increase gradient accumulation: `gradient_accumulation_steps=16`
3. Reduce max sequence length: `max_seq_length=1024`
4. Enable more aggressive gradient checkpointing

### Disk Space Full
**Symptoms**: "No space left on device"

**Solutions**:
1. Reduce `save_total_limit` from 3 to 2
2. Delete intermediate checkpoints manually:
   ```bash
   rm -rf ./deepseek-r1-math-science/checkpoint-250
   ```
3. Monitor disk usage:
   ```bash
   du -sh ./deepseek-r1-math-science/
   ```

### Training Loss Not Decreasing
**Symptoms**: Loss plateaus or increases

**Solutions**:
1. Check learning rate isn't too high/low
2. Verify dataset formatting is correct
3. Check for duplicate or corrupted examples
4. Try different optimizer: `optim="adamw_torch"`

### Overfitting (Eval Loss Increases)
**Symptoms**: Training loss decreases but eval loss increases

**Solutions**:
1. Reduce epochs from 3 to 2
2. Increase `lora_dropout` from 0.05 to 0.1
3. Add more evaluation data
4. Enable early stopping:
   ```python
   training_args.early_stopping_patience = 3
   ```

### Slow Training Speed
**Symptoms**: Training takes >12 hours

**Solutions**:
1. Try Flash Attention 2 (if supported):
   ```python
   attn_implementation="flash_attention_2"
   ```
2. Reduce evaluation frequency: `eval_steps=500`
3. Disable some logging: `logging_steps=50`
4. Check GPU utilization: `nvidia-smi`

## Expected Timeline

On a single GPU (e.g., A100, V100, or RTX 4090):

- **Data loading**: ~2-3 minutes
- **Model loading**: ~1-2 minutes
- **Training**: ~4-8 hours (depends on GPU)
  - ~750 steps per epoch
  - ~2,250 total steps
  - ~10-15 seconds per step
- **Saving**: ~1 minute

**Total**: 4-8 hours for complete training run

## Next Steps After Training

1. **Test on diverse problems**: Math, physics, chemistry, etc.
2. **Compare to base model**: Side-by-side evaluation
3. **Analyze failure cases**: Identify weaknesses
4. **Iterate**:
   - Adjust LoRA rank/alpha if needed
   - Add more data in weak areas
   - Try different hyperparameters
5. **Deploy**: Choose deployment option based on use case
6. **Monitor**: Track performance on real-world tasks

## Additional Resources

- **TRL Documentation**: https://huggingface.co/docs/trl
- **PEFT Documentation**: https://huggingface.co/docs/peft
- **DeepSeek-R1 Model Card**: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
- **QLoRA Paper**: https://arxiv.org/abs/2305.14314
- **llama.cpp GGUF**: https://github.com/ggerganov/llama.cpp

## Contact & Support

For issues specific to:
- **This implementation**: Check troubleshooting section above
- **TRL/PEFT libraries**: HuggingFace GitHub issues
- **DeepSeek model**: DeepSeek GitHub issues
- **Hardware/VRAM**: Vast.ai support

---

**Last Updated**: 2026-02-09
**Implementation Version**: 1.0
**Compatible with**: transformers>=4.50.0, peft>=0.14.0, trl>=0.12.0
