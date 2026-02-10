# Fine-Tuning Quick Start

## 🚀 Getting Started

### 1. Open the Notebook
```bash
jupyter notebook LLM.ipynb
```

### 2. Run Cells in Order
Execute cells **0-20** sequentially:
- Cells 0-6: Setup + load datasets
- Cells 7-11: Prepare for fine-tuning
- Cells 12-16: Configure and train
- Cells 17-20: Test and deploy

### 3. Monitor Training
```python
%load_ext tensorboard
%tensorboard --logdir ./deepseek-r1-math-science/logs
```

## 📊 Key Metrics

| Metric | Target Value | What It Means |
|--------|--------------|---------------|
| Training Loss | Decreasing | Model is learning |
| Eval Loss | < 1.0 | Good generalization |
| VRAM Usage | ~8-9GB | Well within 32GB limit |
| Disk Usage | < 500MB | Checkpoint management working |
| Training Time | 4-8 hours | Normal for this dataset size |

## 💾 What Gets Created

```
deepseek-r1-math-science/          # Training checkpoints
deepseek-r1-math-science-final/    # LoRA adapters (~100MB)
deepseek-r1-math-science-merged/   # Optional merged model (~3GB)
```

## ⚡ Quick Commands

### Check Disk Space
```bash
du -sh ./deepseek-r1-math-science*
```

### Monitor GPU
```bash
watch -n 1 nvidia-smi
```

### Clean Up Checkpoints
```bash
rm -rf ./deepseek-r1-math-science/checkpoint-*
```

## 🔧 Common Issues

### Out of Memory?
- Set `per_device_train_batch_size=1` in cell 13
- Set `gradient_accumulation_steps=16` in cell 13

### Disk Full?
- Set `save_total_limit=2` in cell 13
- Delete old checkpoints manually

### Training Too Slow?
- Reduce `max_seq_length=1024` in cell 14
- Increase `eval_steps=500` in cell 13

## 📈 Expected Results

After training, the model should:
✅ Generate `<think>` formatted reasoning
✅ Solve math problems step-by-step
✅ Answer science questions with proper methodology
✅ Use code/tools when appropriate (from MathCodeInstruct)

## 🎯 Next Steps

1. **Test**: Run cell 18 to test on sample problems
2. **Compare**: Test same prompts on base model (cells 3-4)
3. **Deploy**:
   - Keep LoRA adapters for testing (cells 16)
   - Merge for production (cell 20)
   - Convert to GGUF for on-device (external tool)

## 📚 Full Documentation

See `FINETUNING_GUIDE.md` for complete details on:
- Dataset formatting
- Hyperparameter tuning
- Troubleshooting
- Deployment options
- Advanced configurations

---

**Ready to start?** Run cell 0 and keep going! 🎉
