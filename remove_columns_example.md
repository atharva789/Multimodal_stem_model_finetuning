# Understanding `remove_columns` in Dataset Mapping

## The Issue

When you transform datasets with `.map()`, you need to understand what happens to the original columns.

### Example Dataset: SciBench

**Original structure:**
```python
{
  'problem_text': 'Calculate the force...',
  'solution': 'Using F=ma...',
  'answer_latex': '50',
  'unit': 'N',
  'subject': 'physics',
  'difficulty': 'medium',
  'source_id': 'scibench_001'
}
```

### Scenario 1: WITHOUT `remove_columns`

```python
def format_scibench(example):
    return {
        "messages": [
            {"role": "user", "content": example['problem_text']},
            {"role": "assistant", "content": f"<think>{example['solution']}</think>\n\nAnswer: {example['answer_latex']}"}
        ]
    }

# Map WITHOUT removing columns
formatted = dataset.map(format_scibench)
```

**Result:** Dataset has 8 columns!
```python
{
  'problem_text': 'Calculate the force...',     # ← OLD (unused)
  'solution': 'Using F=ma...',                  # ← OLD (unused)
  'answer_latex': '50',                         # ← OLD (unused)
  'unit': 'N',                                  # ← OLD (unused)
  'subject': 'physics',                         # ← OLD (unused)
  'difficulty': 'medium',                       # ← OLD (unused)
  'source_id': 'scibench_001',                  # ← OLD (unused)
  'messages': [...]                             # ← NEW (what we want)
}
```

**Problems:**
- ❌ 7 extra columns taking up memory
- ❌ Confusing which data is actually used
- ❌ SFTTrainer might get confused
- ❌ Wastes disk space when saving

### Scenario 2: WITH `remove_columns`

```python
formatted = dataset.map(
    format_scibench,
    remove_columns=dataset.column_names  # ← Remove ALL old columns
)
```

**Result:** Clean dataset with 1 column!
```python
{
  'messages': [  # ← ONLY what we need
    {"role": "user", "content": "Calculate the force..."},
    {"role": "assistant", "content": "<think>Using F=ma...</think>\n\nAnswer: 50 N"}
  ]
}
```

**Benefits:**
- ✅ Clean, minimal structure
- ✅ Only training-relevant data
- ✅ ~80% less memory usage
- ✅ No confusion about which fields matter
- ✅ SFTTrainer gets exactly what it expects

## Real Impact on Memory

For our 25k example dataset:

### Without `remove_columns`:
```
Original columns: ~8 fields × 25k examples = ~200k fields
New columns:      ~1 field  × 25k examples = ~25k fields
Total:            ~225k fields in memory
Memory usage:     ~500-800 MB
```

### With `remove_columns`:
```
Original columns: REMOVED
New columns:      ~1 field × 25k examples = ~25k fields
Total:            ~25k fields in memory
Memory usage:     ~100-150 MB
```

**Savings: ~70-80% less memory!**

## What `dataset.column_names` Returns

```python
# MathCodeInstruct
math_subset.column_names
# Output: ['messages', 'id', 'source', 'metadata', ...]

# SciBench
scibench_subset.column_names
# Output: ['problem_text', 'solution', 'answer_latex', 'unit', 'subject', ...]
```

When you pass this to `remove_columns`, it removes ALL those original columns after mapping.

## Alternative: Selective Removal

You can also remove specific columns:

```python
# Keep only 'id' for tracking
formatted = dataset.map(
    format_fn,
    remove_columns=['problem_text', 'solution', 'answer_latex', 'unit']
)
# Result has both 'messages' AND 'id'
```

But for training, we usually want **only** the `messages` field, so we remove everything.

## Why It Matters for SFTTrainer

SFTTrainer expects:
```python
dataset_text_field="messages"
```

It looks for a field named "messages" with chat-formatted data. If your dataset has extra columns, they:
1. Take up memory during training
2. Might cause unexpected behavior
3. Slow down data loading
4. Waste disk space when checkpointing

**Best practice:** Keep only what you need!

## Summary

```python
# ❌ BAD: Keeps old columns (wastes memory)
formatted = dataset.map(format_fn)

# ✅ GOOD: Removes old columns (clean & efficient)
formatted = dataset.map(
    format_fn,
    remove_columns=dataset.column_names
)
```

The `remove_columns` parameter is essential for:
- Memory efficiency
- Clean data structure
- SFTTrainer compatibility
- Training performance

It's not just a "nice to have" - it's a **best practice** for dataset preprocessing!
