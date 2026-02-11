# Fixing "ArrowInvalid: cannot mix list and non-list" Error

## The Problem

**Error Message:**
```
ArrowInvalid: cannot mix list and non-list, non-null values
```

**Root Cause:** Your mapping functions return **inconsistent types** for the same field.

## What Was Wrong

### ❌ Your Original Code

```python
def format_mathcodeinstruct(example):
    messages = example["messages"]
    if not messages:
        return {"messages": ""}  # ← Returns STRING

    # ... processing ...

    if not user_msg or not assistant_msg:
        return {"messages": ""}  # ← Returns STRING

    return {
        "messages": [  # ← Returns LIST
            {"role": "user", ...},
            {"role": "assistant", ...}
        ]
    }

def format_scibench(example):
    if not problem or not solution:
        return None  # ← Returns None

    return {
        "messages": [...]  # ← Returns LIST
    }
```

**The Issue:**
- Example 1: `{"messages": ""}`        ← string
- Example 2: `{"messages": [...]}`     ← list
- Example 3: `None`                    ← null

Arrow can't create a column that's sometimes string, sometimes list, sometimes null!

## The Solution

### ✅ Fixed Code - Always Return Same Type

```python
def format_mathcodeinstruct(example):
    try:
        messages = example.get("messages", [])
        if not messages:
            return None  # ← Always None for invalid

        # ... processing ...

        if not user_msg or not assistant_msg:
            return None  # ← Always None for invalid

        return {
            "messages": [  # ← Always LIST for valid
                {"role": "user", ...},
                {"role": "assistant", ...}
            ]
        }
    except Exception:
        return None  # ← Always None for errors

def format_scibench(example):
    try:
        if not problem or not solution:
            return None  # ← Always None for invalid

        return {
            "messages": [...]  # ← Always LIST for valid
        }
    except Exception:
        return None  # ← Always None for errors
```

**Now:**
- Valid example: `{"messages": [...]}`  ← always list
- Invalid example: `None`               ← always null
- **Consistent!** Arrow is happy! ✅

## Additional Fixes in the Code

### Fix 1: Handle Different assistant_msg Structures

**Problem:** MathCodeInstruct's `assistant_msg` might be:
- A string: `"Solve using x=5"`
- A list of dicts: `[{"type": "text", "content": "..."}, {"type": "code", "content": "..."}]`

**Solution:**
```python
# Check if assistant_msg is a list (structured content)
if isinstance(assistant_msg, list):
    for msg in assistant_msg:
        if isinstance(msg, dict):
            if msg.get('type') == 'code':
                formatted_string += f"<code>\n{msg.get('content', '')}\n</code>\n"
            else:
                formatted_string += f"{msg.get('content', '')}\n"
    # Wrap in <think>
    formatted_string = f"<think>\n{formatted_string}\n</think>\n\nSolution provided above."
else:
    # If assistant_msg is just a string
    formatted_string = f"<think>\n{assistant_msg}\n</think>\n\nSolution provided above."
```

### Fix 2: Use try-except for Safety

```python
def format_mathcodeinstruct(example):
    try:
        # ... all processing ...
        return {"messages": [...]}
    except Exception as e:
        return None  # Gracefully handle any unexpected errors
```

This prevents crashes from malformed data.

### Fix 3: Better Filtering

```python
# Filter with explicit check
math_formatted = math_formatted.filter(
    lambda x: x is not None and x.get('messages') is not None,
    desc="Filtering MathCodeInstruct"
)
```

This removes all `None` values before concatenating.

## How to Debug This Issue

### Step 1: Inspect Your Data

Run this to see the actual structure:
```python
from datasets import load_dataset

# Load small sample
dataset = load_dataset("mathllm/MathCodeInstruct", split="train[:10]")

# Check first example
print(dataset[0])
print("\nMessages structure:")
print(dataset[0]['messages'])
```

Or use the provided script:
```bash
python inspect_dataset_structure.py
```

### Step 2: Test Your Function

```python
# Test on single example
example = math_subset[0]
result = format_mathcodeinstruct(example)
print(f"Result type: {type(result)}")
print(f"Result: {result}")

# Test on multiple examples
for i in range(5):
    result = format_mathcodeinstruct(math_subset[i])
    result_type = type(result).__name__
    messages_type = type(result.get('messages')).__name__ if result else 'None'
    print(f"Example {i}: result={result_type}, messages={messages_type}")
```

**You should see:**
```
Example 0: result=dict, messages=list
Example 1: result=dict, messages=list
Example 2: result=NoneType, messages=None
Example 3: result=dict, messages=list
Example 4: result=dict, messages=list
```

If you see mixed types (string, list, dict), that's the problem!

### Step 3: Check After Filtering

```python
# After mapping
print(f"Before filter: {len(math_formatted)}")

# After filtering
math_formatted = math_formatted.filter(lambda x: x is not None)
print(f"After filter: {len(math_formatted)}")

# Verify all have messages
sample = math_formatted[0]
print(f"Sample messages type: {type(sample['messages'])}")
```

## Common Variations of This Error

### Variant 1: Mixed Nested Structures

```python
# ❌ BAD
return {"messages": [{"role": "user", "content": "..."}]}  # 1 item
return {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}  # 2 items
# Different list lengths are OK, but mixing with non-lists is not!
```

### Variant 2: Mixed String/List

```python
# ❌ BAD
return {"text": "hello"}   # string
return {"text": ["hello"]}  # list with 1 string
# Arrow sees: sometimes string, sometimes list → ERROR
```

### Variant 3: Mixed Dict/List

```python
# ❌ BAD
return {"data": {"key": "value"}}  # dict
return {"data": [{"key": "value"}]}  # list of dicts
# Arrow sees: sometimes dict, sometimes list → ERROR
```

## Key Takeaways

1. **Always return same type** for the same field across all examples
2. **Use `None` for invalid examples**, not empty strings or empty dicts
3. **Filter out `None`** values after mapping
4. **Use try-except** to catch unexpected data structures
5. **Test on small sample** before running on full dataset
6. **Inspect your data** to understand its structure first

## Testing Your Fix

After applying the fix:

```python
# This should now work without errors
math_formatted = math_subset.map(
    format_mathcodeinstruct,
    remove_columns=math_subset.column_names,
    desc="Formatting MathCodeInstruct"
)

# Filter out None
math_formatted = math_formatted.filter(lambda x: x is not None)

# Verify
print(f"✓ Formatted {len(math_formatted):,} examples")
print(f"✓ Sample: {math_formatted[0]}")
```

If you still get the error, run `inspect_dataset_structure.py` to see the actual data format!
