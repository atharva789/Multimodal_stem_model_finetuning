"""
Script to inspect the structure of MathCodeInstruct dataset
Run this to understand the data format before processing
"""

from datasets import load_dataset
import json

print("Loading MathCodeInstruct sample...")
math_dataset = load_dataset("mathllm/MathCodeInstruct", split="train", streaming=True)

# Get first few examples
examples = []
for i, example in enumerate(math_dataset):
    if i >= 5:  # Get 5 examples
        break
    examples.append(example)

print("\n" + "="*80)
print("MATHCODEINSTRUCT STRUCTURE ANALYSIS")
print("="*80)

# Analyze first example in detail
print("\n1. FIRST EXAMPLE - Full Structure:")
print("-" * 80)
print(json.dumps(examples[0], indent=2, default=str))

# Check messages structure
print("\n2. MESSAGES FIELD STRUCTURE:")
print("-" * 80)
if 'messages' in examples[0]:
    messages = examples[0]['messages']
    print(f"Type of 'messages': {type(messages)}")
    print(f"Number of messages: {len(messages) if isinstance(messages, list) else 'N/A'}")

    if isinstance(messages, list) and len(messages) > 0:
        print("\nFirst message:")
        print(json.dumps(messages[0], indent=2, default=str))

        # Check assistant message structure
        for msg in messages:
            if msg.get('role') == 'assistant':
                print("\nAssistant message found:")
                content = msg.get('content')
                print(f"Content type: {type(content)}")

                if isinstance(content, str):
                    print(f"Content (string): {content[:200]}...")
                elif isinstance(content, list):
                    print(f"Content (list) with {len(content)} items:")
                    for i, item in enumerate(content[:3]):  # First 3 items
                        print(f"  Item {i}: {item}")
                break

# Check all column names
print("\n3. AVAILABLE COLUMNS:")
print("-" * 80)
print(list(examples[0].keys()))

# Sample multiple examples to see variations
print("\n4. MESSAGE STRUCTURES ACROSS 5 EXAMPLES:")
print("-" * 80)
for i, ex in enumerate(examples):
    if 'messages' in ex:
        msgs = ex['messages']
        print(f"\nExample {i+1}:")
        for msg in msgs:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            content_type = type(content).__name__
            content_preview = str(content)[:50] if not isinstance(content, list) else f"list[{len(content)}]"
            print(f"  {role}: {content_type} - {content_preview}...")

print("\n" + "="*80)
print("RECOMMENDATION:")
print("="*80)
print("Based on this output, adjust your format_mathcodeinstruct function to handle")
print("the actual structure of the 'messages' field (string vs list of dicts).")
