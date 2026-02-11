# CORRECTED VERSION - Cell 9
# Format MathCodeInstruct to chat format with <think> tags
def format_mathcodeinstruct(example):
    """Convert MathCodeInstruct to DeepSeek-R1 chat format with reasoning."""
    try:
        messages = example.get("messages", [])
        if not messages:
            return None  # ← Consistent: always None for invalid

        # Extract user question and assistant response
        user_msg = None
        assistant_msg = None

        for msg in messages:
            if msg.get('role') == 'user':
                user_msg = msg.get('content', '')
            elif msg.get('role') == 'assistant':
                assistant_msg = msg.get('content', '')

        if not user_msg or not assistant_msg:
            return None  # ← Consistent: always None for invalid

        # Handle assistant_msg - it might be a string or list of content blocks
        formatted_string = ""

        # Check if assistant_msg is a list (structured content)
        if isinstance(assistant_msg, list):
            for msg in assistant_msg:
                if isinstance(msg, dict):
                    if msg.get('type') == 'code':
                        formatted_string += f"<code>\n{msg.get('content', '')}\n</code>\n"
                    else:
                        formatted_string += f"{msg.get('content', '')}\n"
                else:
                    # If it's just a string in the list
                    formatted_string += str(msg) + "\n"

            # Wrap everything in <think> tags
            formatted_string = f"<think>\n{formatted_string.strip()}\n</think>\n\nThe solution is provided above."
        else:
            # If assistant_msg is just a string
            formatted_string = f"<think>\n{assistant_msg}\n</think>\n\nThe solution is provided above."

        return {
            "messages": [  # ← Consistent: always a list
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": formatted_string}
            ]
        }
    except Exception as e:
        # If anything goes wrong, return None
        return None


# Format SciBench to chat format with <think> tags
def format_scibench(example):
    """Convert SciBench to DeepSeek-R1 chat format with reasoning."""
    try:
        problem = example.get('problem_text', '')
        solution = example.get('solution', '')
        answer = example.get('answer_latex', '')
        unit = example.get('unit', '')

        if not problem or not solution:
            return None  # ← Consistent: always None for invalid

        # Build final answer
        final_answer = answer
        if unit:
            final_answer = f"{answer} {unit}"

        # Format with reasoning in <think> tags
        formatted_response = f"<think>\n{solution}\n</think>\n\nFinal Answer: {final_answer}"

        return {
            "messages": [  # ← Consistent: always a list
                {"role": "user", "content": problem},
                {"role": "assistant", "content": formatted_response}
            ]
        }
    except Exception as e:
        return None


print("Formatting datasets to chat format with <think> reasoning tags...")

# Apply formatting
math_formatted = math_subset.map(
    format_mathcodeinstruct,
    remove_columns=math_subset.column_names,
    desc="Formatting MathCodeInstruct"
)

scibench_formatted = scibench_subset.map(
    format_scibench,
    remove_columns=scibench_subset.column_names,
    desc="Formatting SciBench"
)

# Filter out None values (invalid examples)
print("Filtering out invalid examples...")
math_formatted = math_formatted.filter(
    lambda x: x is not None and x.get('messages') is not None,
    desc="Filtering MathCodeInstruct"
)
scibench_formatted = scibench_formatted.filter(
    lambda x: x is not None and x.get('messages') is not None,
    desc="Filtering SciBench"
)

print(f"After filtering:")
print(f"  MathCodeInstruct: {len(math_formatted):,} valid examples")
print(f"  SciBench: {len(scibench_formatted):,} valid examples")

# Recombine and shuffle
from datasets import concatenate_datasets
formatted_dataset = concatenate_datasets([math_formatted, scibench_formatted])
formatted_dataset = formatted_dataset.shuffle(seed=42)

print(f"\n✓ Formatted dataset: {len(formatted_dataset):,} examples")
print("\nExample:")
print(formatted_dataset[0])
