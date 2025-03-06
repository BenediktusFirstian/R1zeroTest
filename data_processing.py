from imports import *
from config import SYSTEM_PROMPT

# Function to structure the training data
def make_conversation(example):
    """Convert dataset examples into conversation format."""
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }

# Load and prepare dataset
def load_math_dataset():
    """Load and prepare the mathematics dataset."""
    dataset = load_dataset(
        "AI-MO/NuminaMath-TIR",
        name="default",
        split=['train', 'test']
    )
    
    # Convert splits into dictionary
    dataset = {
        'train': dataset[0],
        'test': dataset[1]
    }
    
    # Apply conversation format
    for split in dataset:
        dataset[split] = dataset[split].map(make_conversation)

        # Remove 'messages' column if exists
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")
    
    return dataset

def validate_dataset(dataset):
    """Perform basic validation checks on the dataset."""
    
    # Define the required fields for the dataset
    required_fields = ["problem", "prompt"]

    # Loop through the 'train' and 'test' splits of the dataset
    for split in ['train', 'test']:
        print(f"\nValidating {split} split:")

        # Retrieve column names from the dataset
        fields = dataset[split].column_names

        # Check if any required fields are missing
        missing = [field for field in required_fields if field not in fields]
        if missing:
            print(f"Warning: Missing fields: {missing}")  # Warn if fields are missing
        else:
            print("✓ All required fields present")  # Confirm all fields are present

        # Retrieve the first sample from the dataset split
        sample = dataset[split][0]

        # Extract the 'prompt' field, which contains a list of messages
        messages = sample['prompt']

        # Validate the prompt format:
        # - It should contain at least two messages
        # - The first message should be from the 'system' role
        # - The second message should be from the 'user' role
        if (len(messages) >= 2 and
            messages[0]['role'] == 'system' and
            messages[1]['role'] == 'user'):
            print("✓ Prompt format is correct")  # Confirm correct format
        else:
            print("Warning: Incorrect prompt format")  # Warn if format is incorrect

def load_bespoke_dataset():
    """Load the Bespoke-Stratos-17k dataset for SFT."""
    return load_dataset("bespokelabs/Bespoke-Stratos-17k", "default")

# Function for Few-shot Prompting with Long CoT
def generate_few_shot_examples():
    """Generate few-shot examples with long chain-of-thought."""
    few_shot_prompt = """
Problem: What's the square root of 9 plus 5?
Solution: <|special_token|> First, find the square root of 9, which is 3. Then, add 5 to 3.  3 + 5 equals 8. <|special_token|> Summary: The answer is 8.

Problem: Train travels at 60 mph for 2 hours, how far?
Solution: <|special_token|> Use the formula: Distance = Speed times Time. Speed is 60 mph, Time is 2 hours. Distance = 60 * 2 = 120 miles. <|special_token|> Summary: Train travels 120 miles.

Problem: What is 2 + 3 * 4?
Solution:
"""
    return few_shot_prompt

# Function for Direct Prompting
def generate_direct_prompt(question):
    """Generate a direct prompt asking for step-by-step reasoning."""
    return f"""
Problem: Solve this, show reasoning step-by-step, and verify:
{question}
"""

# Function for Post-processing Refinement
def refine_output(messy_text):
    """Refine messy output from R1 Zero to create better training data."""
    try:
        think_content = messy_text.split("<think>")[1].split("</think>")[0].strip()
        answer_content = messy_text.split("<answer>")[1].split("</answer>")[0].strip()

        refined_text = f"""<|special_token|> Reasoning: {think_content.replace('umm...', '').strip().capitalize()}.
<|special_token|> Summary: The answer is {answer_content}."""
        return refined_text
    except IndexError:
        # Handle cases where the expected tags aren't found
        return f"<|special_token|> {messy_text} <|special_token|> Summary: Unable to extract answer."