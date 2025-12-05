# This script demonstrates how to run the Gemma 1B model locally
# using the Hugging Face 'transformers' library for inference,
# while securely loading the Hugging Face token from a .env file.

# IMPORTANT PREREQUISITES:
# 1. Install necessary libraries:
#    pip install torch transformers accelerate bitsandbytes python-dotenv
# 2. For GPU usage (recommended):
#    pip install bitsandbytes
# 3. Create a .env file in the same directory with: HF_TOKEN="your_token_here"
# 4. You MUST have accepted the Gemma license on the Hugging Face Hub.

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os
# New import for loading environment variables from a .env file
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Use 'cuda' if you have an NVIDIA GPU, 'mps' for Apple Silicon, or 'cpu'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
MODEL_ID = "google/gemma-3-4b-it" # Gemma 3 with 1 Billion parameters, instruction-tuned

def generate_local_response(prompt: str, max_new_tokens: int = 100) -> str:
    """
    Loads the Gemma 1B model using the Hugging Face pipeline and generates a response.

    Args:
        prompt: The text prompt to send to the model.
        max_new_tokens: The maximum number of tokens to generate in the response.

    Returns:
        The generated text response.
    """
    
    if not os.getenv("HF_TOKEN"):
        return "Error: HF_TOKEN not found. Please ensure your .env file is correct and in the same directory."

    try:
        print(f"--- Loading model {MODEL_ID} on device: {DEVICE} ---")
        # Set dtype based on device for optimal performance
        model_kwargs = {"device": DEVICE}
        if DEVICE == "cuda":
            model_kwargs["dtype"] = torch.bfloat16

        # 1. Load the model using the high-level pipeline abstraction
        pipe = pipeline(
            "text-generation", 
            model=MODEL_ID,
            **model_kwargs # Pass device and dtype
        )

        # 2. Format the prompt using the chat template (required for -it instruction models)
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # 3. Generate the response
        outputs = pipe(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            return_full_text=False
        )
        
        # The output is a list of dicts. We extract the generated text.
        generated_text = outputs[0]['generated_text']

        return generated_text

    except Exception as e:
        return f"Error during model loading or generation. Did you install the prerequisites and log into Hugging Face? Error: {e}"

# --- Example Usage ---

if __name__ == "__main__":
    my_prompt = "Why is the sky blue? Explain it in two sentences."
    
    response_text = generate_local_response(my_prompt)

    print("\n" + "="*50)
    print(f"MODEL: {MODEL_ID} | DEVICE: {DEVICE}")
    print("="*50)
    print(f"[User Prompt]\n{my_prompt}")
    print(f"\n[Gemma Response]\n{response_text}")
    print("="*50)

    # Example 2: Short creative prompt
    another_prompt = "Tell me a fun fact about the number 42."
    fun_fact_response = generate_local_response(another_prompt, max_new_tokens=50)
    
    print(f"\n[User Prompt]\n{another_prompt}")
    print(f"\n[Gemma Response]\n{fun_fact_response}")
    print("="*50)