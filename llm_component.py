import torch
from transformers import pipeline
import os

def load_llm_pipeline(device: str, model_id: str = "google/gemma-3-4b-it"):
    """
    Loads the LLM pipeline once.
    """
    if not os.getenv("HF_TOKEN"):
        print("Error: HF_TOKEN not set in environment.")
        return None

    print(f"\n[LLM] Loading model {model_id} on {device}...")
    
    model_kwargs = {"device": device}
    if device == "cuda":
        model_kwargs["dtype"] = torch.bfloat16

    pipe = pipeline(
        "text-generation", 
        model=model_id,
        **model_kwargs
    )
    return pipe

def generate_response_live(pipe, prompt: str, max_new_tokens: int = 100) -> str:
    """
    Generates text using the PRE-LOADED pipeline.
    """
    if pipe is None:
        return "Error: LLM pipeline is not loaded."

    try:
        messages = [{"role": "user", "content": prompt}]
        
        outputs = pipe(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            return_full_text=False
        )
        
        return outputs[0]['generated_text']

    except Exception as e:
        return f"Error during generation: {e}"