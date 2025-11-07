import torch
import os
from dotenv import load_dotenv
import sys

# Import the core processing function from the component file
from whisperx_component import process_audio 


# --- Step 0: Load Environment Variables ---
load_dotenv() 

# --- Configuration & Setup ---

# 1. Set your audio file path here
audio_file = "AudioFile.wav" 

# 2. Check for CUDA GPU and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

# 3. Set compute type dynamically based on the detected device
if device == "cuda":
    compute_type = "float16"
elif device == "cpu":
    # Using 'int8' for better CPU compatibility and performance
    compute_type = "int8"
else:
    compute_type = "float32"

print(f"Using compute type: {compute_type}") 

# 4. Get the Hugging Face token securely from the environment
hf_token = os.getenv("HF_AUTH_TOKEN")
if not hf_token:
    raise EnvironmentError("HF_AUTH_TOKEN environment variable not found. Please ensure it is set in your .env file and you have accepted the pyannote/speaker-diarization terms on Hugging Face.")

# --- Execution ---

print("\n--- STARTING TRANSCRIPTION PIPELINE ---")
# Call the function from the imported module
final_segments = process_audio(audio_file, device, compute_type, hf_token)

# --- Output ---
if final_segments:
    print("\n--- FINAL DIARIZED TRANSCRIPT ---")
    previous_speaker = None

    for segment in final_segments:
        current_speaker = segment.get("speaker", "UNKNOWN")
        start = segment["start"]
        text = segment["text"].strip()
        
        # Start a new line and print the speaker label only when the speaker changes or it's the first segment
        if current_speaker != previous_speaker:
            print(f"\n[{start:.2f}s] {current_speaker}: ", end="")
            previous_speaker = current_speaker
        
        # Print the text content
        print(text, end=" ")
    print() # Final newline
else:
    print("\nPipeline failed or returned no segments.")