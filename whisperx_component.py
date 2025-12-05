import whisperx
import torch
import gc
from whisperx.diarize import DiarizationPipeline
from typing import List, Dict, Any

def load_whisperx_models(device: str, compute_type: str, hf_token: str):
    """
    Loads all WhisperX models into memory once.
    Returns a dictionary or tuple of the loaded model objects.
    """
    print(f"\n[WhisperX] Loading Whisper model (medium) on {device}...")
    model = whisperx.load_model("medium", device, compute_type=compute_type)

    print(f"[WhisperX] Loading Diarization model...")
    diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)

    return {
        "model": model,
        "diarize_model": diarize_model
    }

def process_audio_live(
    audio_file: str, 
    pipeline_objects: dict,
    device: str, 
) -> List[Dict[str, Any]]:
    """
    Runs transcription using PRE-LOADED models.
    """
    model = pipeline_objects["model"]
    diarize_model = pipeline_objects["diarize_model"]

    try:
        audio = whisperx.load_audio(audio_file)
    except FileNotFoundError:
        print(f"Error: Audio file not found at '{audio_file}'.")
        return []

    # 1. Transcribe
    result = model.transcribe(audio, batch_size=16)

    # 2. Align (Alignment models are small, safe to load on fly, but better here)
    # Note: Loading alignment model requires knowing the language first. 
    # We accept the small overhead of loading 'model_a' here to keep logic simple,
    # or you can preload 'en' alignment if you only speak English.
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], 
        device=device
    )
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # 3. Diarize
    diarize_segments = diarize_model(audio) 

    # 4. Assign Speakers
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # Cleanup alignment model only (it's transient)
    del model_a
    # Do NOT delete 'model' or 'diarize_model'!
    
    return result["segments"]