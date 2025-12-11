import whisperx
import torch
import gc
from whisperx.diarize import DiarizationPipeline
from typing import List, Dict, Any

def load_whisperx_models(device: str, compute_type: str, hf_token: str):
    """
    Loads all WhisperX models into memory once.
    """
    print(f"\n[WhisperX] Loading Whisper model (medium) on {device}...")
    model = whisperx.load_model("medium", device, compute_type=compute_type)

    print(f"[WhisperX] Loading Diarization model...")
    diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)

    # PRE-LOAD English alignment model to prevent downloading it during live loop
    print(f"[WhisperX] Pre-loading Alignment model (en)...")
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)

    return {
        "model": model,
        "diarize_model": diarize_model,
        "align_model": model_a,    # Save this
        "align_metadata": metadata # Save this
    }

def process_audio_live(
    audio_file: str, 
    pipeline_objects: dict,
    device: str, 
) -> List[Dict[str, Any]]:
    """
    Runs transcription using PRE-LOADED models and forces English.
    """
    model = pipeline_objects["model"]
    diarize_model = pipeline_objects["diarize_model"]
    # Use pre-loaded alignment models
    model_a = pipeline_objects["align_model"]
    metadata = pipeline_objects["align_metadata"]

    try:
        audio = whisperx.load_audio(audio_file)
    except FileNotFoundError:
        return []

    # 1. Transcribe (Force English to avoid hallucinations/downloads)
    result = model.transcribe(audio, batch_size=16, language="en")

    # 2. Align (Use pre-loaded model)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # 3. Diarize
    diarize_segments = diarize_model(audio) 

    # 4. Assign Speakers
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # Garbage collection (Light)
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return result["segments"]