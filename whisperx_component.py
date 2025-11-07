import whisperx
import torch
import gc
from whisperx.diarize import DiarizationPipeline
from typing import List, Dict, Any

def process_audio(
    audio_file: str, 
    device: str, 
    compute_type: str, 
    hf_token: str
) -> List[Dict[str, Any]]:
    """
    Runs the WhisperX transcription and pyannote diarization pipeline.

    Args:
        audio_file (str): Path to the input audio file.
        device (str): Device to use for computation ('cuda' or 'cpu').
        compute_type (str): Precision type ('float16' for CUDA, 'int8' for CPU).
        hf_token (str): Hugging Face access token for pyannote.audio models.

    Returns:
        List[Dict[str, Any]]: List of transcribed segments with speaker labels, or an empty list on error.
    """
    
    print(f"\n[Module] Loading audio from: {audio_file}")
    try:
        audio = whisperx.load_audio(audio_file)
    except FileNotFoundError:
        print(f"Error: Audio file not found at '{audio_file}'.")
        return []

    print("[Module] 1. Loading Whisper model...")
    # Using the medium model as a good balance of speed and accuracy
    model = whisperx.load_model("medium", device, compute_type=compute_type)

    print("[Module] 2. Transcribing audio...")
    result = model.transcribe(audio, batch_size=16)

    print("[Module] 3. Aligning transcript for word-level timestamps...")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    print("[Module] 4. Applying speaker diarization...")
    # NOTE: DiarizationPipeline is imported correctly from whisperx.diarize
    diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarize_segments = diarize_model(audio) 

    print("[Module] 5. Assigning speaker labels...")
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # Optional: Clean up VRAM
    print("[Module] Cleaning up memory...")
    del model
    del model_a
    del diarize_model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return result["segments"]