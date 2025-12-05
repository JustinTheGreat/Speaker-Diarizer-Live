import torch
import os
import sys
import json
import time
from dotenv import load_dotenv

# --- Import Refactored Components ---
# Note the function name changes to match the files above
from whisperx_component import load_whisperx_models, process_audio_live
from llm_component import load_llm_pipeline, generate_response_live
from speaker_recognition_component import load_speaker_encoder, recognize_speakers_live
from recorder_threshold import VADRecorder
from speechbrain_component import extract_and_save_speaker_data
from speaker_training_component import train_speaker_model

# --- Helper Functions (Same as before) ---
def format_transcript_for_llm(segments: list, pre_identified_map: dict = None) -> str:
    if pre_identified_map is None: pre_identified_map = {}
    transcript_text = ""
    for segment in segments:
        raw_speaker = segment.get("speaker", "UNKNOWN")
        display_name = pre_identified_map.get(raw_speaker, raw_speaker)
        text = segment["text"].strip()
        transcript_text += f"[{display_name}]: {text}\n"
    return transcript_text

def create_llm_prompt(transcript_text: str) -> str:
    return f"""
    Given the transcript below, identify speaker names.
    Return JSON format: {{"SPEAKER_00": "Name"}}
    If unknown, ignore.
    
    TRANSCRIPT:
    {transcript_text}
    """

def parse_speaker_map_from_llm(response_text: str) -> dict:
    try:
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start == -1: return {}
        return json.loads(response_text[json_start:json_end])
    except:
        return {}

# --- The Pipeline Function (Optimized) ---
def run_live_pipeline(
    audio_file_path: str, 
    device: str,
    # Pass in the loaded models
    whisper_objects: dict,
    llm_pipe: object,
    speaker_encoder: object
):
    start_time = time.time()
    
    # 1. Transcription (Fast, using pre-loaded model)
    segments = process_audio_live(audio_file_path, whisper_objects, device)
    if not segments:
        return

    # 2. LLM Identification
    raw_transcript = format_transcript_for_llm(segments, {})
    llm_prompt = create_llm_prompt(raw_transcript)
    llm_response = generate_response_live(llm_pipe, llm_prompt)
    llm_map = parse_speaker_map_from_llm(llm_response)

    # 3. Biometric Recognition (Fast, using pre-loaded encoder)
    biometric_map = recognize_speakers_live(
        audio_path=audio_file_path,
        segments=segments,
        classifier_model=speaker_encoder, # Pass model
        registry_path="speaker_registry.pt",
        threshold=0.35,
        device=device
    )

    # 4. Merge
    final_map = biometric_map.copy()
    final_map.update(llm_map)

    # 5. Output
    print(f"\n--- Result ({time.time() - start_time:.2f}s processing) ---")
    prev_spk = None
    for seg in segments:
        raw = seg.get("speaker", "UNKNOWN")
        name = final_map.get(raw, raw)
        if name != prev_spk:
            print(f"\n[{name}]:", end=" ")
            prev_spk = name
        print(seg["text"].strip(), end=" ")
    print("\n----------------------------------------------\n")

    # 6. Update Registry (Optional/Background)
    # We skip optimization here as this happens rarely
    known_segments = [
        s for s in segments 
        if not ("unknown" in final_map.get(s.get("speaker"), "").lower() or 
                final_map.get(s.get("speaker")).startswith("SPEAKER_"))
    ]
    
    if known_segments:
        print("[System] Updating speaker registry...")
        extract_and_save_speaker_data(audio_file_path, known_segments, final_map)
        train_speaker_model(data_dir="speaker_data", model_save_path="speaker_registry.pt", device=device)

# --- Main Startup ---
if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token: raise EnvironmentError("HF_TOKEN missing.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use float16 for CUDA to save VRAM
    compute_type = "float16" if device == "cuda" else "int8"
    
    print("\n" + "="*50)
    print(" SYSTEM STARTUP - LOADING MODELS ")
    print(" (This may take 10-20 seconds) ")
    print("="*50)

    # --- LOAD EVERYTHING ONCE ---
    try:
        # 1. WhisperX
        whisper_objects = load_whisperx_models(device, compute_type, hf_token)
        
        # 2. LLM (Gemma)
        llm_pipe = load_llm_pipeline(device)
        
        # 3. SpeechBrain Encoder
        speaker_encoder = load_speaker_encoder(device)
        
        # 4. VAD Recorder
        recorder = VADRecorder(amplitude_threshold=0.02)
        
    except Exception as e:
        print(f"Startup Failure: {e}")
        sys.exit(1)

    print("\n[System] All models loaded. Ready for live input.\n")

    while True:
        try:
            # 1. Listen
            audio_file = recorder.record("live_input.wav")
            
            if audio_file:
                # 2. Process with loaded models
                run_live_pipeline(
                    audio_file, 
                    device, 
                    whisper_objects, 
                    llm_pipe, 
                    speaker_encoder
                )
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break