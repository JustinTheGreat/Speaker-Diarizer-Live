import torch
import os
import sys
import json
import time
from dotenv import load_dotenv

# --- Import Refactored Components ---
from whisperx_component import load_whisperx_models, process_audio_live
from llm_component import load_llm_pipeline, generate_response_live
from speaker_recognition_component import load_speaker_encoder, recognize_speakers_live
from recorder_threshold import VADRecorder
from speechbrain_component import extract_and_save_speaker_data
from speaker_training_component import train_speaker_model

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

def run_live_pipeline(
    audio_file_path: str, 
    device: str,
    whisper_objects: dict,
    llm_pipe: object,
    speaker_encoder: object
):
    start_time = time.time()
    
    # 1. Transcription
    segments = process_audio_live(audio_file_path, whisper_objects, device)
    if not segments:
        return

    # 2. LLM Identification
    raw_transcript = format_transcript_for_llm(segments, {})
    llm_prompt = create_llm_prompt(raw_transcript)
    llm_response = generate_response_live(llm_pipe, llm_prompt)
    llm_map = parse_speaker_map_from_llm(llm_response)

    # 3. Biometric Recognition
    # Returns complex object: {'SPEAKER_00': {'name': 'Bob', 'score': 0.45}}
    biometric_results = recognize_speakers_live(
        audio_path=audio_file_path,
        segments=segments,
        classifier_model=speaker_encoder, 
        registry_path="speaker_registry.pt",
        threshold=0.35,
        device=device
    )

    # --- FIX: Flatten Biometric Results for Compatibility ---
    # We extract just the name so 'final_map' contains strings, not dicts.
    biometric_map = {}
    for spk, data in biometric_results.items():
        if isinstance(data, dict):
            biometric_map[spk] = data["name"]
        else:
            biometric_map[spk] = data
    # -------------------------------------------------------

    # 4. Smart Merge (Prevent LLM from overwriting good data)
    final_map = biometric_map.copy()
    
    for spk_id, name in llm_map.items():
        # Check if the LLM returned a generic name
        is_generic = any(x in name.lower() for x in ["unknown", "speaker", "person"])
        
        if not is_generic:
            if spk_id in final_map and final_map[spk_id] != name:
                print(f"  [Merge] LLM overriding Biometrics for {spk_id}: '{final_map[spk_id]}' -> '{name}'")
            
            final_map[spk_id] = name

    # 5. ORPHAN HANDLING
    detected_names = list(set(final_map.values()))
    dominant_speaker = detected_names[0] if len(detected_names) == 1 else None

    print(f"\n--- Result ({time.time() - start_time:.2f}s) ---")
    
    prev_spk = None
    for seg in segments:
        raw = seg.get("speaker", "UNKNOWN")
        
        # Resolve Name
        if raw in final_map:
            name = final_map[raw]
        elif raw == "UNKNOWN" and dominant_speaker:
            name = dominant_speaker
        else:
            name = raw

        if name != prev_spk:
            print(f"\n[{name}]:", end=" ")
            prev_spk = name
        print(seg["text"].strip(), end=" ")
    print("\n----------------------------------------------\n")

    # 6. Update Registry
    known_segments = []
    for s in segments:
        raw = s.get("speaker")
        if raw and raw in final_map:
             known_segments.append(s)
    
    if known_segments:
        extract_and_save_speaker_data(audio_file_path, known_segments, final_map)
        train_speaker_model(
            data_dir="speaker_data", 
            model_save_path="speaker_registry.pt", 
            device=device,
            classifier_model=speaker_encoder
        )
        
if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token: 
        # Fallback to prevent immediate crash if not set, though models might fail later
        print("Warning: HF_TOKEN not found in environment.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    print("\n" + "="*50)
    print(" SYSTEM STARTUP - LOADING MODELS ")
    print("="*50)

    try:
        # 1. WhisperX
        whisper_objects = load_whisperx_models(device, compute_type, hf_token)
        
        # 2. LLM
        llm_pipe = load_llm_pipeline(device)
        
        # 3. SpeechBrain Encoder
        speaker_encoder = load_speaker_encoder(device)
        
        # 4. VAD Recorder
        recorder = VADRecorder(amplitude_threshold=0.02)
        
    except Exception as e:
        print(f"Startup Failure: {e}")
        sys.exit(1)

    print("\n[System] Ready. Listening...\n")

    while True:
        try:
            audio_file = recorder.record("live_input.wav")
            
            if audio_file:
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