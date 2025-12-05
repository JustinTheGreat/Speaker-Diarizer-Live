import torch
import os
import sys
import json
import time
from dotenv import load_dotenv

# Import components
from whisperx_component import process_audio
from llm_component import generate_local_response
from speechbrain_component import extract_and_save_speaker_data
from speaker_training_component import train_speaker_model
from speaker_recognition_component import recognize_speakers
# Import the VAD Recorder
from recorder_threshold import VADRecorder

# --- Helper Functions for LLM Interaction ---

def format_transcript_for_llm(segments: list, pre_identified_map: dict = None) -> str:
    """
    Converts segments to text.
    """
    if pre_identified_map is None:
        pre_identified_map = {}

    transcript_text = ""
    for segment in segments:
        raw_speaker = segment.get("speaker", "UNKNOWN")
        # Use existing name if available, else raw ID
        display_name = pre_identified_map.get(raw_speaker, raw_speaker)
        
        text = segment["text"].strip()
        transcript_text += f"[{display_name}]: {text}\n"
    return transcript_text

def create_llm_prompt(transcript_text: str) -> str:
    prompt = f"""
    Given the following transcript,
    I want you to give back speaker name labels if any of them are identified, otherwise label them as "Unknown_Speaker_N".
    Return your result in JSON format in this format:
    {{
    "SPEAKER_00": "Label"
    }}

    TRANSCRIPT:
    ---
    {transcript_text}
    ---
    """
    return prompt

def parse_speaker_map_from_llm(response_text: str) -> dict:
    """Safely parses the LLM's text response to extract a JSON object."""
    try:
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            return {}

        json_string = response_text[json_start:json_end]
        speaker_map = json.loads(json_string)
        if isinstance(speaker_map, dict):
            return speaker_map
        return {}
    except Exception:
        return {}

# --- Pipeline Function ---

def run_pipeline_on_file(audio_file_path: str, hf_token: str, device: str, compute_type: str):
    """
    Runs the full processing pipeline on a single audio file.
    """
    print(f"\n[Pipeline] Processing captured file: {audio_file_path}")

    # --- Step 1: Transcription & Diarization ---
    print("\n--- 1. TRANSCRIPTION & DIARIZATION ---")
    final_segments = process_audio(audio_file_path, device, compute_type, hf_token)

    if not final_segments:
        print("[Pipeline] No speech segments found in recording.")
        return

    # --- Step 2: LLM Contextual Identification (PRIORITY) ---
    print("\n--- 2. LLM CONTEXTUAL IDENTIFICATION ---")
    raw_transcript = format_transcript_for_llm(final_segments, pre_identified_map={})
    llm_prompt = create_llm_prompt(raw_transcript)

    print("[Pipeline] Sending transcript to LLM...")
    llm_response = generate_local_response(llm_prompt, max_new_tokens=300)
    llm_map = parse_speaker_map_from_llm(llm_response)

    print(f"[Pipeline] LLM identified: {llm_map}")

    # --- Step 3: Biometric Speaker Recognition ---
    print("\n--- 3. BIOMETRIC SPEAKER RECOGNITION ---")
    biometric_map = recognize_speakers(
        audio_path=audio_file_path,
        segments=final_segments,
        registry_path="speaker_registry.pt",
        threshold=0.35,
        device=device
    )

    if biometric_map:
        print(f"[Pipeline] Biometrics identified: {biometric_map}")
    else:
        print("[Pipeline] No known speakers recognized biometrically.")

    # --- Step 4: Merge Results ---
    final_speaker_map = biometric_map.copy()
    final_speaker_map.update(llm_map) # LLM overrides biometrics if conflict

    print(f"\n[Pipeline] Final merged speaker map: {final_speaker_map}")

    # --- Step 5: Output Final Labeled Transcript ---
    print("\n" + "="*40)
    print(" LIVE TRANSCRIPT ")
    print("="*40)
    previous_speaker = None
    for segment in final_segments:
        raw_speaker = segment.get("speaker", "UNKNOWN")
        display_name = final_speaker_map.get(raw_speaker, raw_speaker)
        start = segment["start"]
        text = segment["text"].strip()
        
        if display_name != previous_speaker:
            print(f"\n[{start:.2f}s] {display_name}: ", end="")
            previous_speaker = display_name
        print(text, end=" ")
    print("\n" + "="*40 + "\n")

    # --- Step 6: Update Speaker Registry ---
    print("--- 4. UPDATING SPEAKER REGISTRY ---")
    known_segments = []
    for segment in final_segments:
        raw_spk = segment.get("speaker", "UNKNOWN")
        name = final_speaker_map.get(raw_spk, raw_spk)
        
        is_unknown = (
            "unknown" in name.lower() or 
            name.startswith("SPEAKER_") or 
            name == raw_spk 
        )
        
        if not is_unknown:
            known_segments.append(segment)

    if known_segments:
        print(f"[Pipeline] Enrolling {len(known_segments)} segments...")
        extract_and_save_speaker_data(audio_file_path, known_segments, final_speaker_map)
        train_speaker_model(data_dir="speaker_data", model_save_path="speaker_registry.pt", device=device)
    else:
        print("[Pipeline] No new identified speakers to enroll.")

# --- Main Execution Loop ---

if __name__ == "__main__":
    # 1. Load Environment
    load_dotenv() 
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN environment variable not found.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    # 2. Initialize VAD Recorder
    # Adjustable: amplitude_threshold=0.01 to 0.05 depends on mic sensitivity
    recorder = VADRecorder(amplitude_threshold=0.02, min_record_duration_ms=1000)
    
    print(f"\n--- SYSTEM READY ({device.upper()}) ---")
    print("Listening for speech... (Press Ctrl+C to stop)")

    recording_filename = "auto.wav"

    try:
        while True:
            # 3. Blocking call: waits for sound, records, waits for silence, returns filename
            # This uses the logic from stt_threshold.py / recorder_threshold.py
            output_file = recorder.record(recording_filename)
            
            if output_file:
                # 4. Run the existing pipeline on the newly recorded file
                try:
                    run_pipeline_on_file(output_file, hf_token, device, compute_type)
                except Exception as e:
                    print(f"Error in pipeline processing: {e}")
                
                print("\n[System] Listening again...\n")
                
            else:
                # If recorder returns None (e.g. glitch or too short), just loop
                pass

    except KeyboardInterrupt:
        print("\n\nStopping system. Goodbye.")
        sys.exit(0)