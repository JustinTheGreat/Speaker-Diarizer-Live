import os
import torch
import torchaudio
import torchaudio.transforms as T
from speechbrain.lobes.features import MFCC
import time

def extract_and_save_speaker_data(
    audio_path: str, 
    segments: list, 
    speaker_map: dict, 
    output_dir: str = "speaker_data"
):
    """
    Cuts audio segments based on diarization, extracts MFCC features using SpeechBrain,
    and organizes them into folders by speaker name.
    """
    
    # print(f"\n[SpeechBrain Component] Loading audio from {audio_path}...")
    try:
        signal, fs = torchaudio.load(audio_path)
    except Exception as e:
        print(f"[SpeechBrain Component] Error loading audio: {e}")
        return

    # --- Resample to 16kHz if necessary ---
    target_fs = 16000
    if fs != target_fs:
        resampler = T.Resample(fs, target_fs)
        signal = resampler(signal)
        fs = target_fs

    # Initialize SpeechBrain MFCC computer
    compute_mfcc = MFCC(sample_rate=fs, n_mels=40)

    # Generate a unique batch timestamp
    timestamp = int(time.time() * 1000) 

    for i, segment in enumerate(segments):
        # 1. Resolve Speaker Name
        raw_speaker = segment.get("speaker", "UNKNOWN")
        speaker_name = speaker_map.get(raw_speaker, raw_speaker)
        
        # Sanitize speaker name
        safe_name = "".join(x for x in speaker_name if x.isalnum() or x in "_- ")
        safe_name = safe_name.replace(" ", "_")

        # 2. Create Directory
        speaker_dir = os.path.join(output_dir, safe_name)
        os.makedirs(speaker_dir, exist_ok=True)

        # 3. Calculate start/end samples
        start_sec = segment["start"]
        end_sec = segment["end"]
        
        start_sample = int(start_sec * fs)
        end_sample = int(end_sec * fs)

        if end_sample > signal.shape[1]:
            end_sample = signal.shape[1]
        
        # Skip extremely short segments (< 25ms)
        if end_sample - start_sample < 400: 
            continue

        # 4. Extract Audio Slice
        audio_slice = signal[:, start_sample:end_sample]

        # 5. Compute MFCCs
        try:
            with torch.no_grad():
                mfcc_features = compute_mfcc(audio_slice)
        except RuntimeError as e:
            print(f"Warning: Could not extract MFCC for segment {i}: {e}")
            continue

        # 6. Save Data with UNIQUE Filename
        # OLD: f"{safe_name}_seg{i:03d}.wav" (Causes overwrites)
        # NEW: Includes timestamp
        filename_base = f"{safe_name}_{timestamp}_{i:03d}"
        
        wav_path = os.path.join(speaker_dir, f"{filename_base}.wav")
        pt_path = os.path.join(speaker_dir, f"{filename_base}_mfcc.pt")

        torchaudio.save(wav_path, audio_slice, fs)
        torch.save(mfcc_features, pt_path)

    # print(f"[SpeechBrain Component] Saved new training data for: {list(speaker_map.values())}")