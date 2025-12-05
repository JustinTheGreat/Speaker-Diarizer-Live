import os
import torch
import torchaudio
import torchaudio.transforms as T
from speechbrain.lobes.features import MFCC

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
    
    print(f"\n[SpeechBrain Component] Loading audio from {audio_path}...")
    try:
        # Load the full audio file
        # signal is Tensor [channels, time], fs is sample rate
        signal, fs = torchaudio.load(audio_path)
    except Exception as e:
        print(f"[SpeechBrain Component] Error loading audio: {e}")
        return

    # --- FIX: Resample to 16kHz if necessary ---
    # SpeechBrain defaults (n_fft=400) are optimized for 16kHz. 
    # If fs is higher (e.g. 48k), win_length (25ms) becomes 1200 samples, 
    # which is > n_fft, causing the crash.
    target_fs = 16000
    if fs != target_fs:
        print(f"[SpeechBrain Component] Resampling audio from {fs}Hz to {target_fs}Hz...")
        resampler = T.Resample(fs, target_fs)
        signal = resampler(signal)
        fs = target_fs

    # Initialize SpeechBrain MFCC computer
    # n_mels=23 or 40 is standard. 
    compute_mfcc = MFCC(sample_rate=fs, n_mels=40)

    print(f"[SpeechBrain Component] Extracting features and saving to '{output_dir}'...")

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
        # Timestamps are in seconds, so we multiply by the NEW sample rate (16000)
        start_sec = segment["start"]
        end_sec = segment["end"]
        
        start_sample = int(start_sec * fs)
        end_sample = int(end_sec * fs)

        # Ensure we don't go out of bounds
        if end_sample > signal.shape[1]:
            end_sample = signal.shape[1]
        
        # Skip extremely short segments that might cause empty slices
        if end_sample - start_sample < 400: # 400 samples = 25ms
            continue

        # 4. Extract Audio Slice
        audio_slice = signal[:, start_sample:end_sample]

        # 5. Compute MFCCs
        # speechbrain expects [batch, time]. audio_slice is [channels, time].
        try:
            with torch.no_grad():
                mfcc_features = compute_mfcc(audio_slice)
        except RuntimeError as e:
            print(f"Warning: Could not extract MFCC for segment {i}: {e}")
            continue

        # 6. Save Data
        # Save the audio clip (.wav)
        wav_filename = f"{safe_name}_seg{i:03d}.wav"
        wav_path = os.path.join(speaker_dir, wav_filename)
        torchaudio.save(wav_path, audio_slice, fs)

        # Save the MFCC features (.pt)
        pt_filename = f"{safe_name}_seg{i:03d}_mfcc.pt"
        pt_path = os.path.join(speaker_dir, pt_filename)
        torch.save(mfcc_features, pt_path)

    print(f"[SpeechBrain Component] Processing complete. Data saved in '{output_dir}/'.")