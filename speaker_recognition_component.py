import torch
import torchaudio
import os
from speechbrain.inference.speaker import EncoderClassifier
from torch.nn.functional import cosine_similarity
import torchaudio.transforms as T

def recognize_speakers(
    audio_path: str,
    segments: list,
    registry_path: str = "speaker_registry.pt",
    threshold: float = 0.25,
    device: str = "cuda"
) -> dict:
    """
    Identifies speakers in the current audio segments by comparing them against 
    the known embeddings in the registry.

    Args:
        audio_path (str): Path to the new audio file.
        segments (list): The list of segments from WhisperX (contains 'start', 'end', 'speaker').
        registry_path (str): Path to the .pt file containing known speaker embeddings.
        threshold (float): Cosine similarity score (0.0 to 1.0) required to declare a match.
        device (str): 'cuda' or 'cpu'.

    Returns:
        dict: A map of current speaker IDs to known names (e.g., {'SPEAKER_00': 'Alice'}).
              If no match is found, the ID is omitted from the map (or mapped to itself).
    """
    
    # 1. Check if registry exists
    if not os.path.exists(registry_path):
        print(f"[Recognition] Registry '{registry_path}' not found. Skipping biometric recognition.")
        return {}

    print(f"\n[Recognition] Loading registry from {registry_path}...")
    try:
        registry = torch.load(registry_path)
    except Exception as e:
        print(f"[Recognition] Error loading registry: {e}")
        return {}

    if not registry:
        print("[Recognition] Registry is empty.")
        return {}

    # 2. Load Audio & Resample if needed
    print(f"[Recognition] Loading audio {audio_path}...")
    try:
        signal, fs = torchaudio.load(audio_path)
    except Exception as e:
        print(f"[Recognition] Failed to load audio: {e}")
        return {}

    # ECAPA-TDNN expects 16kHz
    if fs != 16000:
        print(f"[Recognition] Resampling {fs}Hz -> 16000Hz...")
        resampler = T.Resample(fs, 16000)
        signal = resampler(signal)
        fs = 16000
    
    signal = signal.to(device)

    # 3. Load Model
    print(f"[Recognition] Loading SpeechBrain model on {device}...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
        savedir=os.path.join("pretrained_models", "spkrec-ecapa-voxceleb")
    )

    # 4. Group segments by Speaker ID
    # We want to aggregate all audio for "SPEAKER_00" to get a robust embedding
    speaker_segments = {}
    for seg in segments:
        label = seg.get("speaker", "UNKNOWN")
        if label == "UNKNOWN": continue
        
        if label not in speaker_segments:
            speaker_segments[label] = []
        
        # Convert start/end seconds to samples
        start_sample = int(seg["start"] * fs)
        end_sample = int(seg["end"] * fs)
        speaker_segments[label].append((start_sample, end_sample))

    # 5. Compute Embeddings & Compare
    identified_map = {}

    print("[Recognition] Analyzing speakers...")
    
    for spk_id, times in speaker_segments.items():
        embeddings = []
        
        # Extract embeddings for up to 5 segments per speaker to save time/memory
        # (Averaging 5 good segments is usually enough)
        for start, end in times[:10]: 
            # Ensure valid bounds
            if end > signal.shape[1]: end = signal.shape[1]
            if end - start < 400: continue # Skip tiny segments (<25ms)

            slice_audio = signal[:, start:end]

            with torch.no_grad():
                # encode_batch returns [1, 1, 192]
                emb = classifier.encode_batch(slice_audio).squeeze(0).cpu()
                embeddings.append(emb)

        if not embeddings:
            continue

        # Average the embeddings for this speaker in the current session
        # Shape: [1, 192]
        session_embedding = torch.mean(torch.stack(embeddings), dim=0)

        # Compare against Registry
        best_score = -1.0
        best_match = None

        for known_name, known_emb in registry.items():
            # Cosine Similarity
            score = cosine_similarity(session_embedding, known_emb, dim=-1).item()
            
            if score > best_score:
                best_score = score
                best_match = known_name

        # Decide if match is valid
        if best_score >= threshold:
            print(f"  > {spk_id} matches '{best_match}' (Score: {best_score:.3f})")
            identified_map[spk_id] = best_match
        else:
            print(f"  > {spk_id} is Unknown (Best guess: {best_match}, Score: {best_score:.3f})")

    return identified_map