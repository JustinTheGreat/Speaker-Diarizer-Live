import torch
import torchaudio
import os
from speechbrain.inference.speaker import EncoderClassifier
from torch.nn.functional import cosine_similarity
import torchaudio.transforms as T

def load_speaker_encoder(device: str):
    """
    Loads the ECAPA-TDNN model once.
    """
    print(f"\n[Recognition] Loading SpeechBrain Encoder on {device}...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
        savedir=os.path.join("pretrained_models", "spkrec-ecapa-voxceleb")
    )
    return classifier

def recognize_speakers_live(
    audio_path: str,
    segments: list,
    classifier_model,
    registry_path: str = "speaker_registry.pt",
    threshold: float = 0.25,
    device: str = "cuda"
) -> dict:
    """
    Identifies speakers using the PRE-LOADED classifier and prints comparison scores.
    """
    # 1. Check registry
    if not os.path.exists(registry_path):
        return {}
    
    try:
        registry = torch.load(registry_path)
    except Exception as e:
        print(f"[Recognition] Error loading registry: {e}")
        return {}

    if not registry:
        return {}

    # 2. Load Audio
    try:
        signal, fs = torchaudio.load(audio_path)
    except:
        return {}

    # Resample if needed
    if fs != 16000:
        resampler = T.Resample(fs, 16000)
        signal = resampler(signal)
        fs = 16000
    
    signal = signal.to(device)

    # 3. Group segments
    speaker_segments = {}
    for seg in segments:
        label = seg.get("speaker", "UNKNOWN")
        if label == "UNKNOWN": continue
        
        if label not in speaker_segments:
            speaker_segments[label] = []
        
        start_sample = int(seg["start"] * fs)
        end_sample = int(seg["end"] * fs)
        speaker_segments[label].append((start_sample, end_sample))

    # 4. Compare
    identified_map = {}
    print(f"\n[Recognition] Comparing {len(speaker_segments)} detected speakers against registry...")
    
    for spk_id, times in speaker_segments.items():
        embeddings = []
        
        # Limit to 10 segments per speaker for speed
        for start, end in times[:10]: 
            if end > signal.shape[1]: end = signal.shape[1]
            if end - start < 400: continue 

            slice_audio = signal[:, start:end]

            with torch.no_grad():
                emb = classifier_model.encode_batch(slice_audio).squeeze(0).cpu()
                embeddings.append(emb)

        if not embeddings:
            continue

        # Average embedding
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

        # --- DETAILED LOGGING ADDED HERE ---
        if best_score >= threshold:
            print(f"  > {spk_id} MATCHED '{best_match}' (Score: {best_score:.3f})")
            # CHANGE: Return a dictionary with name AND score
            identified_map[spk_id] = {"name": best_match, "score": best_score} 
        else:
            print(f"  > {spk_id} is Unknown. (Closest: '{best_match}', Score: {best_score:.3f})")

    return identified_map