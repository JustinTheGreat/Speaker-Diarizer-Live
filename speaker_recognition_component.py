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
    classifier_model,  # The pre-loaded model
    registry_path: str = "speaker_registry.pt",
    threshold: float = 0.25,
    device: str = "cuda"
) -> dict:
    """
    Identifies speakers using the PRE-LOADED classifier.
    """
    if not os.path.exists(registry_path):
        return {}

    # Load registry (this is small, can be loaded every time or passed in)
    try:
        registry = torch.load(registry_path)
    except:
        return {}

    # Load Audio
    try:
        signal, fs = torchaudio.load(audio_path)
    except:
        return {}

    if fs != 16000:
        resampler = T.Resample(fs, 16000)
        signal = resampler(signal)
        fs = 16000
    
    signal = signal.to(device)

    # ... [Logic identical to previous version, but using 'classifier_model'] ...
    
    speaker_segments = {}
    for seg in segments:
        label = seg.get("speaker", "UNKNOWN")
        if label == "UNKNOWN": continue
        if label not in speaker_segments: speaker_segments[label] = []
        start_sample = int(seg["start"] * fs)
        end_sample = int(seg["end"] * fs)
        speaker_segments[label].append((start_sample, end_sample))

    identified_map = {}
    
    for spk_id, times in speaker_segments.items():
        embeddings = []
        for start, end in times[:10]: 
            if end > signal.shape[1]: end = signal.shape[1]
            if end - start < 400: continue 

            slice_audio = signal[:, start:end]
            with torch.no_grad():
                # Use passed model
                emb = classifier_model.encode_batch(slice_audio).squeeze(0).cpu()
                embeddings.append(emb)

        if embeddings:
            session_embedding = torch.mean(torch.stack(embeddings), dim=0)
            best_score = -1.0
            best_match = None

            for known_name, known_emb in registry.items():
                score = cosine_similarity(session_embedding, known_emb, dim=-1).item()
                if score > best_score:
                    best_score = score
                    best_match = known_name

            if best_score >= threshold:
                identified_map[spk_id] = best_match
            # Optional: Print debug info here if needed

    return identified_map