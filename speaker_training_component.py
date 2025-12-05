import torch
import torchaudio
import os
import glob
from speechbrain.inference.speaker import EncoderClassifier
from torch.nn.functional import cosine_similarity

def train_speaker_model(
    data_dir: str = "speaker_data",
    model_save_path: str = "speaker_registry.pt",
    device: str = "cuda"
):
    """
    Scans the data_dir for speaker folders, computes the mean embedding (fingerprint)
    for each speaker using ECAPA-TDNN, and saves the resulting registry to disk.

    Args:
        data_dir (str): Path to the folder containing speaker subfolders (e.g., 'speaker_data').
        model_save_path (str): Where to save the resulting dictionary of embeddings.
        device (str): 'cuda' or 'cpu'.
    """
    if not os.path.exists(data_dir):
        print(f"[Training] Data directory '{data_dir}' not found. Skipping training.")
        return

    print(f"\n[Training] Loading ECAPA-TDNN model on {device}...")
    # Load the pre-trained SpeechBrain model
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
        savedir=os.path.join("pretrained_models", "spkrec-ecapa-voxceleb") 
    )

    speaker_registry = {}
    
    # Iterate over each speaker folder in the data directory
    speaker_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    
    if not speaker_folders:
        print("[Training] No speaker folders found.")
        return

    print(f"[Training] Found speakers: {speaker_folders}. Starting enrollment...")

    for speaker_name in speaker_folders:
        speaker_path = os.path.join(data_dir, speaker_name)
        # Find all .wav files for this speaker
        wav_files = glob.glob(os.path.join(speaker_path, "*.wav"))
        
        if not wav_files:
            continue

        embeddings_list = []
        
        for wav_file in wav_files:
            try:
                # Load audio
                signal, fs = torchaudio.load(wav_file)
                
                # Ensure correct device
                signal = signal.to(device)

                # Generate embedding (Result is [1, 1, 192])
                # We use .detach() to save memory since we aren't backpropagating
                with torch.no_grad():
                    embedding = classifier.encode_batch(signal)
                    embeddings_list.append(embedding.squeeze(0)) # Remove batch dim -> [1, 192]
            
            except Exception as e:
                print(f"[Training] Error processing {wav_file}: {e}")

        if embeddings_list:
            # Stack all embeddings for this speaker and calculate the mean (centroid)
            # Shape: [N_samples, 1, 192] -> Mean -> [1, 192]
            all_embeddings = torch.stack(embeddings_list)
            mean_embedding = torch.mean(all_embeddings, dim=0)
            
            # Store in registry (move to CPU for saving)
            speaker_registry[speaker_name] = mean_embedding.cpu()
            print(f"[Training] Enrolled speaker: {speaker_name} ({len(wav_files)} samples)")

    # Save the registry
    torch.save(speaker_registry, model_save_path)
    print(f"[Training] Model saved to '{model_save_path}'.")


def identify_speaker(
    audio_path: str, 
    model_path: str = "speaker_registry.pt",
    threshold: float = 0.25,
    device: str = "cpu" 
) -> str:
    """
    (Helper Function)
    Identifies the speaker in a new audio file using the saved registry.
    """
    if not os.path.exists(model_path):
        return "Error: Model not found."

    # Load registry and model
    registry = torch.load(model_path)
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
        savedir=os.path.join("pretrained_models", "spkrec-ecapa-voxceleb")
    )

    # Process input audio
    signal, fs = torchaudio.load(audio_path)
    # Note: ECAPA-TDNN expects 16kHz. If input is different, add resampling here.
    if fs != 16000:
        resampler = torchaudio.transforms.Resample(fs, 16000)
        signal = resampler(signal)
    
    signal = signal.to(device)
    
    with torch.no_grad():
        input_embedding = classifier.encode_batch(signal).squeeze(0).cpu()

    # Compare with registry
    best_score = -1.0
    best_speaker = "Unknown"

    for speaker_name, target_embedding in registry.items():
        # Cosine similarity
        score = cosine_similarity(input_embedding, target_embedding, dim=-1)
        score = score.item()
        
        if score > best_score:
            best_score = score
            best_speaker = speaker_name

    if best_score < threshold:
        return f"Unknown (Best match: {best_speaker}, Score: {best_score:.2f})"
    
    return f"{best_speaker} (Score: {best_score:.2f})"