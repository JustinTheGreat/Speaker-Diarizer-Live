import torch
import torchaudio
import os
import glob
from speechbrain.inference.speaker import EncoderClassifier

def train_speaker_model(
    data_dir: str = "speaker_data",
    model_save_path: str = "speaker_registry.pt",
    device: str = "cuda",
    classifier_model = None # New optional argument
):
    """
    Updates the registry using the pre-loaded classifier model.
    """
    if not os.path.exists(data_dir):
        return

    # Use the passed model, or load it if not provided (fallback)
    if classifier_model is None:
        print(f"[Training] Loading ECAPA-TDNN model on {device}...")
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
            savedir=os.path.join("pretrained_models", "spkrec-ecapa-voxceleb") 
        )
    else:
        classifier = classifier_model

    speaker_registry = {}
    
    speaker_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    
    if not speaker_folders:
        return

    # print(f"[Training] Enrollment started for {len(speaker_folders)} speakers...")

    for speaker_name in speaker_folders:
        speaker_path = os.path.join(data_dir, speaker_name)
        wav_files = glob.glob(os.path.join(speaker_path, "*.wav"))
        
        if not wav_files: continue

        embeddings_list = []
        
        for wav_file in wav_files:
            try:
                signal, fs = torchaudio.load(wav_file)
                signal = signal.to(device)
                
                with torch.no_grad():
                    embedding = classifier.encode_batch(signal)
                    embeddings_list.append(embedding.squeeze(0))
            except Exception:
                pass

        if embeddings_list:
            all_embeddings = torch.stack(embeddings_list)
            mean_embedding = torch.mean(all_embeddings, dim=0)
            speaker_registry[speaker_name] = mean_embedding.cpu()

    # Save
    torch.save(speaker_registry, model_save_path)
    print(f"[Training] Registry updated. Total speakers: {len(speaker_registry)}")