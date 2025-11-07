import torch
import sounddevice as sd
import numpy as np
from pyannote.audio import Pipeline
from speechbrain.pretrained import EncoderClassifier
import queue, threading, time

# ==============================
# 🔊 CONFIGURATION
# ==============================
SAMPLE_RATE = 16000
CHUNK_DURATION = 5.0  # process every 5 seconds

# ==============================
# 🧠 GLOBAL MEMORY
# ==============================
known_speakers = {}  # {speaker_id: embedding_tensor}
next_speaker_id = 0
audio_buffer = np.zeros((0, 1), dtype=np.float32)
audio_queue = queue.Queue()

# ==============================
# 🎤 INITIALIZE PIPELINES
# ==============================
print("🔧 Loading models...")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")  # No token needed
speaker_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa"
)
print("✅ Models loaded successfully!\n")
print("🎧 Live speaker diarization & tracking started... press Ctrl+C to stop.\n")

# ==============================
# 🧩 FUNCTIONS
# ==============================

def get_embedding(audio_tensor):
    """Extract ECAPA-TDNN embedding from waveform tensor."""
    with torch.no_grad():
        emb = speaker_model.encode_batch(audio_tensor)
    return emb.squeeze()  # ensure it's 1D


def match_speaker(new_embedding, threshold=0.75):
    """Match or register a new speaker using cosine similarity."""
    global next_speaker_id

    # Normalize new embedding for stability
    new_embedding = torch.nn.functional.normalize(new_embedding, dim=0)

    if not known_speakers:
        sid = next_speaker_id
        known_speakers[sid] = new_embedding
        next_speaker_id += 1
        print(f"🆕 New speaker detected: Speaker {sid}")
        return sid

    best_match, best_score = None, -1.0
    for sid, emb in known_speakers.items():
        emb = torch.nn.functional.normalize(emb, dim=0)
        sim = torch.nn.functional.cosine_similarity(
            new_embedding.unsqueeze(0), emb.unsqueeze(0)
        ).item()
        if sim > best_score:
            best_score, best_match = sim, sid

    if best_score > threshold:
        print(f"🔁 Matched existing speaker: Speaker {best_match} (similarity={best_score:.2f})")
        return best_match
    else:
        sid = next_speaker_id
        known_speakers[sid] = new_embedding
        next_speaker_id += 1
        print(f"🆕 New speaker detected: Speaker {sid} (similarity={best_score:-+.2f})")
        return sid


def process_audio():
    """Continuously process queued audio chunks."""
    while True:
        chunk = audio_queue.get()
        if chunk is None:
            break

        try:
            waveform = torch.tensor(chunk.T, dtype=torch.float32)

            print("\n--- Processing new 5-second chunk ---")
            diarization = pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE})

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start = int(turn.start * SAMPLE_RATE)
                end = int(turn.end * SAMPLE_RATE)
                if end <= start or end > waveform.shape[1]:
                    continue

                segment_audio = waveform[:, start:end]
                if segment_audio.shape[1] < 4000:
                    continue

                emb = get_embedding(segment_audio)
                sid = match_speaker(emb)
                print(f"[{turn.start:.1f}s - {turn.end:.1f}s] 🗣️  Speaker {sid}\n")

        except Exception as e:
            print(f"⚠️ Error processing chunk: {e}")


def audio_callback(indata, frames, time_info, status):
    """Collect live audio and buffer until 5 seconds of data is ready."""
    global audio_buffer
    if status:
        print(f"⚠️ Audio status: {status}")

    # Append new audio samples
    audio_buffer = np.concatenate((audio_buffer, indata), axis=0)

    # Once we have 5 seconds (5 * 16000 samples)
    if len(audio_buffer) >= int(CHUNK_DURATION * SAMPLE_RATE):
        chunk = audio_buffer[: int(CHUNK_DURATION * SAMPLE_RATE)]
        audio_buffer = audio_buffer[int(CHUNK_DURATION * SAMPLE_RATE):]  # keep leftover
        audio_queue.put(chunk.copy())


# ==============================
# 🚀 MAIN LOOP
# ==============================
threading.Thread(target=process_audio, daemon=True).start()

with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, callback=audio_callback):
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n🛑 Stopping live diarization...")
        audio_queue.put(None)
