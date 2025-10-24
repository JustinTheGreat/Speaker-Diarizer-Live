import sounddevice as sd
import numpy as np
from pyannote.audio import Pipeline
import queue
import threading
import time
import torch

# Initialize diarization model (replace with your token)
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

# Audio settings
SAMPLE_RATE = 16000
CHUNK_DURATION = 5  # seconds
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION

# Queue to hold audio data
audio_queue = queue.Queue()

def record_audio():
    """Continuously record audio and push chunks to a queue."""
    def callback(indata, frames, time, status):
        if status:
            print(status)
        audio_queue.put(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
        while True:
            time.sleep(CHUNK_DURATION)

def process_audio():
    """Continuously pull chunks from queue and run diarization."""
    buffer = np.array([], dtype=np.float32)
    while True:
        data = audio_queue.get()
        buffer = np.append(buffer, data.flatten())

        if len(buffer) >= CHUNK_SIZE:
            chunk = buffer[:CHUNK_SIZE]
            buffer = buffer[CHUNK_SIZE:]

            # Run diarization on this chunk
            print("\n--- Processing new chunk ---")
            diarization = pipeline({
                "waveform": torch.tensor(chunk, dtype=torch.float32).unsqueeze(0),
                "sample_rate": SAMPLE_RATE
            })

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                print(f"[{turn.start:.1f}s - {turn.end:.1f}s] Speaker {speaker}")

# Run in parallel threads
threading.Thread(target=record_audio, daemon=True).start()
threading.Thread(target=process_audio, daemon=True).start()

print("🎧 Live speaker diarization started... press Ctrl+C to stop.")
while True:
    time.sleep(0.5)
