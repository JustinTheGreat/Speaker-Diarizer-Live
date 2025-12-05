# stt_threshold.py
import os
import time
from faster_whisper import WhisperModel
# Removed socket import as it is no longer needed

from recorder_threshold import VADRecorder

class STTProcessor:
    def __init__(self, model_size="small"):
        # You can change device to "cuda" if you have an NVIDIA GPU set up
        self.device = "cpu" 
        self.compute_type = "float32" if self.device == "cuda" else "int8"
        print(f"\nLoading Whisper model ({model_size}) with compute_type='{self.compute_type}'...")
        print(f"Using device: {self.device}")
        try:
            self.model = WhisperModel(model_size, device=self.device, compute_type=self.compute_type)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def transcribe(self, audio_file_path, language="en", task="transcribe", beam_size=5):
        if not self.model:
            print("Model is not loaded.")
            return None

        if not os.path.exists(audio_file_path):
            print(f"Error: Audio file not found at {audio_file_path}")
            return None

        print(f"Transcribing audio from: {audio_file_path}")
        try:
            segments, info = self.model.transcribe(
                audio_file_path,
                language=language,
                task=task,
                beam_size=beam_size
            )
            transcribed_text = "".join(segment.text for segment in segments)
            print("Transcription complete!")
            return transcribed_text
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None

    def record_audio(self, filename="auto.wav"):
        print("\nThreshold-based VAD recording started.")
        vad = VADRecorder()
        return vad.record(output_wav=filename)

    # Removed send_to_wsl_server function

if __name__ == "__main__":
    wav_file = "auto.wav" 
    transcriber = STTProcessor()

    print("STT Processor initialized. Ready to record.")

    while True:
        try:
            # 1. Record Audio based on VAD
            recorded_file = transcriber.record_audio(wav_file)
            
            # 2. If a file was saved, transcribe it
            if recorded_file:
                transcription = transcriber.transcribe(recorded_file)
                
                if transcription:
                    print("\n--- Transcribed Text ---")
                    print(transcription)
                    print("------------------------")
                    
                    # Here is where you would add local logic if you wanted to 
                    # do something with the text (e.g., pass it to a local LLM variable)
            
            time.sleep(1) # Small delay before monitoring again
            
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt detected (Ctrl+C). Exiting...")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            time.sleep(1)