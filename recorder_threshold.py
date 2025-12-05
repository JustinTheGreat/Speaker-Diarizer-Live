import collections
import time
import numpy as np
import webrtcvad
import wavio
import sounddevice as sd

class VADRecorder:
    def __init__(
        self,
        fs: int = 16000,
        frame_duration_ms: int = 30,
        padding_duration_ms: int = 1000,   # Auto silence padding duration 2 seconds
        vad_mode: int = 1,
        min_record_duration_ms: int = 2000,
        amplitude_threshold: float = 0.05  # New parameter: RMS amplitude threshold
    ):
        """
        Args:
            fs (int): Sample rate. Must be one of 8000, 16000, 32000, or 48000.
            frame_duration_ms (int): Frame size in milliseconds. Must be 10, 20, or 30.
            padding_duration_ms (int): Duration of silence (in ms) to wait before stopping (default is 2000 ms).
            vad_mode (int): VAD aggressiveness (0–3, higher = more aggressive).
            min_record_duration_ms (int): Minimum total recording time before allowing silence to stop.
            amplitude_threshold (float): RMS amplitude threshold to start recording (normalized to -1.0 to 1.0).
        """
        self.fs = fs
        self.vad = webrtcvad.Vad(vad_mode)
        self.frame_ms = frame_duration_ms
        self.frame_size = int(fs * frame_duration_ms / 1000)  # in samples
        self.num_padding_frames = int(padding_duration_ms / frame_duration_ms)
        self.min_frames = int(min_record_duration_ms / frame_duration_ms)
        self.format = np.int16
        self.amplitude_threshold = amplitude_threshold

    def _frame_generator(self):
        """Yield raw 16-bit mono audio frames from the mic."""
        with sd.RawInputStream(samplerate=self.fs,
                               blocksize=self.frame_size,
                               dtype='int16',
                               channels=1) as stream:
            while True:
                data, _ = stream.read(self.frame_size)
                yield data

    def _calculate_rms(self, frame_data: bytes) -> float:
        """Calculate the Root Mean Square (RMS) amplitude of a frame."""
        # Convert bytes to numpy array of int16
        audio_array = np.frombuffer(frame_data, dtype=self.format)
        # Normalize to -1.0 to 1.0 range
        normalized_audio = audio_array / (2**15)
        rms = np.sqrt(np.mean(normalized_audio**2))
        return rms

    def record(self, output_wav: str = "vad_recording.wav") -> str:
        """
        Record until we detect padding_duration_ms of silence *after*
        at least min_record_duration_ms of audio.
        Starts recording if input sound is above amplitude_threshold.
        """
        # Buffer to hold frames for silence detection at the end
        silence_buffer = collections.deque(maxlen=self.num_padding_frames)
        # Buffer to hold frames for initial amplitude detection
        initial_detection_buffer = collections.deque(maxlen=self.num_padding_frames) # Use a buffer for initial detection too
        
        voiced_frames = []
        triggered = False
        frame_count = 0

        # input("Press Enter to start monitoring for sound.")
        print("Monitoring for sound level to start recording...")

        for frame in self._frame_generator():
            frame_count += 1
            current_rms = self._calculate_rms(frame)
            is_speech_vad = self.vad.is_speech(frame, sample_rate=self.fs)

            if not triggered:
                initial_detection_buffer.append(frame)
                
                # Check if any frame in the initial detection buffer is above the amplitude threshold
                if any(self._calculate_rms(f) > self.amplitude_threshold for f in initial_detection_buffer):
                    triggered = True
                    print(f"Sound above threshold ({self.amplitude_threshold:.4f}) detected – starting recording.")
                    # Add all frames from initial_detection_buffer (which contain the initial sound)
                    voiced_frames.extend(initial_detection_buffer)
                    initial_detection_buffer.clear()
                    start_time = time.time() # This variable is not used but kept for consistency
                else:
                    # If not triggered, clear old frames from initial_detection_buffer to prevent
                    # it from growing indefinitely before a trigger.
                    # This also ensures we don't accidentally include very old "silent" frames
                    # if the buffer maxlen is very large.
                    if len(initial_detection_buffer) == self.num_padding_frames:
                        initial_detection_buffer.popleft() 


            else:
                voiced_frames.append(frame)
                silence_buffer.append(frame) # Use silence_buffer for the trailing silence detection

                # Only allow silence-stop after minimum record time
                if frame_count >= self.min_frames:
                    # Check if *all* frames in the silence_buffer are considered non-speech by VAD
                    # This combines the level-based start with VAD-based end for robustness
                    if not any(self.vad.is_speech(f, self.fs) for f in silence_buffer):
                        print("Silence detected – stopping.")
                        break

        # concatenate
        audio_bytes = b"".join(voiced_frames)
        audio = np.frombuffer(audio_bytes, dtype=self.format)
        
        if len(audio) > 0: # Ensure there's audio to write
            wavio.write(output_wav, audio, self.fs, sampwidth=2)
            print(f"Saved VAD recording to {output_wav}")
        else:
            print("No audio recorded.")
            output_wav = None # Indicate no file was saved
        
        return output_wav

if __name__ == "__main__":
    # Example usage:
    # Set amplitude_threshold to a value that works for your microphone and environment.
    # A value between 0.005 and 0.05 is often a good starting point.
    recorder = VADRecorder(amplitude_threshold=0.2)
    while True:
        output_file = recorder.record("auto.wav")
        if output_file:
            print(f"Recording saved to: {output_file}")
            print("Ready to record again. Press Ctrl+C to exit.")
        else:
            print("No recording detected. Monitoring again...")