# 🎙️ Speaker Diarizer Live

A simple real-time **speaker diarization** tool built with [Pyannote Audio](https://github.com/pyannote/pyannote-audio).  
It captures live audio from your microphone, detects **who is speaking**, and displays speaker segments in real time.

---

## 🚀 Features
- 🎧 Live microphone streaming  
- 🗣️ Real-time speaker diarization (detects different speakers)  
- ⏱️ Processes short chunks of audio (default: 5 seconds)  
- 🧩 Easy to extend with speech-to-text or visualization

---

## 🧰 Requirements

Install dependencies:
```bash
pip install pyannote.audio torch sounddevice numpy

*P.S. make sure you have ffmpeg*

