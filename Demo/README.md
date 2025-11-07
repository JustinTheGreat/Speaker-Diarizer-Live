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

## References
- DIART
- Speechbrain

*P.S. make sure you have ffmpeg*

Running speaker verification through speech brain with text based speaker validation based off chunks, combines all the chunks together on occasion to do speaker validation on the occasion based from speaker diarization

venv\Scripts\activate 

# Diarizing Methods
- Live Chunking based of VAD and total recorded time for chunking threshold
- First utilizes text based manuscript to diarize/label and match speakers
- Matches to  most similar current speakers in database if no name is found based off of a certain threshold 
- Else labeled as speaker
- Database uses ML clustering algorithms to adjust voice profiles to their labels/make sure that the voice profile is a stand-alone
