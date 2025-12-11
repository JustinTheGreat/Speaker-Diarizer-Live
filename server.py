import os
import time
import asyncio
import threading
import json
import subprocess
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import torch
from dotenv import load_dotenv

# --- Imports ---
from recorder_threshold import VADRecorder
from whisperx_component import load_whisperx_models, process_audio_live
from llm_component import load_llm_pipeline, generate_response_live
from speaker_recognition_component import load_speaker_encoder, recognize_speakers_live
from speechbrain_component import extract_and_save_speaker_data
from speaker_training_component import train_speaker_model

# --- Global State ---
ml_models = {}
global_state = {
    "vad_thread": None,
    "is_listening": False,
    "vad_recorder": None,
    "event_loop": None
}

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        stats = get_speaker_stats()
        await websocket.send_json({"status_update": global_state['is_listening'], "stats": stats})

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections[:]:
            try:
                await connection.send_json(message)
            except:
                self.disconnect(connection)

manager = ConnectionManager()

# --- Helper: Get Speaker Stats ---
def get_speaker_stats(data_dir="speaker_data"):
    stats = {}
    if not os.path.exists(data_dir):
        return stats
    try:
        for name in os.listdir(data_dir):
            speaker_path = os.path.join(data_dir, name)
            if os.path.isdir(speaker_path):
                count = len([f for f in os.listdir(speaker_path) if f.endswith('.wav')])
                stats[name] = count
    except Exception as e:
        print(f"Error reading stats: {e}")
    return stats

# --- VAD Thread ---
def vad_listening_loop():
    recorder = global_state["vad_recorder"]
    loop = global_state["event_loop"]
    audio_filename = "live_server_buffer.wav"
    clean_filename = "live_server_clean.wav"
    print("[VAD Thread] Started.")
    
    while global_state["is_listening"]:
        file_path = recorder.record(audio_filename)
        if not global_state["is_listening"]: break

        if file_path:
            asyncio.run_coroutine_threadsafe(
                process_and_broadcast(file_path, clean_filename), loop
            )
        time.sleep(0.01)

    global_state["vad_thread"] = None
    asyncio.run_coroutine_threadsafe(manager.broadcast({"status_update": False}), loop)
    print("[VAD Thread] Exited.")

def start_vad_thread():
    if global_state["is_listening"]: return False
    global_state["is_listening"] = True
    thread = threading.Thread(target=vad_listening_loop, daemon=True)
    global_state["vad_thread"] = thread
    thread.start()
    return True

def stop_vad_thread():
    if not global_state["is_listening"]: return False
    global_state["is_listening"] = False
    return True

# --- Processing Pipeline ---
async def process_and_broadcast(raw_path, clean_path):
    start_time = time.time()
    device = ml_models["device"]

    try:
        # A. Clean Audio
        subprocess.run(["ffmpeg", "-y", "-i", raw_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", clean_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # B. Transcribe
        segments = process_audio_live(clean_path, ml_models["whisper"], device)
        if not segments: return

        # C. LLM Identification
        raw_transcript = format_transcript_for_llm(segments, {})
        prompt = f"Given the transcript below, identify speaker names. Return JSON format: {{\"SPEAKER_00\": \"Name\"}}\nTRANSCRIPT:\n{raw_transcript}"
        llm_resp = generate_response_live(ml_models["llm"], prompt)
        llm_map = parse_llm_json(llm_resp)

        # D. Biometric Identification
        bio_map = recognize_speakers_live(clean_path, segments, ml_models["encoder"], "speaker_registry.pt", 0.35, device)

        # E. MERGE
        final_decisions = {}

        # 1. Biometrics
        for spk_id, data in bio_map.items():
            if isinstance(data, dict):
                name = data["name"]
                score = f"{data['score']:.2f}"
            else:
                name = data
                score = "?"
            final_decisions[spk_id] = {"name": name, "method": "BIO", "score": score}

        # 2. LLM Override
        for spk_id, name in llm_map.items():
            is_generic = any(x in name.lower() for x in ["unknown", "speaker", "person"])
            if not is_generic:
                final_decisions[spk_id] = {"name": name, "method": "LLM", "score": None}

        # 3. Orphans
        detected_names = list(set(d["name"] for d in final_decisions.values()))
        dominant_speaker = detected_names[0] if len(detected_names) == 1 else None

        results = []
        final_map_simple = {} 

        for seg in segments:
            raw = seg.get("speaker", "UNKNOWN")
            display_name = raw
            method = ""
            score_display = ""

            if raw in final_decisions:
                info = final_decisions[raw]
                display_name = info["name"]
                method = info["method"]
                score_display = info["score"]
                final_map_simple[raw] = display_name
            elif raw == "UNKNOWN" and dominant_speaker:
                display_name = dominant_speaker
                method = "INF"
            
            results.append({
                "speaker": display_name,
                "text": seg["text"].strip(),
                "method": method,
                "score": score_display
            })

        # F. Save Data & Train (STRICTLY LLM OVERRIDES ONLY)
        # =======================================================
        llm_segments = []
        llm_map_subset = {}

        for s in segments:
            raw = s.get("speaker")
            # Only proceed if we have a decision for this speaker AND the method was LLM
            if raw in final_decisions and final_decisions[raw]["method"] == "LLM":
                llm_segments.append(s)
                llm_map_subset[raw] = final_decisions[raw]["name"]

        if llm_segments:
            print(f"[Training] Learning from LLM Override for: {list(llm_map_subset.values())}")
            extract_and_save_speaker_data(clean_path, llm_segments, llm_map_subset)
            train_speaker_model("speaker_data", "speaker_registry.pt", device, ml_models["encoder"])
        # =======================================================
        
        # G. Get Updated Stats
        current_stats = get_speaker_stats()

        # H. Broadcast
        await manager.broadcast({
            "segments": results,
            "stats": current_stats,
            "time": f"{time.time()-start_time:.2f}s"
        })

    except Exception as e:
        print(f"Error in pipeline: {e}")
    finally:
        if os.path.exists(raw_path): os.remove(raw_path)
        if os.path.exists(clean_path): os.remove(clean_path)

# --- Helpers ---
def format_transcript_for_llm(segments, m):
    t = ""
    for s in segments:
        n = m.get(s.get("speaker", "UNKNOWN"), s.get("speaker", "UNKNOWN"))
        t += f"[{n}]: {s['text'].strip()}\n"
    return t

def parse_llm_json(text):
    try:
        s = text.find('{'); e = text.rfind('}') + 1
        return json.loads(text[s:e]) if s != -1 else {}
    except: return {}

# --- Server Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n--- LOADING MODELS ON {device.upper()} ---\n")
    ml_models["whisper"] = load_whisperx_models(device, "float16" if device=="cuda" else "int8", token)
    ml_models["llm"] = load_llm_pipeline(device)
    ml_models["encoder"] = load_speaker_encoder(device)
    ml_models["device"] = device
    
    global_state["vad_recorder"] = VADRecorder(amplitude_threshold=0.02, min_record_duration_ms=1000)
    global_state["event_loop"] = asyncio.get_running_loop()
    
    print("\n--- SERVER READY ---\n")
    yield
    ml_models.clear()
    if global_state["vad_thread"]: global_state["is_listening"] = False 
    if device == "cuda": torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "start":
                if start_vad_thread(): await manager.broadcast({"status_update": True})
            elif data == "stop":
                if stop_vad_thread(): await manager.broadcast({"status_message": "Stopping..."})
    except: manager.disconnect(websocket)

@app.get("/")
async def get():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>AI Speaker Recognition</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { 
                font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
                background-color: #121212; 
                color: #e0e0e0; 
                height: 100vh; 
                overflow: hidden; 
                display: flex;
                flex-direction: column;
            }
            header {
                background: #1e1e1e;
                padding: 15px 25px;
                border-bottom: 1px solid #333;
                display: flex;
                align-items: center;
                justify-content: space-between;
                height: 70px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            }
            h1 { font-size: 1.2rem; font-weight: 600; color: #fff; letter-spacing: 0.5px; }
            .controls { display: flex; gap: 10px; align-items: center; }
            #status-dot { height: 10px; width: 10px; background: #e74c3c; border-radius: 50%; display: inline-block; margin-right: 8px; }
            #status-text { font-size: 0.9rem; font-weight: 600; color: #aaa; margin-right: 15px; }
            button { padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; font-weight: 600; font-size: 0.9rem; transition: background 0.2s; }
            .btn-start { background: #27ae60; color: white; }
            .btn-start:hover { background: #2ecc71; }
            .btn-stop { background: #c0392b; color: white; }
            .btn-stop:hover { background: #e74c3c; }
            button:disabled { opacity: 0.4; cursor: not-allowed; }
            #main-container { display: flex; flex: 1; gap: 20px; padding: 20px; overflow: hidden; }
            #chat-panel { flex: 7; background: #1e1e1e; border-radius: 12px; display: flex; flex-direction: column; border: 1px solid #333; }
            .panel-header { padding: 15px; border-bottom: 1px solid #333; font-weight: 600; color: #888; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; }
            #chat-box { flex: 1; overflow-y: auto; padding: 20px; scroll-behavior: smooth; }
            .msg { margin-bottom: 20px; padding: 15px; border-radius: 8px; background: #252525; border-left: 4px solid #555; animation: fadeIn 0.3s ease-out; }
            @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
            .msg-header { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
            .speaker-name { font-weight: 700; color: #fff; font-size: 1rem; }
            .msg-text { font-size: 1.05rem; line-height: 1.5; color: #ddd; }
            .msg.Justin { border-left-color: #3498db; background: #1a2630; }
            .msg.Kristen { border-left-color: #9b59b6; background: #241a29; }
            .msg.Unknown { border-left-color: #e74c3c; }
            .badge { font-size: 0.7rem; padding: 2px 6px; border-radius: 4px; font-weight: bold; text-transform: uppercase; }
            .badge-llm { background: #8e44ad; color: #fff; }
            .badge-bio { background: #2980b9; color: #fff; }
            .badge-inf { background: #7f8c8d; color: #fff; }
            #stats-panel { flex: 3; background: #1e1e1e; border-radius: 12px; display: flex; flex-direction: column; border: 1px solid #333; }
            #stats-list { flex: 1; overflow-y: auto; padding: 15px; }
            .stat-card { background: #2b2b2b; padding: 12px 15px; border-radius: 8px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; border: 1px solid #383838; }
            .stat-name { font-weight: 600; color: #fff; }
            .stat-count-box { background: #333; padding: 4px 10px; border-radius: 12px; color: #2ecc71; font-weight: bold; font-size: 0.9rem; }
            .empty-stats { color: #555; text-align: center; margin-top: 20px; font-style: italic; }
        </style>
    </head>
    <body>
        <header>
            <h1>🎙️ Live Identification System</h1>
            <div class="controls">
                <span id="status-dot"></span>
                <span id="status-text">Ready</span>
                <button id="btnStart" class="btn-start" onclick="send('start')">Start</button>
                <button id="btnStop" class="btn-stop" onclick="send('stop')" disabled>Stop</button>
            </div>
        </header>
        <div id="main-container">
            <div id="chat-panel">
                <div class="panel-header">Live Transcript</div>
                <div id="chat-box">
                    <div style="color: #555; text-align: center; margin-top: 50px;">Waiting for audio...</div>
                </div>
            </div>
            <div id="stats-panel">
                <div class="panel-header">Speaker Registry</div>
                <div id="stats-list">
                    <div class="empty-stats">Loading registry...</div>
                </div>
            </div>
        </div>
        <script>
            const ws = new WebSocket((window.location.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + window.location.host + '/ws');
            const chatBox = document.getElementById("chat-box");
            const statsList = document.getElementById("stats-list");
            const statusText = document.getElementById("status-text");
            const statusDot = document.getElementById("status-dot");
            const btnStart = document.getElementById("btnStart");
            const btnStop = document.getElementById("btnStop");
            let firstMessage = true;

            function send(cmd) { ws.send(cmd); }
            function updateControls(listening) {
                btnStart.disabled = listening;
                btnStop.disabled = !listening;
                if (listening) {
                    statusText.innerText = "Listening...";
                    statusDot.style.background = "#2ecc71";
                    statusDot.style.boxShadow = "0 0 8px #2ecc71";
                } else {
                    statusText.innerText = "Stopped";
                    statusDot.style.background = "#e74c3c";
                    statusDot.style.boxShadow = "none";
                }
            }
            function renderStats(stats) {
                statsList.innerHTML = "";
                if (Object.keys(stats).length === 0) {
                    statsList.innerHTML = '<div class="empty-stats">No speakers registered yet.</div>';
                    return;
                }
                const names = Object.keys(stats).sort();
                names.forEach(name => {
                    const count = stats[name];
                    const div = document.createElement("div");
                    div.className = "stat-card";
                    div.innerHTML = `<span class="stat-name">${name}</span><div class="stat-count-box">${count} recs</div>`;
                    statsList.appendChild(div);
                });
            }
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.status_update !== undefined) updateControls(data.status_update);
                if (data.status_message) statusText.innerText = data.status_message;
                if (data.stats) renderStats(data.stats);
                if (data.segments) {
                    if (firstMessage) { chatBox.innerHTML = ""; firstMessage = false; }
                    data.segments.forEach(seg => {
                        const div = document.createElement("div");
                        let badgeHtml = "";
                        if (seg.method === "LLM") badgeHtml = `<span class="badge badge-llm">LLM OVERRIDE</span>`;
                        else if (seg.method === "BIO") {
                            let score = seg.score ? `(${seg.score})` : "";
                            badgeHtml = `<span class="badge badge-bio">BIO ${score}</span>`;
                        }
                        else if (seg.method === "INF") badgeHtml = `<span class="badge badge-inf">INFERRED</span>`;
                        div.className = `msg ${seg.speaker}`;
                        div.innerHTML = `<div class="msg-header"><span class="speaker-name">${seg.speaker}</span>${badgeHtml}</div><div class="msg-text">${seg.text}</div>`;
                        chatBox.appendChild(div);
                        chatBox.scrollTop = chatBox.scrollHeight;
                    });
                }
            };
        </script>
    </body>
    </html>
    """)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)