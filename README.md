# TTS
VoiceCraft AI Studio is an open-source, web-based neural audio suite offering ElevenLabs-grade Text-to-Speech (TTS), Zero-Shot Voice Cloning, and Speech-to-Speech Accent Conversion. Features real-time voice companions, high-fidelity waveform visualizers, fine-grained voice tuning, and seamless integration with WebGPU, XTTSv2, and Gemini AI APIs.
🎙️ VoiceCraft AI Studio Pro v3.2
VoiceCraft AI Studio is a production-ready, browser-based neural audio application designed to rival platforms like ElevenLabs. It combines state-of-the-art Text-to-Speech (TTS), Zero-Shot Voice Cloning, Speech-to-Speech Accent Shifting, and Real-Time Conversational AI into a sleek, dark-mode interface.
✨ Features At A Glance
1. 📢 Ultra-Realistic Text-to-Speech (TTS)
ElevenLabs-Style Parameter Tuning:
Stability Slider: Controls expression variability vs. monotonous stability.
Clarity + Similarity Enhancement: Boosts audio fidelity and neural reconstruction quality.
Style Exaggeration: Amplifies emotional dynamics and pitch inflection.
Pacing & Pitch Shift: Granular controls for semi-tone shift and speaking rate (0.85x – 1.30x).
Cinematic & Podcast Samplers: Pre-loaded prompt templates for immediate testing.
Paragraph Progress Tracking: Visual indication during multi-paragraph audio synthesis.
2. 🧬 Zero-Shot Voice Cloning Engine
Flexible Reference Capture: Upload WAV/MP3 files or record reference audio (10s–60s) directly via browser mic.
Acoustic Feature Extraction: Real-time extraction and visualization of:
Fundamental F0 Pitch (Hz)
Spectral Centroid Density
Formant Frequencies (F1/F2)
512-dimensional Float32 Neural Embeddings
Custom Profile Management: Save cloned profiles and instantly inject them into the TTS voice selection pipeline.
3. 🌐 Speech-to-Speech & Accent Conversion
Cadence Preservation: Retains original speaker rhythm and emotional contours while converting the voice model.
Accent Shift Engine: Seamlessly map accents (e.g., convert Indian, Spanish, or French accented speech to US General or British Received Pronunciation).
Automatic Speech Transcription: Inbuilt transcription display for source audio verification.
4. 🎧 Real-Time Conversational AI Companion
Interactive Live Companion Mode: Interactive voice assistant (inspired by Call Annie / Live Companion apps).
Dynamic Pulsing Orb Visualizer: Audio-reactive visual feedback during active conversation streams.
Live Dialogue Stream Log: Real-time text transcript of ongoing user-AI voice dialogue.
5. 📊 Audio Management & Visualization
Live Waveform Visualizer: Canvas-rendered amplitude waveforms for active playback.
Audio Library & Export: History log with one-click .WAV or .MP3 downloads and metadata inspection.
🛠️ Architecture & Backend Options
VoiceCraft AI Studio is architected to be modular, supporting three flexible backend options:
🚀 Quick Start Guide
1. Running the Frontend
Because VoiceCraft AI Studio is packaged as a single-file web application (index.html), no node build process or compiler is required.
Open index.html in any modern web browser (Chrome, Edge, Brave, or Firefox recommended).
(Optional) Click the Sliders/Settings icon in the header to enter your custom API key or local backend URL.
2. Setting Up a Local Python Backend (Coqui XTTSv2)
For production-grade zero-shot voice cloning with <3 second sample latency, run a local Python FastAPI backend using Coqui XTTSv2:
Prerequisite Installation
pip install fastapi uvicorn torch TTS python-multipart
main.py Backend Script
import os
import uuid
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from TTS.api import TTS

app = FastAPI(title="VoiceCraft Local Engine")

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Coqui XTTSv2 Model onto GPU
print("Loading XTTSv2 Model onto CUDA...")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

@app.post("/api/v1/clone-tts")
async def clone_tts(
    text: str = Form(...),
    speaker_wav: UploadFile = File(None),
    language: str = Form("en")
):
    out_path = f"output_{uuid.uuid4().hex}.wav"
    
    if speaker_wav:
        ref_path = f"ref_{uuid.uuid4().hex}.wav"
        with open(ref_path, "wb") as f:
            f.write(await speaker_wav.read())
        
        tts.tts_to_file(
            text=text,
            speaker_wav=ref_path,
            language=language,
            file_path=out_path
        )
        if os.path.exists(ref_path):
            os.remove(ref_path)
    else:
        # Default fallback voice
        tts.tts_to_file(text=text, speaker_name="Ana Florence", language=language, file_path=out_path)

    return FileResponse(out_path, media_type="audio/wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    Launch the Server
    python main.py
    Once running, set the backend in VoiceCraft Studio to http://localhost:8000/api/v1/clone-tts.
3. Setting Up In-Browser WebGPU Execution (Transformers.js)
To run SpeechT5 or VITS models completely inside the client's browser without sending audio data to any server, add the @xenova/transformers library script:
https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0
import { pipeline } from '@xenova/transformers';

// Initialize pipeline with WebGPU execution provider
const synthesizer = await pipeline('text-to-speech', 'Xenova/speecht5_tts', {
    quantized: true,
    device: 'webgpu'
});

// Synthesize audio
const output = await synthesizer(text, { speaker_embeddings: speakerVector });
🎨 Technology Stack
Frontend Framework: Vanilla JS (ES6+), HTML5, Web Audio API
Styling & Icons: Tailwind CSS CDN, FontAwesome 6 Pro
Audio Processing: PCM16 Binary Encoder, HTML5 Canvas Visualizer
Inference Endpoints: Google Gemini Neural API, Coqui XTTSv2 (FastAPI), Hugging Face Transformers.js
📜 License
Distributed under the MIT License. Feel free to fork, modify, and integrate into your own projects!
