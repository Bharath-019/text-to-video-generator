# 🎬 Text to Video Generator
**Final Year Project — Bharath Kumar | East West Institute of Technology**
*MindMatrix VTU Internship Program*

An AI-powered system that converts any text topic or script into a complete educational video with narration, subtitles, and AI-generated visuals.

---

## 🚀 Features
- **Narration Video** — Topic/script → captions → images → voice → subtitles → MP4
- **AI Animation** — Text prompt → local diffusion model → animated video
- **Fallback** — If AI animation fails, auto-switches to image slideshow
- **100% Local** — Runs on your own machine (RTX 4060 GPU supported)

---

## 🛠️ Tech Stack
| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask |
| Frontend | HTML, CSS, JavaScript |
| AI/Video | Diffusers (HuggingFace), PyTorch |
| Speech | gTTS (narration), Whisper (subtitles) |
| NLP | KeyBERT (keyword extraction) |
| Captions | MetaAI API |
| Images | Pexels API |
| Video | MoviePy, Pillow |

---

## ⚙️ Setup Instructions

### 1. Prerequisites
- Python 3.9+
- CUDA-compatible GPU (RTX 4060 recommended)
- [ImageMagick](https://imagemagick.org/script/download.php) installed at:
  `C:\Program Files\ImageMagick-7.1.2-Q16\magick.exe`
- [FFmpeg](https://ffmpeg.org/download.html) added to PATH

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ For PyTorch with CUDA support, install from https://pytorch.org/get-started/locally/
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

### 3. Project Structure
```
text_to_video/
├── app.py              ← Main Flask backend
├── requirements.txt    ← Python dependencies
├── static/
│   └── index.html      ← Frontend UI
├── videos/             ← Generated output videos
├── images/             ← Temp downloaded images
├── audio/              ← Temp generated audio
└── generated_clips/    ← Temp AI animation clips
```

### 4. Run the App
```bash
python app.py
```
Then open your browser at: **http://localhost:5000**

---

## 🎮 How to Use

### Narration Video
1. Click **"Narration Video"** tab
2. Type a topic (e.g. *"Photosynthesis"*) or paste a full paragraph
3. Click **Generate Narration Video**
4. Wait ~1–3 minutes — video will appear when ready
5. Download your MP4!

### AI Animation
1. Click **"AI Animation"** tab
2. Describe your animation (e.g. *"A glowing solar system with planets orbiting"*)
3. Set the duration (5–30 seconds)
4. Click **Generate Animation**
5. Progress updates automatically — video appears when ready

---

## 🔑 API Keys
- **Pexels API** — already in `app.py` (replace if expired)
- **MetaAI** — no key needed (uses `meta-ai-api` package)

---

## ⚠️ Notes
- First animation run will download the T2V model (~3GB) automatically
- ImageMagick path in `app.py` must match your installation
- `videos/`, `images/`, `audio/` folders are auto-created on startup

---

## 👨‍💻 Developer
**Bharath Kumar**
- GitHub: [github.com/Bharath-019](https://github.com/Bharath-019)
- LinkedIn: [linkedin.com/in/bharath-kumar-219b47296](https://linkedin.com/in/bharath-kumar-219b47296)
