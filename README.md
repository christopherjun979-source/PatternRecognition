# Real-Time Gesture Recognition Dashboard

A clean web dashboard that uses your webcam to recognize hand gestures in real time. It extracts face and hand tracking points using MediaPipe and runs them through a custom PyTorch AI model optimized with ONNX Runtime to make instant predictions directly inside your browser.

## Tech Stack
- **AI Framework:** PyTorch (Model Training)
- **Computer Vision:** Google MediaPipe (Face & Hand Tracking)
- **Web Engine:** ONNX Runtime Web (Fast Web AI Inference)
- **Frontend:** Vanilla JavaScript, HTML5 Canvas, CSS3
- **Typography:** Geist Mono (Vercel)

## Project Files
- `index.html` — The frontend dashboard, UI, and live camera tracking system.
- `gesture_model.onnx` — Your trained AI model weights ready for browser use.
- `training_log.json` — Saved accuracy details parsed by the history chart.
- `collect_data.py` — Script to gather and save custom hand coordinate datasets.
- `train.py` — PyTorch script to train your neural network and export to ONNX.

## How to Run Locally

### 1. Start a Local Server
Because the app loads advanced browser modules and AI model assets, it must be run on a local server. Open your terminal in this project folder and run:
```bash
python -m http.server 8000
