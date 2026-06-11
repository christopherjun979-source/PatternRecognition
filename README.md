# Real-Time Gesture Recognition Dashboard

A clean web dashboard that uses your webcam to recognize hand gestures in real time. It extracts face and hand tracking points using MediaPipe and runs them through a custom PyTorch AI model optimized with ONNX Runtime to make instant predictions directly inside your browser.

## Live Demo
Try the dashboard directly in your browser:
**https://christopherjun979-source.github.io/PatternRecognition/**

## What was used
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

---
*Note: This project is a work-in-progress, may not be 100% accurate at all times, and will be continuously updated in the future.*
