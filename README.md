# ASTROX AI service

FastAPI inference service for the ASTROX UAV platform: YOLO + TensorRT
detection, ByteTrack multi-object tracking, multi-camera GStreamer
capture, MediaMTX/WebRTC (WHEP) video, and mission recording.

## Runs on
Jetson Orin Nano · JetPack 6 / L4T r36.x · Python 3.10

## Setup
- System deps (torch, tensorrt, GStreamer, PyGObject) come from JetPack / apt —
  see `requirements.txt` header and `README_CAMERA_PIPELINE.md` §5.
- `pip install -r requirements.txt` for the app-level packages.
- `cp cameras.yaml.example cameras.yaml` and edit for your cameras.
- `uvicorn app:app --host 0.0.0.0 --port 8000`

## Not in git
`venv/`, `best.engine`/`best.pt` (rebuild: `yolo export model=best.pt format=engine`),
`cameras.yaml`, and everything under `outputs/ uploads/ recordings/`.
