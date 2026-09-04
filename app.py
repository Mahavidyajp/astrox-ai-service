import asyncio
import logging
import traceback
import shutil
import uuid
import os

import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from aiortc import RTCPeerConnection, RTCSessionDescription

from detector import Detector
from video_detector import process_video
from camera import CameraManager, CameraError
from camera.manager import MISSION_RECORD_DIR
from camera.webrtc import (
    LatestFrameVideoTrack,
    PeerConnectionRegistry,
    build_rtc_configuration,
    new_consumer_id,
    wait_for_ice_gathering_complete,
)

logger = logging.getLogger("app")

app = FastAPI(title="YOLO TensorRT API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "best.engine"
)

detector = Detector(MODEL_PATH)


# ============================================================
# DIRECTORIES
# ============================================================

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MISSION_RECORD_DIR, exist_ok=True)


# ============================================================
# CAMERA MANAGER
# ============================================================

camera_manager = CameraManager(detector, config_path=os.getenv("CAMERAS_CONFIG_PATH", "cameras.yaml"))

webrtc_peers = PeerConnectionRegistry()


# ============================================================
# ROOT / HEALTH
# ============================================================

@app.get("/")
def root():
    return {"status": "running", "message": "YOLO TensorRT Backend Running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_PATH, "device": "Jetson"}


# ============================================================
# MODEL CLASSES  (dynamic -- the report / alert config MUST use this,
# never a hard-coded list; see detector.names)
# ============================================================

@app.get("/model/classes")
def model_classes():
    names = detector.names
    try:
        if isinstance(names, dict):
            classes = [names[k] for k in sorted(names)]
        else:
            classes = list(names)
    except Exception:
        classes = []
    return {"success": True, "classes": classes, "count": len(classes)}


# ============================================================
# IMAGE DETECTION
# ============================================================

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        print("===== IMAGE DETECTION =====")
        image_bytes = await file.read()
        print("Image Size:", len(image_bytes))

        result = detector.predict(image_bytes)

        if not result.get("success"):
            print(result)
            return result

        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return {"success": False, "error": "Could not decode uploaded image."}

        annotated = image.copy()
        detections = []
        class_counts: dict = {}

        for index, det in enumerate(result.get("detections", [])):
            object_id = index + 1

            det_with_id = {
                "track_id": object_id,
                "class": det["class"],
                "class_id": det["class_id"],
                "confidence": det["confidence"],
                "bbox": det["bbox"],
            }
            detections.append(det_with_id)
            class_counts[det["class"]] = class_counts.get(det["class"], 0) + 1

            x1, y1, x2, y2 = det["bbox"]

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{det['class']} #{object_id} {det['confidence']:.2f}"
            text_y = max(y1 - 10, 20)

            cv2.putText(
                annotated,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        uid = str(uuid.uuid4())
        output_filename = f"{uid}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        write_ok = cv2.imwrite(output_path, annotated)

        if not write_ok or not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            return {
                "success": False,
                "error": "Detection succeeded but the annotated output image failed to save.",
            }

        print(f"Annotated image saved: {output_path}")

        response = {
            "success": True,
            "mode": "image",
            "objects_detected": len(detections),
            "count": len(detections),
            "class_counts": class_counts,
            "detections": detections,
            "image_url": f"/image/download/{output_filename}",
        }
        print(response)
        return response

    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.get("/image/download/{filename}")
def download_image(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        return {"success": False, "error": "Processed image not found."}
    return FileResponse(path=file_path, media_type="image/jpeg", filename=filename)


# ============================================================
# VIDEO DETECTION
# ============================================================

@app.post("/video/detect")
async def detect_video(file: UploadFile = File(...)):
    try:
        print("===== VIDEO DETECTION STARTED =====")
        uid = str(uuid.uuid4())
        input_path = os.path.join(UPLOAD_DIR, uid + ".mp4")
        output_path = os.path.join(OUTPUT_DIR, uid + ".mp4")

        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print("Video Saved:", input_path)

        stats = process_video(input_path, output_path, detector=detector)

        print("Processing Finished")

        return {
            "success": True,
            "video": f"/video/download/{uid}.mp4",
            "stats": stats,
        }
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.get("/video/download/{filename}")
def download_video(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        return {"success": False, "error": "Processed video not found."}
    return FileResponse(path=file_path, media_type="video/mp4", filename=filename)


# ============================================================
# MISSION RECORDING DOWNLOAD  (Phase A)
# Files written by CameraManager's mission recorder; linked from the
# saved mission via mission.recording.video_url. Served here (not via
# StaticFiles) to match the existing FileResponse pattern and its
# HTTP Range support for <video> playback.
# ============================================================

@app.get("/webcam/recording/{filename}")
def download_recording(filename: str):
    safe = os.path.basename(filename)  # path-traversal guard
    file_path = os.path.join(MISSION_RECORD_DIR, safe)
    if not os.path.exists(file_path):
        return {"success": False, "error": "Recording not found."}
    return FileResponse(path=file_path, media_type="video/mp4", filename=safe)


# ============================================================
# CAMERA DISCOVERY
# ============================================================

@app.get("/webcam/cameras")
def webcam_cameras():
    try:
        cameras = camera_manager.discover_cameras()
        return {"success": True, "cameras": cameras}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e), "cameras": []}


# ============================================================
# MISSION CONFIG  (call BEFORE /webcam/start)
# ============================================================
# Sets the alert rules + recording flag for the NEXT mission. Rules
# are model-class-agnostic (see /model/classes). Recording ON now
# actually captures an annotated .mp4 via CameraManager (Phase A);
# video_available flips true on Stop once the file is finalized.

class AlertRuleIn(BaseModel):
    class_: str
    metric: str = "unique"        # "visible" | "unique"
    operator: str = ">="
    threshold: int = 1
    severity: str = "MEDIUM"
    enabled: bool = True

    class Config:
        fields = {"class_": "class"}
        populate_by_name = True


class MissionConfig(BaseModel):
    alert_rules: list[dict] = []
    recording: bool = False


@app.post("/webcam/session/config")
def webcam_session_config(cfg: MissionConfig):
    try:
        camera_manager.set_mission_config(
            alert_rules=cfg.alert_rules,
            recording=cfg.recording,
        )
        return {
            "success": True,
            "alert_rules": len(cfg.alert_rules),
            "recording": cfg.recording,
        }
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# ============================================================
# START CAMERA
# ============================================================

@app.get("/webcam/start")
def start_webcam(camera_id: str = Query(...)):
    try:
        info = camera_manager.start(camera_id)
        return {
            "success": True,
            "message": "Camera started",
            "camera_id": info["camera_id"],
            "camera_type": info["type"],
            "camera": info.get("device"),
            "resolution": info.get("resolution"),
        }
    except CameraError as e:
        camera_manager.stop()
        return {"success": False, "error": str(e)}
    except Exception as e:
        traceback.print_exc()
        camera_manager.stop()
        return {"success": False, "error": str(e)}


# ============================================================
# STOP CAMERA
# ============================================================

@app.get("/webcam/stop")
async def stop_webcam():
    try:
        await asyncio.to_thread(camera_manager.stop)
        await webrtc_peers.close_all()
        return {"success": True, "message": "Camera stopped"}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# ============================================================
# WEBCAM STATUS
# ============================================================

@app.get("/webcam/status")
def webcam_status():
    return camera_manager.get_status()


# ============================================================
# WEBCAM MISSION  (the current mission object -- same shape the
# frontend persists on Stop)
# ============================================================

@app.get("/webcam/mission")
def webcam_mission():
    try:
        return {"success": True, "mission": camera_manager.get_mission()}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# ============================================================
# CLEAR WEBCAM SESSION
# ============================================================

@app.post("/webcam/session/clear")
def clear_webcam_session():
    try:
        camera_manager.clear_session()
        return {"success": True, "message": "Session cleared"}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# ============================================================
# WEBCAM SETTINGS
# ============================================================

class WebcamSettings(BaseModel):
    confidence: float


@app.post("/webcam/settings")
def update_webcam_settings(settings: WebcamSettings):
    confidence = float(settings.confidence)
    if not 0.0 <= confidence <= 1.0:
        return {"success": False, "error": "Confidence must be between 0 and 1."}
    camera_manager.set_confidence(confidence)
    print("Confidence threshold:", confidence)
    return {"success": True, "confidence": confidence}


# ============================================================
# WEBCAM VIDEO — WebRTC offer (legacy aiortc path; the frontend now
# plays via MediaMTX WHEP. Kept so an old client doesn't 404.)
# ============================================================

class RTCOffer(BaseModel):
    sdp: str
    type: str


@app.post("/webcam/webrtc/offer")
async def webrtc_offer(offer: RTCOffer):
    if not camera_manager.is_running:
        return JSONResponse(
            {"success": False, "error": "Start a camera with /webcam/start before requesting a WebRTC stream."},
            status_code=400,
        )

    pc = RTCPeerConnection(configuration=build_rtc_configuration())
    webrtc_peers.add(pc)
    consumer_id = new_consumer_id()

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("WebRTC peer %s connection state: %s", consumer_id, pc.connectionState)
        if pc.connectionState in ("failed", "closed", "disconnected"):
            webrtc_peers.discard(pc)
            camera_manager.forget_video_consumer(consumer_id)
            try:
                await pc.close()
            except Exception:
                pass

    pc.addTrack(LatestFrameVideoTrack(camera_manager, consumer_id))

    try:
        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer.sdp, type=offer.type))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await wait_for_ice_gathering_complete(pc)
    except Exception as e:
        traceback.print_exc()
        webrtc_peers.discard(pc)
        camera_manager.forget_video_consumer(consumer_id)
        await pc.close()
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


# ============================================================
# DETECTIONS — WebSocket
# ============================================================

@app.websocket("/webcam/ws/detections")
async def webcam_detections_ws(websocket: WebSocket):
    await websocket.accept()
    queue = camera_manager.subscribe_detections()
    try:
        while True:
            payload = await queue.get()
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Detections WebSocket error")
    finally:
        camera_manager.unsubscribe_detections(queue)


# ============================================================
# STARTUP / SHUTDOWN
# ============================================================

@app.on_event("startup")
async def startup_event():
    camera_manager.set_event_loop(asyncio.get_running_loop())


@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down camera manager...")
    await webrtc_peers.close_all()
    await asyncio.to_thread(camera_manager.shutdown)
    print("Camera manager stopped.")
