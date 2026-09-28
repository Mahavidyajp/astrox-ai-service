# ASTROX AI Service

FastAPI inference backend for the ASTROX UAV platform. It runs a supplied
Ultralytics YOLO model, using TensorRT engines on Jetson, for image and video
detection, ByteTrack tracking, live multi-camera inference, mission recording,
and system telemetry.

## Platform and prerequisites

- NVIDIA Jetson Orin Nano running JetPack 6 (L4T r36.x).
- Python 3.10, with the JetPack-compatible Python packages for PyTorch,
  torchvision, TensorRT, and OpenCV (`cv2`). These are supplied by NVIDIA or
  apt; do not replace them with generic PyPI builds.
- GStreamer 1.0 and the camera/video plugins. CSI capture also needs the
  JetPack GStreamer elements such as `nvarguscamerasrc`; V4L2 tools help with
  USB/HDMI cameras.
- A compatible model file supplied separately; see [Model](#model).
- For `aiortc`/PyAV, the FFmpeg, SSL, Opus, and VPX development headers listed
  in `setup-jetson.sh` may be needed when PyAV must build from source.

`setup-jetson.sh` installs system packages with apt, including GStreamer
plugins and PyAV build headers. Its final printed pip commands are not a safe
one-size-fits-all Jetson recipe: `requirements.txt` includes `ultralytics`,
whose transitive requirements can install or shadow packages provided by
JetPack. Do not blindly run `pip install -r requirements.txt`, nor assume that
running it before `pip install --no-deps ultralytics ...` undoes any such
replacement.

### Python environment and dependencies

First install/verify the JetPack-provided Python stack for this JetPack image.
If using a virtual environment, make the JetPack Python packages visible to it:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python --version
```

Review `requirements.txt` before installing. It lists the application's
top-level Python packages, but it is not a JetPack-safe lockfile. Install only
after checking the target's existing JetPack package versions and how pip will
resolve dependencies. In particular, protect `torch`, `torchvision`,
`tensorrt`, `cv2`/OpenCV, and the JetPack-compatible `numpy` from replacement.
For example, install the application's other declared top-level packages
without dependency resolution, deliberately leaving out `numpy` so pip cannot
replace the JetPack build:

```bash
python -m pip install --no-deps \
  "fastapi>=0.110" "uvicorn[standard]>=0.29" \
  "python-multipart>=0.0.9" "pydantic>=2.6" "PyYAML>=6.0" \
  "ultralytics>=8.2" "ultralytics-thop>=2.0" "aiortc>=1.9.0"
```

This is a conservative starting point, not a complete dependency installer:
`--no-deps` does **not** install missing transitive dependencies. Check the
imports below and use `python -m pip check` to identify unmet requirements;
install any missing, non-JetPack dependencies deliberately, verifying versions
and avoiding packages that shadow JetPack libraries. `requirements.txt` remains
the project's declared top-level package list, but avoid using it as-is on
Jetson because its `numpy` entry is also a direct pip install request.
`ultralytics` and `ultralytics-thop` are included above with dependency
resolution disabled so pip does not fetch replacement PyTorch or other
transitive packages. The correct dependency set can vary with the JetPack
image and existing environment, so validate on the target rather than relying
on an unqualified requirements install.

The GStreamer/PyGObject bindings (`gi`) and Jetson camera elements are system
packages, not pip dependencies. `setup-jetson.sh` is the repository's apt
setup helper; inspect it and skip packages already supplied by JetPack if
appropriate.

## Model

The service does not download or create a model. Supply a model compatible
with the installed Ultralytics and TensorRT stack. By default, startup loads:

```text
models/best.engine
```

`app.py` reads `MODEL_PATH` once and constructs `Detector` while the module is
imported. A missing or invalid model, or an incompatible runtime, prevents
application startup; there is no fallback to another model. To use another
location, set `MODEL_PATH` in the process environment before launching Uvicorn.
For example:

```bash
export MODEL_PATH=/opt/astrox/models/best.engine
```

If starting from trained weights, exporting an engine may be possible with
`yolo export model=best.pt format=engine`, but export is not performed by this
service and engine compatibility depends on the target Jetson/TensorRT
environment.

## Install and run

From the repository root, with the model in place and the Python/GStreamer
environment prepared:

```bash
cp cameras.yaml.example cameras.yaml
# Edit cameras.yaml for this installation; see Camera configuration below.
uvicorn app:app --host 0.0.0.0 --port 8000
```

The process loads the model during import/startup. Binding to `0.0.0.0` makes
the API reachable on the network; restrict access with a private network and
firewall rules. FastAPI's interactive API documentation is available at
`/docs` while the service is running.

Useful import and hardware checks:

```bash
python -c "import cv2, numpy, torch, tensorrt, ultralytics; print('inference imports OK')"
python -c "import aiortc, av; print('aiortc', aiortc.__version__)"
gst-inspect-1.0 v4l2src
gst-inspect-1.0 nvarguscamerasrc
v4l2-ctl --list-devices
```

The `nvarguscamerasrc` check applies to CSI camera support; a missing element
may not matter for a deployment using only USB or RTSP cameras.

## Camera configuration

Copy `cameras.yaml.example` to `cameras.yaml` in the service working directory.
The default path is `cameras.yaml`; set `CAMERAS_CONFIG_PATH` to use another
file. A missing file is allowed and results in USB/HDMI auto-discovery only.
Configuration is reloaded for camera discovery/start operations.

Supported sources:

- **USB/HDMI V4L2**: discovered automatically when `auto_discover_usb: true`;
  per-device overrides can be placed in `usb_cameras`.
- **CSI**: explicitly list entries under `csi_cameras` with a sensor ID and
  capture settings.
- **RTSP/IP**: list entries under `rtsp_cameras`, including the stream URL,
  codec, latency, and transport.
- **Custom GStreamer**: explicitly define pipelines under `custom_cameras`.

The RTSP test endpoint probes a stream without saving it. RTSP connect saves a
camera entry for later discovery/start. **Credentials supplied to RTSP connect
are embedded in the URL and persisted in `cameras.yaml` as plaintext.** Keep
that file private, restrict its permissions, and do not commit it or expose it
in logs or support bundles. Prefer supplying credentials through a secure
deployment mechanism where the deployment allows it; the example's suggested
environment-variable URL is a YAML example, not automatic variable
substitution by the service.

## API and streaming

The principal routes are:

| Route | Purpose |
|---|---|
| `GET /`, `GET /health` | Basic running/health response; `/health` reports the configured model path and device label. |
| `GET /model/classes` | Class names from the loaded model. |
| `POST /detect` | Run image detection; returns detections and an annotated-image URL. |
| `GET /image/download/{filename}` | Download a processed image from `outputs/`. |
| `POST /video/detect` | Process an uploaded video with tracking; returns statistics and a processed-video URL. |
| `GET /video/download/{filename}` | Download a processed video from `outputs/`. |
| `GET /webcam/cameras` | Discover configured and available cameras. |
| `POST /webcam/rtsp/test`, `POST /webcam/rtsp/connect` | Probe an RTSP stream; save a tested RTSP camera configuration. |
| `POST /webcam/session/config`, `POST /webcam/settings` | Configure the next webcam mission or set the manual confidence threshold. |
| `GET /webcam/start?camera_id=...`, `GET /webcam/stop` | Start or stop the active camera session. |
| `GET /webcam/status`, `GET /webcam/mission` | Read camera/inference status and mission state. |
| `POST /webcam/session/clear` | Clear the webcam session state. |
| `POST /webcam/webrtc/offer` | Exchange a WebRTC SDP offer for a direct aiortc video stream; a camera must already be started. |
| `WS /webcam/ws/detections` | Receive detection updates separately as JSON over WebSocket. |
| `GET /webcam/recording/{filename}` | Download a mission recording, when recording was enabled. |
| `GET /system/stats` | Read CPU, memory, GPU, temperature, network, and uptime telemetry where available. |

Image/video uploads and processed outputs use the local `uploads/` and
`outputs/` directories. Mission recordings default to `recordings/`; set
`MISSION_RECORD_DIR` to change the location. `GET /system/stats` reads Jetson
sysfs and network counters; some measurements can be `null` or unavailable on
other systems. CPU/RAM/swap/uptime details also depend on optional `psutil`.

### Direct WebRTC and optional MediaMTX

`POST /webcam/webrtc/offer` is a direct aiortc WebRTC SDP offer/answer path.
Video is delivered by the service, while detections are a separate
`/webcam/ws/detections` JSON stream. Optional ICE settings are
`WEBRTC_ICE_URLS` (comma-separated), `WEBRTC_ICE_USERNAME`, and
`WEBRTC_ICE_CREDENTIAL`.

Separately, the camera manager also tries to publish raw camera frames to
MediaMTX over RTSP (default `rtsp://127.0.0.1:8554/cam`; override with
`MEDIAMTX_RTSP_URL`). MediaMTX is **not** started by Uvicorn or installed by
`setup-jetson.sh`; it must be deployed and configured separately. The supplied
`config/mediamtx/mediamtx.yml` is an example configuration, not an automatic
service setup. This publisher path supports MediaMTX's browser-facing WebRTC
(WHEP) separately from the direct aiortc endpoint. If GStreamer is unavailable
or MediaMTX cannot be reached, publishing is disabled/retried; inference is
designed to continue.

## Repository layout

| Path | Contents |
|---|---|
| `app.py` | FastAPI app, inference and camera routes, WebRTC signaling, telemetry. |
| `detector.py`, `video_detector.py`, `bytetrack.yaml` | YOLO inference, video processing, and ByteTrack configuration. |
| `camera/` | Camera discovery/sources, frame buffering, inference manager, mission recording, aiortc, and MediaMTX publishing. |
| `models/` | Expected location for a manually supplied model; model files are not included. |
| `cameras.yaml.example` | Starter camera configuration. |
| `setup-jetson.sh`, `requirements.txt` | Jetson system package helper and Python top-level dependencies. |
| `config/mediamtx/mediamtx.yml` | Example configuration for a separately run MediaMTX instance. |
| `uploads/`, `outputs/`, `recordings/` | Runtime upload, processed-output, and mission-recording directories. |
| `README_CAMERA_PIPELINE.md` | Detailed camera/WebRTC pipeline notes and hardware checks. |

## Security and operations

The API has **no authentication or authorization**, and CORS allows requests
from any origin (without credentials). Do not expose it directly to the
internet or an untrusted network. Keep it behind a trusted private network,
firewall, or authenticated reverse proxy; protect camera configuration and
recordings as sensitive data.

The service uses one active camera source per process. Video processing and
live inference share the loaded model/tracker, so concurrent tracking work
shares state and is serialized by the detector. Ensure there is sufficient
disk space for uploads, processed files, and mission recordings, and manage
their retention externally. The service creates runtime directories as needed;
it does not provide a retention policy.

## Troubleshooting

- **Uvicorn fails before serving requests:** check that `MODEL_PATH` points to
  a readable, valid model and that TensorRT, PyTorch, Ultralytics, and model
  engine versions are compatible. Model loading occurs at import/startup.
- **Import fails or JetPack inference stops working after pip install:** check
  whether pip installed or shadowed JetPack packages. Restore the versions
  appropriate to the installed JetPack image and redo dependency installation
  conservatively; do not use generic PyPI `torch`, `torchvision`, TensorRT, or
  OpenCV packages for this stack.
- **`aiortc`/`av` fails to install/import:** install the `libav*-dev`,
  `libssl-dev`, `libopus-dev`, and `libvpx-dev` packages from
  `setup-jetson.sh` before building PyAV; then verify with the import check.
- **Camera is not listed or will not start:** inspect `cameras.yaml`, check
  `/dev/video*` access and `v4l2-ctl --list-devices` for V4L2 cameras, verify
  CSI GStreamer plugins for CSI cameras, and test RTSP reachability, URL,
  codec, and TCP/UDP settings for IP cameras.
- **Direct WebRTC connects locally but not across networks:** configure
  reachable STUN/TURN servers through the `WEBRTC_ICE_*` settings and ensure
  firewall/NAT rules permit the required traffic.
- **MediaMTX video is unavailable but detection works:** confirm the separate
  MediaMTX process, its RTSP listener, GStreamer plugins (including
  `openh264enc`), and `MEDIAMTX_RTSP_URL`. This is a distinct publishing path
  from `/webcam/webrtc/offer`.
