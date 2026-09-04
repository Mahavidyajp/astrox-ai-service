# ASTROX AI — Multi-Camera Low-Latency Pipeline (WebRTC)

## 1. Architecture

```
                       ┌───────────────────────────────────────────┐
                       │              CameraManager                │
                       │                                           │
discover_cameras() ────┼──> discovery.py (v4l2-ctl + cameras.yaml) │
                       │                                           │
start(camera_id) ──────┼──> builds one CameraSource:               │
                       │      USB/HDMI -> V4L2CameraSource         │
                       │      CSI      -> CSICameraSource          │
                       │      RTSP     -> RTSPCameraSource         │
                       │      CUSTOM   -> CustomCameraSource       │
                       │        │                                  │
                       │        ▼ (GStreamer appsink callback)     │
                       │   LatestFrameBuffer (1 slot, drop-old,    │
                       │   multiple independent consumers)         │
                       │        │                    │             │
                       │        │ (raw frames)        │ (raw frames)│
                       │        ▼                    ▼             │
                       │  inference worker    LatestFrameVideoTrack │
                       │  thread              (one per WebRTC       │
                       │   │                   viewer)              │
                       │   ▼ detector.predict_frame() [existing]   │
                       │   │                                        │
                       │   ▼ publish (asyncio queue, latest-only)   │
                       └───┼────────────────────────┼──────────────┘
                           │                         │
             GET /webcam/status (poll)               │ WebRTC H.264
             WS  /webcam/ws/detections (push)         │ (raw video,
                           │                          │  no boxes baked in)
                           ▼                          ▼
                  <canvas> overlay              <video> element
                  (drawn client-side,           (native browser
                   positioned onto the           decode, hardware-
                   video's displayed box)         accelerated)
```

**Video and detections are two independent transports** that share
only the camera capture stage:

- **Video** — `LatestFrameVideoTrack` (an aiortc `VideoStreamTrack`)
  reads raw, undecorated frames straight off the active
  `CameraSource`'s buffer and streams them over WebRTC as H.264. It
  never calls the detector and never waits on it.
- **Detections** — the inference worker thread (unchanged: still one
  `detector.predict_frame()` call per frame, same as before) publishes
  each result over `/webcam/ws/detections`. It never touches video
  frames or encoding.

Both stages read the *same* `LatestFrameBuffer`, but as of this
change that buffer supports multiple independent consumers (see
`frame_buffer.py`): each WebRTC viewer and the inference worker each
get "the newest frame I haven't already read yet", non-blocking for
WebRTC viewers, blocking-with-timeout for the inference worker —
never a shared queue, never a backlog, at any stage. That's the whole
latency fix, same as before, now applied to two transports instead of
one MJPEG stream.

## 2. Files

**Created**
- `ai-service/camera/__init__.py`
- `ai-service/camera/frame_buffer.py` — latest-frame buffer, now multi-consumer
- `ai-service/camera/base.py` — `CameraSource`, `GStreamerSource`, `CameraInfo`, `CameraError`
- `ai-service/camera/v4l2_source.py` — USB + HDMI (fallback pipeline chain)
- `ai-service/camera/csi_source.py` — Jetson CSI (`nvarguscamerasrc`)
- `ai-service/camera/rtsp_source.py` — RTSP/IP cameras
- `ai-service/camera/custom_source.py` — raw-pipeline escape hatch
- `ai-service/camera/discovery.py`
- `ai-service/camera/config.py`
- `ai-service/camera/manager.py` — `CameraManager` (adds raw-frame access + detections pub/sub)
- `ai-service/camera/webrtc.py` — **new**: `LatestFrameVideoTrack`, ICE/signaling helpers, `PeerConnectionRegistry`
- `ai-service/cameras.yaml.example`
- `ai-service/requirements-camera.txt`
- `frontend/src/routes/webcam.tsx` — `<video>` + WebRTC + canvas overlay + detections WebSocket

**Modified**
- `ai-service/app.py` — webcam section only (`/detect`, `/video/detect`, `/video/download` byte-for-byte unchanged)

**Unchanged**
- `ai-service/detector.py` (`detector.predict_frame(frame)` interface untouched)
- `ai-service/video_detector.py`
- everything in `backend/` (Node/Express, MongoDB history, auth)
- `frontend/src/api/*`, `frontend/src/lib/*`

## 3. API changes

Preserved, same shape as before: `/webcam/cameras`, `/webcam/start`,
`/webcam/stop`, `/webcam/status`, `/webcam/settings` (confidence
threshold), the `detections` JSON shape, `/detect`, `/video/detect`,
`/video/download/{filename}`, and the Save Result / MongoDB history
flow (all client-side, untouched).

**Removed:** `GET /webcam/stream` (MJPEG). Replaced by:

| Endpoint | Purpose |
|---|---|
| `POST /webcam/webrtc/offer` | WebRTC signaling. Body `{sdp, type}` (browser's SDP offer); response `{sdp, type}` (SDP answer). Requires a camera already started via `/webcam/start`. Raw video only — no detection boxes. |
| `WS /webcam/ws/detections` | Pushes `{type:"detections", frame_id, timestamp, resolution, detections}` the instant a new inference result is ready — latest-only, no backlog for a slow client. `/webcam/status.detections` still works too, for anyone polling instead. |

`/webcam/stop` now also closes every live WebRTC viewer connection, so
a stopped camera visibly disconnects viewers instead of freezing them
on the last frame.

`/webcam/status` gains nothing new in shape; `stream_fps` is still
present for compatibility but now reflects inference rate (video is no
longer JPEG-per-inferred-frame — it's paced independently by WebRTC).

## 4. Why WebRTC, and how it stays low-latency

The previous version of this pipeline shipped MJPEG-over-HTTP with the
same latest-frame buffer, reasoning that removing all queueing was the
actual latency fix and WebRTC's added complexity (`aiortc`, SDP
signaling, ICE, a `<video>` + `RTCPeerConnection` frontend) wasn't
justified for a single/few-viewer LAN deployment. This change
implements that WebRTC path:

- **Video path**: `detector.predict_frame()` is completely off the
  video critical path now — `LatestFrameVideoTrack` reads raw frames
  directly, so a slow or stalled model can no longer add latency to
  what you see. H.264 over RTP also carries far fewer bytes per frame
  than JPEG-over-chunked-HTTP, and gets real hardware-accelerated
  decode in the browser via `<video>`, instead of `<img>` re-decoding
  a full JPEG bitmap every frame.
- **Detections path**: pushed over its own WebSocket the instant
  they're computed, instead of waiting for the next `/webcam/status`
  poll tick (previously every 500ms) — and instead of being baked into
  the video frame, which coupled the two transports' timing together.
- **No new queues anywhere**: both the video track's frame reads and
  the detections WebSocket's fan-out use non-blocking,
  latest-value-only handoffs (see `frame_buffer.py`'s multi-consumer
  support and `manager.py`'s `_queue_put_latest`). A slow WebRTC
  encoder or a slow browser tab can only ever fall behind and repeat
  frames/miss an update — never build a backlog that grows the
  pipeline's actual latency over time.
- **Signaling is plain HTTP offer/answer**, not trickle ICE over a
  second channel — simpler, and sufficient because this deployment is
  same-LAN (see `WEBRTC_ICE_URLS` below if you need STUN/TURN for a
  non-LAN deployment later).

## 5. Jetson installation

```bash
# GStreamer core + plugins (skip any you can already gst-inspect-1.0)
sudo apt update
sudo apt install -y \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly \
  gstreamer1.0-libav \
  python3-gi python3-gi-cairo gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
  v4l-utils \
  libavdevice-dev libavfilter-dev libavformat-dev libavcodec-dev \
  libswresample-dev libswscale-dev libavutil-dev pkg-config libssl-dev \
  libopus-dev libvpx-dev

# Your gst-inspect-1.0 output already confirms nvarguscamerasrc,
# nvv4l2decoder, nvvidconv, v4l2src, appsink are present -- no NVIDIA
# multimedia API packages needed beyond what JetPack ships.
#
# The extra libav*/libssl/libopus/libvpx -dev packages above are for
# `av` (PyAV), which aiortc depends on for H.264 encode -- it builds
# FFmpeg bindings from source on most Jetson/ARM setups, so these
# headers need to be present *before* pip install.

pip install -r requirements-camera.txt --break-system-packages

cp cameras.yaml.example cameras.yaml
# edit cameras.yaml: RTSP URL(s), CSI sensor ids, etc.
```

Optional: if browsers connect to the Jetson from off-LAN (not the
default assumption here), set STUN/TURN before starting the backend:

```bash
export WEBRTC_ICE_URLS="stun:stun.l.google.com:19302"
# For TURN (needed behind symmetric NAT):
# export WEBRTC_ICE_URLS="turn:your-turn-host:3478"
# export WEBRTC_ICE_USERNAME="..."
# export WEBRTC_ICE_CREDENTIAL="..."
```

## 6. Verification commands (run before starting the backend)

```bash
# Elements exist and load:
gst-inspect-1.0 v4l2src
gst-inspect-1.0 nvarguscamerasrc
gst-inspect-1.0 nvv4l2decoder
gst-inspect-1.0 nvvidconv
gst-inspect-1.0 appsink

# USB devices present:
v4l2-ctl --list-devices

# aiortc/av import cleanly (do this once after installing requirements):
python3 -c "import aiortc, av; print('aiortc', aiortc.__version__)"

# Manual GStreamer pipeline smoke tests (unchanged from before --
# capture is still GStreamer, only the delivery mechanism changed):

# USB
gst-launch-1.0 v4l2src device=/dev/video0 io-mode=2 ! \
  image/jpeg,width=1280,height=720,framerate=30/1 ! jpegdec ! \
  videoconvert ! autovideosink

# CSI
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! \
  'video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1,format=NV12' ! \
  nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! autovideosink

# RTSP
gst-launch-1.0 rtspsrc location="rtsp://<user>:<pass>@<ip>:554/stream1" latency=100 protocols=tcp ! \
  rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! \
  video/x-raw,format=BGRx ! videoconvert ! autovideosink
```

## 7. Starting the app

```bash
# Backend (from ai-service/)
uvicorn app:app --host 0.0.0.0 --port 8000

# Frontend (from frontend/, unchanged)
npm run dev
```

Then from your Windows browser: `http://<jetson-ip>:5173` (or whatever
port your dev server uses) with `VITE_API_BASE_URL=http://<jetson-ip>:8000`.

## 8. Test procedure

**A. USB** — select the `usb-N` entry, Start Camera, confirm `<video>`
shows a live picture within ~1-2s (WebRTC handshake + first frame) and
FPS/inference time populate in Technical Details.

**B. CSI** — requires a `csi_cameras` entry in `cameras.yaml`; select
`csi-0`, Start Camera. If it fails, check `journalctl -f` for Argus
daemon errors (only one process can hold the CSI camera at a time).

**C. RTSP** — add camera to `cameras.yaml`, restart backend, select
it, Start Camera. A wrong URL/credentials shows up as a GStreamer
error surfaced through `/webcam/status.error`, same as before.

**D. WebRTC handshake failure** — block UDP between browser and
Jetson (or test cross-subnet without STUN configured) and confirm the
UI surfaces "Jetson camera video connection lost" via
`pc.onconnectionstatechange`/`RTCPeerConnection` state, rather than
hanging silently.

**E. Detections independent of video** — with the stream running,
open the browser's Network tab and confirm `/webcam/ws/detections` is
a separate WS connection receiving JSON messages, distinct from the
WebRTC media flow (which won't show as an HTTP/WS request at all once
established).

**F. Disconnect/reconnect** — unplug USB camera while streaming;
`/webcam/status.error` should populate without crashing the process;
Stop then Start again after replugging should recover, and the
`<video>` element should reconnect cleanly (old `RTCPeerConnection`
closed, new one created).

**G. Repeated start/stop** — click Start/Stop rapidly ~10x; confirm no
orphaned GStreamer processes/threads, `/webcam/status.running` is
`false` after the final Stop, and no lingering `RTCPeerConnection`s
(check `webrtc_peers` count server-side, or that the browser's
`chrome://webrtc-internals` shows no open connections).

**H. Confidence threshold** — move slider, confirm detection boxes on
the canvas overlay (and `/webcam/status` detection count) change
accordingly without restarting the camera or the video connection.

**I. Detection JSON** — confirm `class`, `class_id`, `confidence`,
`bbox` fields are unchanged in shape, both via `/webcam/status` and
over `/webcam/ws/detections`.

**J. Save Result** — click Save Result mid-stream, confirm exactly one
history record is created via the existing Node/Mongo flow — no video
saved, matches previous behavior.

**K. Overlay alignment** — resize the browser window and toggle
Fullscreen while streaming; confirm detection boxes stay correctly
aligned to their objects (this exercises the canvas overlay's
letterboxing math in `drawDetectionOverlay`, since `<video>` uses
`object-fit: contain` and native camera resolution rarely matches the
container's aspect ratio exactly).

**L. Low latency** — point the camera at a stopwatch/phone timer on
screen; photograph both the source screen and the browser `<video>`
simultaneously; compare timestamps. Expect noticeably lower end-to-end
delay than the old MJPEG path on LAN for USB/CSI; RTSP will be
somewhat higher depending on `rtsp_latency_ms` (unchanged, still the
GStreamer capture stage).

**M. CPU/GPU/FPS** — `tegrastats` in one terminal while streaming;
`frames_dropped` in `/webcam/status` should still only grow when
inference genuinely can't keep up with camera FPS — that stat is tied
to the inference worker's consumption specifically, not to WebRTC
viewers (see `frame_buffer.py`).

## 9. Common errors and fixes

| Symptom | Cause | Fix |
|---|---|---|
| `GStreamer Python bindings ... not available` | `python3-gi` not installed / wrong Python env | `sudo apt install python3-gi gir1.2-gstreamer-1.0`; ensure the venv can see system site-packages, or don't use a venv for this service |
| `Pipeline must contain an appsink element named 'sink'` | custom pipeline missing `! appsink name=sink` | fix `cameras.yaml` custom pipeline |
| CSI start fails immediately | another process holds the camera | `sudo systemctl restart nvargus-daemon`, ensure Stop was called before killing the backend |
| RTSP connects then immediately errors | wrong `codec` (h264 vs h265) or auth | check camera's actual codec via `ffprobe rtsp://...`; verify credentials |
| `Start a camera with /webcam/start before requesting a WebRTC stream.` (400) | frontend called `/webcam/webrtc/offer` before/without a successful `/webcam/start` | check the `/webcam/start` response's `success` field first; the frontend already does this |
| `<video>` never shows a picture, no errors in console | UDP blocked between browser and Jetson (firewall/VPN), and no STUN/TURN configured for that path | for same-LAN this should just work with host candidates; for anything routed, set `WEBRTC_ICE_URLS` (see §5) |
| `pip install` fails building `av` | missing FFmpeg dev headers | install the `libav*-dev`/`libssl-dev`/`libopus-dev`/`libvpx-dev` packages listed in §5 before retrying |
| Detections stop updating but video keeps playing | `/webcam/ws/detections` connection dropped (e.g. dev-server proxy restart) | this is exactly the point of separating the transports — reconnect just that WebSocket without touching the WebRTC video connection; the frontend does this per Start/Stop cycle |
| `/webcam/cameras` returns empty | `v4l2-ctl` not installed | `sudo apt install v4l-utils` |

## 10. Rollback procedure

1. Stop the backend.
2. `git checkout -- ai-service/app.py ai-service/camera/manager.py ai-service/camera/frame_buffer.py frontend/src/routes/webcam.tsx`
   (or restore from your own backup).
3. `rm -f ai-service/camera/webrtc.py`
4. `pip uninstall aiortc av aioice -y` (optional; harmless to leave installed)
5. Restart the backend — `detector.py`, `video_detector.py`, and the
   Node history backend were never touched, so nothing else needs
   reverting.
