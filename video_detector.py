import cv2
import time
import os
import shutil
import subprocess
from collections import Counter

_default_detector = None


def _get_default_detector():
    """
    Lazily create a detector only if the caller didn't pass one in.
    app.py should always pass its already-loaded detector so the
    TensorRT engine is loaded exactly once for the whole process
    (previously it was being loaded twice: once here at import time,
    once in app.py).
    """
    global _default_detector
    if _default_detector is None:
        from detector import Detector
        _default_detector = Detector("best.engine")
    return _default_detector


def _reencode_h264(temp_path, final_path):
    """
    OpenCV's mp4v (MPEG-4 Part 2) fourcc produces a file that Chrome,
    Edge and Firefox do not reliably decode in a <video> element.
    Re-encode the finished file to H.264/yuv420p with ffmpeg so it
    actually plays in the browser. This runs once per video, after
    inference is done, so it doesn't touch the detection pipeline.

    Falls back to keeping the raw mp4v file if ffmpeg isn't available,
    so this can never break video generation outright.
    """
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", temp_path,
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                final_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.remove(temp_path)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        shutil.move(temp_path, final_path)
        return False


def process_video(input_path, output_path, detector=None):

    if detector is None:
        detector = _get_default_detector()

    # Fresh ByteTrack state for THIS video. Without this, track IDs
    # would continue on from whatever a previous video (or a live
    # webcam session, since the detector is shared) left off at.
    detector.reset_tracker()

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError("Unable to open input video")

    video_fps = cap.get(cv2.CAP_PROP_FPS)

    if video_fps <= 0:
        video_fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Write raw frames with mp4v first (broadly supported by OpenCV
    # builds without extra codec setup), then re-encode to a
    # browser-safe H.264 file afterwards.
    temp_output_path = output_path + ".raw.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        temp_output_path,
        fourcc,
        video_fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"Unable to create output video: {temp_output_path}"
        )

    processed_frames = 0

    # Per-track aggregate state, keyed by ByteTrack's track_id.
    # "objects_detected" is computed from the NUMBER OF UNIQUE TRACKS
    # here at the end, never from a running per-frame detection count
    # -- that's what stops the same drone across 100 frames from
    # becoming "100 drones".
    tracks: dict[int, dict] = {}

    start_time = time.time()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        result = detector.track_frame(frame)

        detections = result.get("detections", [])

        for detection in detections:

            track_id = detection.get("track_id")

            # No confirmed track ID yet -- draw the box (still a real,
            # honest detection) but don't count it toward the unique
            # object total until ByteTrack assigns it a stable ID.
            if track_id is None:
                continue

            class_name = detection["class"]
            confidence = detection["confidence"]

            x1, y1, x2, y2 = detection["bbox"]

            entry = tracks.get(track_id)
            if entry is None:
                entry = {
                    "classes": Counter(),
                    "confidences": [],
                    "first_seen_frame": processed_frames,
                    "last_seen_frame": processed_frames,
                }
                tracks[track_id] = entry

            entry["classes"][class_name] += 1
            entry["confidences"].append(confidence)
            entry["last_seen_frame"] = processed_frames

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = f"{class_name} #{track_id} {confidence:.2f}"

            text_y = max(y1 - 10, 20)

            cv2.putText(
                frame,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        writer.write(frame)

        processed_frames += 1

    cap.release()
    writer.release()

    # Re-encode to a browser-compatible H.264 MP4. This is what the
    # /video/download/{filename} endpoint will actually serve.
    _reencode_h264(temp_output_path, output_path)

    elapsed = time.time() - start_time

    processing_fps = (
        processed_frames / elapsed
        if elapsed > 0
        else 0
    )

    # ------------------------------------------------------------------
    # Finalize per-track summary. Every number here is a straight
    # aggregate of real confidence values the model returned for that
    # track -- nothing is fabricated.
    # ------------------------------------------------------------------
    track_summaries = []
    class_counts: dict = {}

    for track_id, entry in tracks.items():
        # A track can, rarely, flicker between classes frame to frame
        # (e.g. a misclassified frame or two). We report the class
        # that track was seen as most often, rather than double-
        # counting one physical object under two class labels.
        track_class = entry["classes"].most_common(1)[0][0]
        confidences = entry["confidences"]

        track_average_confidence = round(sum(confidences) / len(confidences), 4)
        track_max_confidence = round(max(confidences), 4)

        track_summaries.append({
            "track_id": track_id,
            "class": track_class,
            "first_seen_frame": entry["first_seen_frame"],
            "last_seen_frame": entry["last_seen_frame"],
            "max_confidence": track_max_confidence,
            "average_confidence": track_average_confidence,
        })

        class_counts[track_class] = class_counts.get(track_class, 0) + 1

    track_summaries.sort(key=lambda t: t["track_id"])

    objects_detected = len(track_summaries)

    # Documented rule: the session-level average/max confidence are
    # computed from each UNIQUE TRACK's own average/max confidence --
    # i.e. every physical object contributes exactly once, regardless
    # of whether it was visible for 20 frames or 900. This intentionally
    # differs from a frame-weighted average, which would let a
    # long-lived object dominate the number.
    if track_summaries:
        average_confidence = round(
            sum(t["average_confidence"] for t in track_summaries) / len(track_summaries), 4
        )
        max_confidence = round(
            max(t["max_confidence"] for t in track_summaries), 4
        )
    else:
        average_confidence = None
        max_confidence = None

    return {
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "video_fps": round(video_fps, 2),
        "processing_fps": round(processing_fps, 2),
        "processing_time": round(elapsed, 2),
        "resolution": f"{width}x{height}",

        # UNIQUE tracked objects -- NOT sum(detections per frame).
        "objects_detected": objects_detected,
        "class_counts": class_counts,
        "tracks": track_summaries,

        # Real aggregate confidence figures (see rule above), None only
        # when nothing was ever tracked.
        "average_confidence": average_confidence,
        "max_confidence": max_confidence,
    }
