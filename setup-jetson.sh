#!/usr/bin/env bash
# System prerequisites for the ASTROX AI service on Jetson (JetPack 6).
# torch / torchvision / tensorrt / opencv come WITH JetPack — do not touch them.
set -e

sudo apt update
sudo apt install -y \
  python3-gi python3-gi-cairo \
  gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad  gstreamer1.0-plugins-ugly \
  gstreamer1.0-libav \
  v4l-utils \
  libavdevice-dev libavfilter-dev libavformat-dev libavcodec-dev \
  libswresample-dev libswscale-dev libavutil-dev pkg-config \
  libssl-dev libopus-dev libvpx-dev        # <- needed before: pip install aiortc

echo
echo "Now, inside your venv:"
echo "  pip install -r requirements.txt"
echo "  pip install --no-deps ultralytics ultralytics-thop   # torch/cv2 come from JetPack"
