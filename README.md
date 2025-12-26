# Counting_system_using_YOLOv8n
This project demonstrates real-time object detection, tracking, and unique object counting using YOLOv8 and ByteTrack. It detects objects from a live webcam feed (or video file), assigns persistent tracking IDs, and maintains per-class unique object counts across frames.
Key Features

YOLOv8 object detection using Ultralytics

ByteTrack multi-object tracking with persistent IDs

Unique object counting per class (no double counting)

Live counter overlay rendered directly on the video stream

Works with webcam or video files

Final summary of unique object counts printed on exit

How It Works

YOLOv8 detects objects frame-by-frame.

ByteTrack assigns a stable tracking ID to each object.

A defaultdict(set) stores unique tracking IDs per class.

Each object is counted once, even if it appears across many frames.

Counts update live and are displayed on the video.

Code Overview

model.track(...) enables detection + tracking in one call.

persist=True keeps tracking IDs consistent across frames.

Each detected box provides:

box.cls → class ID

box.id → unique tracking ID

IDs are stored per class to avoid duplicates.

Example Use Cases

People counting

Vehicle counting

Crowd analytics

Surveillance and monitoring

Computer vision learning projects

Requirements

Python 3.8+

ultralytics

opencv-python

lap (required for ByteTrack)
