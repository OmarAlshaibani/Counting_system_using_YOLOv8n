from ultralytics import YOLO
import cv2
from collections import defaultdict

# Load YOLOv8 model
model = YOLO("yolov8n.pt")
print("YOLO model loaded successfully")

# Open webcam (change 0 to video path if needed)
cap = cv2.VideoCapture(0)

# Store unique IDs per class
tracked_ids = defaultdict(set)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO with tracking (ByteTrack)
    results = model.track(
        frame,
        persist=True,
        conf=0.5,
        iou=0.5,
        tracker="bytetrack.yaml"
    )

    if results[0].boxes is not None:
        boxes = results[0].boxes

        for box in boxes:
            if box.id is None:
                continue  # skip if no tracking ID

            track_id = int(box.id[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # Count UNIQUE object per class
            tracked_ids[class_name].add(track_id)

    # Draw detections
    annotated_frame = results[0].plot()

    # ---- Overlay live counters ----
    y_offset = 30
    for cls, ids in tracked_ids.items():
        text = f"{cls}: {len(ids)}"
        cv2.putText(
            annotated_frame,
            text,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        y_offset += 30

    cv2.imshow("YOLOv8 Unique Object Counting", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# ---- Final counts after exit ----
print("\nFinal Unique Object Counts:")
for cls, ids in tracked_ids.items():
    print(f"{cls}: {len(ids)}")
