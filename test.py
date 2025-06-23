import cv2
from ultralytics import YOLO
import cvzone

# Load YOLOv8 model (person detection)
model = YOLO('yolo12s.pt')
names = model.names

# Define slanted line coordinates
line_p1 = (344, 512)
line_p2 = (643, 222)

# To store previous center positions of each tracked ID
hist = {}

# IN/OUT counters
in_count = 0
out_count = 0

# To avoid double-counting
counted_ids = set()

# Function to check side of point relative to a line
def point_side_of_line(x, y, x1, y1, x2, y2):
    return (x - x1)*(y2 - y1) - (y - y1)*(x2 - x1)

# Open video file or webcam
cap = cv2.VideoCapture("vid.mp4")  # Replace with 0 for webcam

# Mouse debugging
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse moved to: [{x}, {y}]")

cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 2 != 0:
        continue

    frame = cv2.resize(frame, (1020,600))

    # Detect people (class 0)
    results = model.track(frame, persist=True, classes=[0])

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, class_id in zip(boxes, ids, class_ids):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

            if track_id in hist:
                prev_cx, prev_cy = hist[track_id]
                prev_side = point_side_of_line(prev_cx, prev_cy, *line_p1, *line_p2)
                curr_side = point_side_of_line(cx, cy, *line_p1, *line_p2)
                # Check if object crossed the line
                if prev_side * curr_side < 0 and track_id not in counted_ids:
                   if curr_side < 0:
                      direction = "OUT"
                      out_count += 1
                   else:
                      direction = "IN"
                      in_count += 1
                   counted_ids.add(track_id)  
                   color = (0, 0, 255) if direction == "OUT" else (0, 255, 0)
                   cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                   cvzone.putTextRect(frame, f"ID: {track_id} {direction}", (x1, y1 + 10), scale=1, thickness=2,
                                       colorT=(255, 255, 255), colorR=color, offset=5, border=2)

            hist[track_id] = (cx, cy)

    # Draw counting line
    cv2.line(frame, line_p1, line_p2, (0, 0, 255), 2)

    # Show IN/OUT counts
    cvzone.putTextRect(frame, f'IN: {in_count}', (316, 33), scale=2, thickness=2,
                       colorT=(255, 255, 255), colorR=(0, 128, 0))
    cvzone.putTextRect(frame, f'OUT: {out_count}', (316, 100), scale=2, thickness=2,
                       colorT=(255, 255, 255), colorR=(0, 0, 255))

    # Display frame
    cv2.imshow("RGB", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

