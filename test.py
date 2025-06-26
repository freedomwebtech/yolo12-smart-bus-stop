import cv2
from ultralytics import YOLO
import cvzone

# Load YOLOv8 model (person detection)
model = YOLO('yolo12s.pt')
names = model.names



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
            cvzone.putTextRect(frame,f'{track_id}',(x1,y1),1,1)
            

    # Draw counting line
#    cv2.line(frame, line_p1, line_p2, (0, 0, 255), 2)

    # Show IN/OUT counts
 #   cvzone.putTextRect(frame, f'IN: {in_count}', (316, 33), scale=2, thickness=2,
 #                      colorT=(255, 255, 255), colorR=(0, 128, 0))
 #   cvzone.putTextRect(frame, f'OUT: {out_count}', (316, 100), scale=2, thickness=2,
 #                      colorT=(255, 255, 255), colorR=(0, 0, 255))

    # Display frame
    cv2.imshow("RGB", frame)

    # Exit on ESC
    if cv2.waitKey(0) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

