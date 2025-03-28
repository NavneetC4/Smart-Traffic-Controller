import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load YOLOv8 model
model = YOLO("yolov8m.pt")

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    print("❌ Error: Could not open camera.")
    exit()

# Tracking and timing
prev_centers = {}
light_states = {'N': 'green', 'S': 'green', 'E': 'green', 'W': 'green'}
hold_red_until = {'N': 0, 'S': 0, 'E': 0, 'W': 0}

# Parameters
RED_HOLD_TIME = 3  # seconds
MOTION_THRESHOLD = 10  # pixels
AXIS_PROXIMITY_THRESHOLD = 50  # pixels distance from axis line
zone_counts = {'NW': 0, 'NE': 0, 'SW': 0, 'SE': 0}

def get_zone(cx, cy, frame_w, frame_h):
    if cx < frame_w // 2 and cy < frame_h // 2:
        return "NW"
    elif cx >= frame_w // 2 and cy < frame_h // 2:
        return "NE"
    elif cx < frame_w // 2 and cy >= frame_h // 2:
        return "SW"
    else:
        return "SE"

def update_lights(movement_vector, zone, cx, cy, frame_w, frame_h):
    now = time.time()

    dist_x = abs(cx - frame_w // 2)
    dist_y = abs(cy - frame_h // 2)

    if dist_x <= AXIS_PROXIMITY_THRESHOLD:
        hold_red_until['N'] = now + RED_HOLD_TIME
        hold_red_until['S'] = now + RED_HOLD_TIME

    if dist_y <= AXIS_PROXIMITY_THRESHOLD:
        hold_red_until['E'] = now + RED_HOLD_TIME
        hold_red_until['W'] = now + RED_HOLD_TIME

def draw_traffic_lights(frame):
    colors = {'red': (0, 0, 255), 'green': (0, 255, 0)}
    h, w, _ = frame.shape
    radius = 30
    now = time.time()

    for dir in light_states:
        light_states[dir] = 'red' if hold_red_until[dir] > now else 'green'

    cv2.circle(frame, (w//2, 50), radius, colors[light_states['N']], -1)
    cv2.circle(frame, (w//2, h-50), radius, colors[light_states['S']], -1)
    cv2.circle(frame, (w-50, h//2), radius, colors[light_states['E']], -1)
    cv2.circle(frame, (50, h//2), radius, colors[light_states['W']], -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'N', (w//2 - 10, 45), font, 0.6, (255,255,255), 2)
    cv2.putText(frame, 'S', (w//2 - 10, h - 55), font, 0.6, (255,255,255), 2)
    cv2.putText(frame, 'E', (w - 60, h//2 - 10), font, 0.6, (255,255,255), 2)
    cv2.putText(frame, 'W', (60, h//2 - 10), font, 0.6, (255,255,255), 2)

    # Display zone counts
    cv2.putText(frame, f"NW: {zone_counts['NW']}", (10, 30), font, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"NE: {zone_counts['NE']}", (w - 150, 30), font, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"SW: {zone_counts['SW']}", (10, h - 20), font, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"SE: {zone_counts['SE']}", (w - 150, h - 20), font, 0.6, (255, 255, 255), 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = frame.shape[:2]

    results = model(frame)
    zone_counts = {'NW': 0, 'NE': 0, 'SW': 0, 'SE': 0}
    new_centers = {}

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if conf > 0.5 and cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                zone = get_zone(cx, cy, w, h)
                zone_counts[zone] += 1
                center_key = (cx // 20, cy // 20)

                if center_key in prev_centers:
                    prev_cx, prev_cy = prev_centers[center_key]
                    dx, dy = cx - prev_cx, cy - prev_cy
                    update_lights((dx, dy), zone, cx, cy, w, h)

                new_centers[center_key] = (cx, cy)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    prev_centers = new_centers.copy()

    # Draw semi-transparent road overlays
    overlay = frame.copy()
    cv2.rectangle(overlay, (w//2 - 60, 0), (w//2 + 60, h), (50, 50, 50), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h//2 - 60), (w, h//2 + 60), (50, 50, 50), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    draw_traffic_lights(frame)

    cv2.imshow("4-Way Intersection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
