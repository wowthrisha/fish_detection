from ultralytics import YOLO
import cv2
import time
import requests

# 🔌 Arduino
arduino = None

# ── MODEL & CAMERA ──
model = YOLO("../runs/detect/train6/weights/best.pt")
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ── CALIBRATION ──
# If fish showing as juvenile wrongly → lower this number
# If fish showing too large → raise this number
PIXEL_TO_CM        = 20.0   # ← TUNE THIS (try 15, 18, 20, 25)
MIN_LENGTH_CM      = 10     # below this = juvenile
JUVENILE_THRESHOLD = 20     # % alert limit

# ── SERVER ──
SERVER_URL = "http://localhost:5050/update"
FRAME_URL  = "http://localhost:5050/frame"

# ── FISH FILTER ──
FRAME_H = 480

def is_fish(x1, y1, x2, y2, conf):
    w = x2 - x1
    h = y2 - y1
    aspect = w / (h + 1e-5)
    if conf   < 0.10:           return False  # low conf
    if w      > 650:            return False  # too wide = human
    if w*h    > 200000:         return False  # too large
    if aspect < 0.35:           return False  # too tall
    if aspect > 9.0:            return False  # too wide
    if h      > FRAME_H * 0.5:  return False  # takes up too much height
    return True

def estimate_weight_g(length_cm, a=0.01, b=3.0):
    return a * (length_cm ** b)

# ── SESSION STATE ──
session_biomass = 0.0
last_send_time  = time.time()
SEND_INTERVAL   = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Lower conf + iou to detect MORE fish
    results = model(frame, conf=0.10, iou=0.30, verbose=False)

    frame_total    = 0
    frame_juvenile = 0
    frame_biomass  = 0.0

    for box in results[0].boxes:
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        conf            = float(box.conf[0])

        if not is_fish(x1, y1, x2, y2, conf):
            continue

        fish_length_cm = (x2 - x1) / PIXEL_TO_CM
        weight_g       = estimate_weight_g(fish_length_cm)

        frame_total   += 1
        frame_biomass += weight_g

        if fish_length_cm < MIN_LENGTH_CM:
            frame_juvenile += 1
            color = (0, 0, 255)
            label = f"Juvenile {fish_length_cm:.1f}cm | {weight_g:.1f}g"
        else:
            color = (0, 255, 0)
            label = f"Adult {fish_length_cm:.1f}cm | {weight_g:.1f}g"

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    session_biomass += frame_biomass
    juvenile_pct     = (frame_juvenile / frame_total * 100) if frame_total > 0 else 0
    is_alert         = juvenile_pct > JUVENILE_THRESHOLD

    if arduino:
        arduino.write(("ALERT\n" if is_alert else "SAFE\n").encode())

    if is_alert:
        cv2.putText(frame, "WARNING: HIGH JUVENILE CATCH!",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    cv2.putText(frame, f"Total Fish: {frame_total}",         (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255,   0), 2)
    cv2.putText(frame, f"Juvenile %: {juvenile_pct:.1f}%",   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (  0, 255, 255), 2)
    cv2.putText(frame, f"Biomass:    {session_biomass:.0f}g", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200,   0), 2)

    now = time.time()
    if now - last_send_time >= SEND_INTERVAL:
        try:
            requests.post(SERVER_URL, json={
                "total_count":         frame_total,
                "juvenile_count":      frame_juvenile,
                "adult_count":         frame_total - frame_juvenile,
                "juvenile_percentage": round(juvenile_pct, 1),
                "total_biomass_g":     round(session_biomass, 1),
                "alert":               is_alert,
            }, timeout=0.3)
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            requests.post(FRAME_URL, data=buf.tobytes(),
                          headers={"Content-Type": "application/octet-stream"},
                          timeout=0.3)
        except:
            pass
        last_send_time = now

    cv2.imshow("Precision Harvester", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()