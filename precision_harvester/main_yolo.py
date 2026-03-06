from ultralytics import YOLO
import cv2
import numpy as np
import time
import requests

# ── ARDUINO ──
arduino = None
# To enable: import serial; arduino = serial.Serial('COM8', 9600); time.sleep(2)

# ── MODEL ──
model = YOLO("../runs/detect/train6/weights/best.pt")
cap   = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ── SERVER ──
SERVER_URL = "http://localhost:5050/update"
FRAME_URL  = "http://localhost:5050/frame"

# ── CALIBRATION ──
PIXEL_TO_CM        = 50.47   # calibrated from credit card
CREDIT_CARD_CM     = 8.56
calibration_mode   = False
calibration_points = []

# ── THRESHOLDS ──
MIN_LENGTH_CM      = 5
JUVENILE_THRESHOLD = 20
FRAME_H            = 480

# ════════════════════════════════════════════════
# SPECIES DATABASE (Von Bertalanffy parameters)
# ════════════════════════════════════════════════
SPECIES_DB = [
    {"name":"Juvenile", "sci":"Unknown sp.",    "min_cm":0,  "max_cm":10,  "color":(0,0,255),     "L_inf":40.0,"K":0.30,"t0":-0.5,"a":0.010,"b":3.00},
    {"name":"Rohu",     "sci":"Labeo rohita",   "min_cm":10, "max_cm":22,  "color":(0,200,255),   "L_inf":45.0,"K":0.28,"t0":-0.4,"a":0.012,"b":3.05},
    {"name":"Catla",    "sci":"Catla catla",    "min_cm":22, "max_cm":38,  "color":(0,255,100),   "L_inf":60.0,"K":0.22,"t0":-0.6,"a":0.009,"b":3.10},
    {"name":"Tilapia",  "sci":"O. niloticus",   "min_cm":38, "max_cm":999, "color":(255,165,0),   "L_inf":55.0,"K":0.35,"t0":-0.3,"a":0.015,"b":2.95},
]

def classify_species(length_cm):
    for sp in SPECIES_DB:
        if sp["min_cm"] <= length_cm < sp["max_cm"]:
            return sp
    return SPECIES_DB[-1]

def estimate_age(length_cm, sp):
    ratio = length_cm / sp["L_inf"]
    if ratio >= 0.98: return 99.0
    age = sp["t0"] - (1.0 / sp["K"]) * np.log(1.0 - ratio)
    return max(0.1, round(age, 1))

def estimate_weight(length_cm, sp):
    return sp["a"] * (length_cm ** sp["b"])

# ════════════════════════════════════════════════
# FISH FILTER
# ════════════════════════════════════════════════
def is_fish(x1, y1, x2, y2, conf):
    w = x2 - x1; h = y2 - y1
    aspect = w / (h + 1e-5)
    if conf   < 0.15:          return False
    if w      > 600:           return False
    if w*h    > 180000:        return False
    if aspect < 0.4:           return False
    if aspect > 8.0:           return False
    if h      > FRAME_H * 0.4: return False
    return True

# ════════════════════════════════════════════════
# PERSPECTIVE CORRECTION
# ════════════════════════════════════════════════
def perspective_correction(x1, y1, x2, y2, fw, fh):
    cx = (x1+x2)/2; cy = (y1+y2)/2
    dx = abs(cx - fw/2) / (fw/2)
    dy = abs(cy - fh/2) / (fh/2)
    dist = np.sqrt(dx**2 + dy**2) / np.sqrt(2)
    return 1.0 + 0.12 * dist

# ════════════════════════════════════════════════
# MOUSE CALLBACK — only registered once
# ════════════════════════════════════════════════
def mouse_callback(event, x, y, flags, param):
    global calibration_points, calibration_mode, PIXEL_TO_CM
    if calibration_mode and event == cv2.EVENT_LBUTTONDOWN:
        calibration_points.append((x, y))
        print(f"[Cal] Point {len(calibration_points)}: ({x},{y})")
        if len(calibration_points) == 2:
            px = np.sqrt((calibration_points[1][0]-calibration_points[0][0])**2 +
                         (calibration_points[1][1]-calibration_points[0][1])**2)
            PIXEL_TO_CM = px / CREDIT_CARD_CM
            print(f"[Cal] Done! {px:.1f}px = {CREDIT_CARD_CM}cm → PIXEL_TO_CM={PIXEL_TO_CM:.2f}")
            calibration_points = []
            calibration_mode   = False

# Register mouse callback ONCE before loop
cv2.namedWindow("Precision Harvester")
cv2.setMouseCallback("Precision Harvester", mouse_callback)

# ── SESSION STATE ──
session_biomass = 0.0
last_send_time  = time.time()
SEND_INTERVAL   = 0.5

# ════════════════════════════════════════════════
# MAIN LOOP
# ════════════════════════════════════════════════
while True:
    ret, frame = cap.read()
    if not ret: break

    fh, fw = frame.shape[:2]

    # ── CALIBRATION OVERLAY ──
    if calibration_mode:
        dark = frame.copy()
        cv2.rectangle(dark, (0,0), (fw,fh), (0,0,0), -1)
        cv2.addWeighted(dark, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, "CALIBRATION MODE — Click LEFT then RIGHT edge of card",
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,229,195), 2)
        cv2.putText(frame, f"Card width = {CREDIT_CARD_CM}cm  |  Current PIXEL_TO_CM = {PIXEL_TO_CM:.1f}",
                    (20, fh-20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,200,0), 1)
        if calibration_points:
            cv2.circle(frame, calibration_points[0], 7, (0,229,195), -1)
            cv2.putText(frame, "Point 1 set — now click RIGHT edge",
                        (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,229,195), 1)
        cv2.imshow("Precision Harvester", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break
        if key == ord('c'): calibration_mode = False
        continue

    # ── YOLO ──
    results = model(frame, conf=0.15, iou=0.25, verbose=False)

    frame_total = 0; frame_juvenile = 0
    frame_biomass = 0.0; frame_species = {}

    for box in results[0].boxes:
        x1,y1,x2,y2 = [float(v) for v in box.xyxy[0]]
        conf = float(box.conf[0])
        if not is_fish(x1,y1,x2,y2,conf): continue

        # Perspective + length
        corr       = perspective_correction(x1,y1,x2,y2,fw,fh)
        length_cm  = (x2-x1) * corr / PIXEL_TO_CM
        sp         = classify_species(length_cm)
        age        = estimate_age(length_cm, sp)
        weight     = estimate_weight(length_cm, sp)

        frame_total   += 1
        frame_biomass += weight
        is_juv         = length_cm < MIN_LENGTH_CM
        if is_juv: frame_juvenile += 1
        frame_species[sp["name"]] = frame_species.get(sp["name"],0) + 1

        # Box + label
        color  = (0,0,255) if is_juv else sp["color"]
        age_str = f"{age:.1f}yr" if age < 99 else "Mature"
        label1 = f"{sp['name']} ({sp['sci']})"
        label2 = f"{length_cm:.1f}cm | {weight:.0f}g | Age:{age_str}"

        (tw1,_),_ = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        (tw2,_),_ = cv2.getTextSize(label2, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        lx = int(x1); ly = max(int(y1)-32, 0)
        cv2.rectangle(frame,(lx,ly),(lx+max(tw1,tw2)+8,ly+32),(0,0,0),-1)
        cv2.rectangle(frame,(lx,ly),(lx+max(tw1,tw2)+8,ly+32),color,1)
        cv2.putText(frame, label1, (lx+3,ly+13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
        cv2.putText(frame, label2, (lx+3,ly+27), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,200), 1)
        cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),color,2)
        cv2.circle(frame,(int(x2)-8,int(y1)+8),4,(0,255,0) if conf>0.5 else (255,165,0),-1)

    session_biomass += frame_biomass
    juvenile_pct = (frame_juvenile/frame_total*100) if frame_total>0 else 0
    is_alert     = juvenile_pct > JUVENILE_THRESHOLD

    if arduino:
        arduino.write(("ALERT\n" if is_alert else "SAFE\n").encode())

    # ── HUD ──
    cv2.rectangle(frame,(0,0),(fw,128),(0,0,0),-1)
    cv2.rectangle(frame,(0,0),(fw,128),(0,229,195),1)
    cv2.putText(frame, f"Fish: {frame_total}",             (10,22),  cv2.FONT_HERSHEY_SIMPLEX, 0.6,  (255,255,0),   1)
    cv2.putText(frame, f"Juv%: {juvenile_pct:.1f}%",       (10,46),  cv2.FONT_HERSHEY_SIMPLEX, 0.6,  (0,255,255),   1)
    cv2.putText(frame, f"Biomass: {session_biomass:.0f}g", (10,70),  cv2.FONT_HERSHEY_SIMPLEX, 0.6,  (255,200,0),   1)
    cv2.putText(frame, f"Scale: 1px={1/PIXEL_TO_CM:.3f}cm",(10,92),  cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1)
    cv2.putText(frame, "C:Calibrate | ESC:Quit",           (10,112), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80,80,80),    1)

    # Species list top-right
    cv2.putText(frame,"SPECIES:",(fw-190,20),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,229,195),1)
    for i,(sn,cnt) in enumerate(frame_species.items()):
        sc = next((s["color"] for s in SPECIES_DB if s["name"]==sn),(255,255,255))
        cv2.putText(frame,f"{sn}: {cnt}",(fw-190,40+i*20),cv2.FONT_HERSHEY_SIMPLEX,0.42,sc,1)

    if is_alert:
        cv2.rectangle(frame,(0,fh-36),(fw,fh),(0,0,180),-1)
        cv2.putText(frame,"WARNING: HIGH JUVENILE CATCH — RELEASE RECOMMENDED",
                    (10,fh-12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

    # ── SEND ──
    now = time.time()
    if now - last_send_time >= SEND_INTERVAL:
        try:
            requests.post(SERVER_URL, json={
                "total_count":         frame_total,
                "juvenile_count":      frame_juvenile,
                "adult_count":         frame_total - frame_juvenile,
                "juvenile_percentage": round(juvenile_pct,1),
                "total_biomass_g":     round(session_biomass,1),
                "alert":               is_alert,
                "species_counts":      frame_species,
                "pixel_to_cm":         round(PIXEL_TO_CM,2),
            }, timeout=0.3)
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY,75])
            requests.post(FRAME_URL, data=buf.tobytes(),
                          headers={"Content-Type":"application/octet-stream"}, timeout=0.3)
        except: pass
        last_send_time = now

    cv2.imshow("Precision Harvester", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27: break
    if key == ord('c'):
        calibration_mode   = True
        calibration_points = []
        print("[Cal] Mode ON — click left then right edge of credit card")

cap.release()
cv2.destroyAllWindows()