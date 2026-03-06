from ultralytics import YOLO
import cv2
import serial
import time

# 🔌 Connect Arduino
arduino = serial.Serial('COM8', 9600)
time.sleep(2)

model = YOLO("runs/detect/train6/weights/best.pt")

cap = cv2.VideoCapture(1)

# 🔧 CALIBRATION VALUE
PIXEL_TO_CM = 10   # Example: 10 pixels = 1 cm (YOU MUST CALIBRATE)
MIN_LENGTH_CM = 15  # Below this = Juvenile
JUVENILE_THRESHOLD = 20  # % limit

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.3)

    total_count = 0
    juvenile_count = 0

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]

        length_pixels = x2 - x1
        fish_length_cm = float(length_pixels) / PIXEL_TO_CM

        total_count += 1

        # 🔥 CLASSIFY
        if fish_length_cm < MIN_LENGTH_CM:
            juvenile_count += 1
            color = (0, 0, 255)  # Red for juvenile
            label = f"Juvenile {fish_length_cm:.1f}cm"
        else:
            color = (0, 255, 0)  # Green for adult
            label = f"Adult {fish_length_cm:.1f}cm"

        # Draw box
        cv2.rectangle(frame,
                      (int(x1), int(y1)),
                      (int(x2), int(y2)),
                      color, 2)

        cv2.putText(frame, label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)

    # 🔥 CALCULATE PERCENTAGE
    if total_count > 0:
        juvenile_percentage = (juvenile_count / total_count) * 100
    else:
        juvenile_percentage = 0

    # Display stats
    cv2.putText(frame, f"Total Fish: {total_count}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255,255,0), 2)

    cv2.putText(frame, f"Juvenile %: {juvenile_percentage:.1f}%",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0,255,255), 2)

    # 🚨 TRIGGER ALERT
    if juvenile_percentage > JUVENILE_THRESHOLD:
        cv2.putText(frame, "⚠ HIGH JUVENILE CATCH!",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,0,255), 3)

        arduino.write("ALERT\n".encode())
    else:
        arduino.write("SAFE\n".encode())

    cv2.imshow("Precision Harvester - Sustainable Catch Monitor", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()