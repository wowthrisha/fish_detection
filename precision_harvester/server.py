from flask import Flask, jsonify, send_file, request, Response
from flask_cors import CORS
import time, os, cv2, threading
from collections import deque

app = Flask(__name__)
CORS(app)

# ── Fish detection state ──
state = {
    "total_count": 0, "juvenile_count": 0, "adult_count": 0,
    "juvenile_percentage": 0.0, "total_biomass_g": 0.0,
    "alert": False, "session_start": time.time(),
}
history = deque(maxlen=60)

# ── ANFIS / Sensor state ──
sensor_state = {
    "temp": 27.0, "ph": 7.2,
    "predicted_weight": 0.0, "condition_score": 0.0,
    "growth_status": "Waiting...",
    "pred_history": [], "sensor_history": [],
    "arduino_connected": False,
}

# ── Camera stream ──
latest_frame = None
frame_lock   = threading.Lock()


@app.route("/")
def dashboard():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
    return send_file(path)


@app.route("/update", methods=["POST"])
def update():
    data = request.json
    state.update(data)
    history.append({
        "timestamp":    time.time(),
        "juvenile_pct": data.get("juvenile_percentage", 0),
        "biomass":      data.get("total_biomass_g", 0),
        "total":        data.get("total_count", 0),
    })
    return jsonify({"ok": True})


@app.route("/sensor", methods=["POST"])
def receive_sensor():
    sensor_state.update(request.json)
    return jsonify({"ok": True})


@app.route("/sensor_data")
def get_sensor_data():
    return jsonify(sensor_state)


@app.route("/frame", methods=["POST"])
def receive_frame():
    global latest_frame
    import numpy as np
    nparr = np.frombuffer(request.data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    with frame_lock:
        latest_frame = frame
    return jsonify({"ok": True})


def generate_stream():
    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.05)


@app.route("/stream")
def stream():
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/data")
def get_data():
    return jsonify({
        **state,
        "history": list(history),
        "elapsed": int(time.time() - state["session_start"])
    })


@app.route("/reset", methods=["POST"])
def reset():
    state.update({
        "total_count": 0, "juvenile_count": 0, "adult_count": 0,
        "juvenile_percentage": 0.0, "total_biomass_g": 0.0,
        "alert": False, "session_start": time.time()
    })
    history.clear()
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(port=5050, debug=False, threaded=True)