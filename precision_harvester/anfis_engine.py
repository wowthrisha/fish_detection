"""
ANFIS Digital Twin — Fish Biomass Predictor
Inputs:  Temperature (C), pH
Output:  Predicted fish weight (grams)

Runs in SIMULATION MODE if Arduino is not connected.
Automatically switches to LIVE MODE when Arduino is plugged in.
"""

import numpy as np
import time
import requests
import threading
from collections import deque

# ── CONFIG ──
SERIAL_PORT = 'COM8'        # Change to your Arduino COM port
BAUD_RATE   = 9600
SERVER_URL  = 'http://localhost:5050'

# ════════════════════════════════════════════════
# ANFIS IMPLEMENTATION (pure numpy, no libraries)
# ════════════════════════════════════════════════

class GaussianMF:
    """Gaussian membership function"""
    def __init__(self, c, sigma):
        self.c     = c
        self.sigma = sigma

    def forward(self, x):
        return np.exp(-0.5 * ((x - self.c) / (self.sigma + 1e-8)) ** 2)

    def grad(self, x):
        mu     = self.forward(x)
        dc     = mu * (x - self.c)     / (self.sigma ** 2 + 1e-8)
        dsigma = mu * (x - self.c) **2 / (self.sigma ** 3 + 1e-8)
        return dc, dsigma


class ANFIS:
    """
    5-Layer ANFIS
    Input 1: Temperature (20-33 C)
    Input 2: pH (6.0-9.0)
    Output:  Fish weight (grams)
    """
    def __init__(self, n_mf=3):
        self.n_mf    = n_mf
        self.n_rules = n_mf ** 2

        # Centers for temperature MFs
        temp_centers = np.linspace(20, 32, n_mf)
        ph_centers   = np.linspace(6.0, 9.0, n_mf)

        self.mf1 = [GaussianMF(c, 2.0) for c in temp_centers]
        self.mf2 = [GaussianMF(c, 0.5) for c in ph_centers]

        # Consequent params [p, q, r] per rule
        self.C  = np.random.randn(self.n_rules, 3) * 0.1
        self.lr_mf = 0.005
        self.lr_c  = 0.01

    def forward(self, temp, ph):
        # Layer 1: Fuzzify
        mu1 = np.array([mf.forward(temp) for mf in self.mf1])
        mu2 = np.array([mf.forward(ph)   for mf in self.mf2])

        # Layer 2: Firing strength
        w = np.outer(mu1, mu2).flatten()

        # Layer 3: Normalize
        w_bar = w / (np.sum(w) + 1e-8)

        # Layer 4-5: Consequent + output
        x      = np.array([temp, ph, 1.0])
        f      = self.C @ x
        output = np.sum(w_bar * f)
        return output, w, w_bar, mu1, mu2

    def train_step(self, temp, ph, target):
        output, w, w_bar, mu1, mu2 = self.forward(temp, ph)
        error = output - target
        x     = np.array([temp, ph, 1.0])

        # Update consequents
        self.C -= self.lr_c * np.outer(w_bar, x) * error

        # Update MF params
        f     = self.C @ x
        sum_w = np.sum(w) + 1e-8

        for i, (i1, i2) in enumerate(np.ndindex(self.n_mf, self.n_mf)):
            dE_dw = (f[i] - output) / sum_w * error
            dc1, ds1 = self.mf1[i1].grad(temp)
            self.mf1[i1].c     -= self.lr_mf * dE_dw * mu2[i2] * dc1
            self.mf1[i1].sigma -= self.lr_mf * dE_dw * mu2[i2] * ds1
            dc2, ds2 = self.mf2[i2].grad(ph)
            self.mf2[i2].c     -= self.lr_mf * dE_dw * mu1[i1] * dc2
            self.mf2[i2].sigma -= self.lr_mf * dE_dw * mu1[i1] * ds2

        return abs(error)

    def predict(self, temp, ph):
        output, *_ = self.forward(temp, ph)
        return max(0, output)


# ════════════════════════════════════════════════
# TRAINING DATA
# Based on real fisheries biology:
# Optimal temp: 26-28C, Optimal pH: 7.0-7.5
# ════════════════════════════════════════════════

def generate_training_data(n=500):
    np.random.seed(42)
    temps = np.random.uniform(20, 33, n)
    phs   = np.random.uniform(5.8, 9.0, n)

    temp_response = np.exp(-0.5 * ((temps - 27) / 3.5) ** 2)
    ph_response   = np.exp(-0.5 * ((phs   - 7.2) / 0.8) ** 2)

    weights = 400 * temp_response * ph_response + np.random.normal(0, 15, n)
    weights = np.clip(weights, 30, 500)
    return temps, phs, weights


def train_anfis(epochs=80):
    print("\n" + "="*50)
    print("  ANFIS Digital Twin — Training")
    print("="*50)

    temps, phs, weights = generate_training_data(500)
    w_mean = weights.mean()
    w_std  = weights.std()
    w_norm = (weights - w_mean) / w_std

    model = ANFIS(n_mf=3)
    print(f"  Rules  : {model.n_rules}")
    print(f"  Epochs : {epochs}")
    print(f"  Samples: 500")
    print("="*50)

    for epoch in range(epochs):
        idx = np.random.permutation(len(temps))
        total_err = sum(model.train_step(temps[i], phs[i], w_norm[i]) for i in idx)
        if (epoch+1) % 20 == 0:
            mae = total_err / len(temps) * w_std
            print(f"  Epoch {epoch+1:3d}/{epochs} | MAE: {mae:.2f}g")

    print("  Training complete!")
    print("="*50 + "\n")
    return model, w_mean, w_std


# ════════════════════════════════════════════════
# ARDUINO READER
# Auto-detects connection, falls back to simulation
# ════════════════════════════════════════════════

class ArduinoReader:
    def __init__(self, port, baud):
        self.temp      = 27.0
        self.ph        = 7.2
        self.connected = False
        self._try_connect(port, baud)

    def _try_connect(self, port, baud):
        try:
            import serial
            self.ser       = serial.Serial(port, baud, timeout=1)
            time.sleep(2)
            self.connected = True
            print(f"[Arduino] Connected on {port}")
        except Exception as e:
            print(f"[Arduino] Not connected ({e})")
            print("[Arduino] Running in SIMULATION MODE")

    def read_loop(self):
        while True:
            if self.connected:
                try:
                    line = self.ser.readline().decode('utf-8').strip()
                    if 'TEMP:' in line and 'PH:' in line:
                        parts      = dict(p.split(':') for p in line.split(','))
                        self.temp  = float(parts.get('TEMP', self.temp))
                        self.ph    = float(parts.get('PH',   self.ph))
                except:
                    pass
            else:
                # Simulate realistic sensor values
                self.temp += np.random.normal(0, 0.08)
                self.ph   += np.random.normal(0, 0.015)
                self.temp  = float(np.clip(self.temp, 20, 33))
                self.ph    = float(np.clip(self.ph,  6.0, 9.0))
            time.sleep(0.5)

    def get(self):
        return round(float(self.temp), 2), round(float(self.ph), 3)


# ════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════

def run():
    # Train ANFIS
    model, w_mean, w_std = train_anfis(epochs=80)

    # Connect Arduino (or simulate)
    arduino = ArduinoReader(SERIAL_PORT, BAUD_RATE)

    # Start sensor reading thread
    t = threading.Thread(target=arduino.read_loop, daemon=True)
    t.start()

    pred_history   = deque(maxlen=60)
    sensor_history = deque(maxlen=60)

    print("[Digital Twin] Sending predictions to dashboard...")
    print("Press Ctrl+C to stop.\n")

    while True:
        temp, ph = arduino.get()

        # ANFIS prediction
        pred_norm   = model.predict(temp, ph)
        pred_weight = max(30, round(pred_norm * w_std + w_mean, 1))

        # Growth condition score (0-100)
        temp_score = 100 * np.exp(-0.5 * ((temp - 27) / 3.5) ** 2)
        ph_score   = 100 * np.exp(-0.5 * ((ph   - 7.2) / 0.8) ** 2)
        condition  = round((temp_score + ph_score) / 2, 1)

        status = "Optimal" if condition >= 75 else "Moderate" if condition >= 50 else "Poor"

        now = time.time()
        pred_history.append({"timestamp": now, "predicted_weight": pred_weight, "condition": condition})
        sensor_history.append({"timestamp": now, "temp": temp, "ph": ph})

        payload = {
            "temp":              temp,
            "ph":                ph,
            "predicted_weight":  pred_weight,
            "condition_score":   condition,
            "growth_status":     status,
            "pred_history":      list(pred_history),
            "sensor_history":    list(sensor_history),
            "arduino_connected": arduino.connected,
        }

        try:
            requests.post(f"{SERVER_URL}/sensor", json=payload, timeout=0.5)
        except:
            pass

        print(f"  Temp: {temp}°C | pH: {ph} | Weight: {pred_weight}g | Condition: {condition}% [{status}]")
        time.sleep(1)


if __name__ == "__main__":
    run()