"""
arduino_reader.py — Reads sensors from Arduino and sends to Flask server
Sends: temp, ph, oxygen → /sensor endpoint
Run this in Terminal 3 (instead of anfis_engine.py)
"""

import serial
import requests
import time
import numpy as np

# ── CONFIG ──
SERIAL_PORT = "COM8"       # Change to your Arduino COM port
BAUD_RATE   = 9600
SERVER_URL  = "http://127.0.0.1:5050/sensor"

# ════════════════════════════════════════════════
# ANFIS IMPLEMENTATION (runs locally here)
# Predicts fish weight from temp + pH
# ════════════════════════════════════════════════
class GaussianMF:
    def __init__(self, c, sigma):
        self.c = c; self.sigma = sigma
    def forward(self, x):
        return np.exp(-0.5 * ((x - self.c) / (self.sigma + 1e-8)) ** 2)
    def grad(self, x):
        mu = self.forward(x)
        return mu*(x-self.c)/(self.sigma**2+1e-8), mu*(x-self.c)**2/(self.sigma**3+1e-8)

class ANFIS:
    def __init__(self, n_mf=3):
        self.n_mf    = n_mf
        self.n_rules = n_mf ** 2
        self.mf1 = [GaussianMF(c, 2.0) for c in np.linspace(20, 32, n_mf)]
        self.mf2 = [GaussianMF(c, 0.5) for c in np.linspace(6.0, 9.0, n_mf)]
        self.C   = np.random.randn(self.n_rules, 3) * 0.1
        self.lr_mf = 0.003; self.lr_c = 0.008

    def forward(self, temp, ph):
        mu1 = np.array([mf.forward(temp) for mf in self.mf1])
        mu2 = np.array([mf.forward(ph)   for mf in self.mf2])
        w   = np.outer(mu1, mu2).flatten()
        w_bar = w / (np.sum(w) + 1e-8)
        f   = self.C @ np.array([temp, ph, 1.0])
        return np.sum(w_bar * f), w, w_bar, mu1, mu2

    def train_step(self, temp, ph, target):
        output, w, w_bar, mu1, mu2 = self.forward(temp, ph)
        error = output - target
        x = np.array([temp, ph, 1.0])
        self.C -= self.lr_c * np.outer(w_bar, x) * error
        f = self.C @ x
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

def train_anfis():
    print("="*50)
    print("  Training ANFIS model...")
    print("="*50)
    np.random.seed(42)
    n = 1000
    temps = np.random.uniform(20, 33, n)
    phs   = np.random.uniform(5.8, 9.0, n)
    tr    = np.exp(-0.5*((temps-27)/3.5)**2)
    pr    = np.exp(-0.5*((phs-7.2)/0.8)**2)
    weights = np.clip(400*tr*pr + np.random.normal(0,15,n), 30, 500)
    w_mean, w_std = weights.mean(), weights.std()
    w_norm = (weights - w_mean) / w_std
    model = ANFIS(n_mf=3)
    for epoch in range(150):
        idx = np.random.permutation(n)
        err = sum(model.train_step(temps[i], phs[i], w_norm[i]) for i in idx)
        if (epoch+1) % 50 == 0:
            print(f"  Epoch {epoch+1}/150 | MAE: {err/n*w_std:.1f}g")
    print("  Training complete!")
    print("="*50)
    return model, w_mean, w_std

# ── Train ANFIS on startup ──
anfis_model, w_mean, w_std = train_anfis()

# ── Try connecting Arduino ──
arduino = None
try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"[Arduino] Connected on {SERIAL_PORT}")
except Exception as e:
    print(f"[Arduino] Not connected: {e}")
    print("[Arduino] Running in SIMULATION MODE")

# ── Simulation state ──
sim_temp = 27.0
sim_ph   = 7.2
sim_o2   = 7.5

print("\n[System] Sending data to dashboard...")
print("Press Ctrl+C to stop.\n")

while True:
    try:
        temp = ph = oxygen = None

        # ── Read from Arduino ──
        if arduino:
            try:
                line = arduino.readline().decode().strip()
                if line.startswith("DATA:"):
                    values  = line.replace("DATA:", "").split(",")
                    temp    = float(values[0])
                    ph      = float(values[1])
                    oxygen  = float(values[2])
            except:
                pass

        # ── Simulation fallback ──
        if temp is None:
            sim_temp += np.random.normal(0, 0.08)
            sim_ph   += np.random.normal(0, 0.015)
            sim_o2   += np.random.normal(0, 0.05)
            sim_temp  = float(np.clip(sim_temp, 20, 33))
            sim_ph    = float(np.clip(sim_ph,  6.0, 9.0))
            sim_o2    = float(np.clip(sim_o2,  4.0, 12.0))
            temp   = sim_temp
            ph     = sim_ph
            oxygen = sim_o2

        # ── ANFIS prediction ──
        pred_norm   = anfis_model.predict(temp, ph)
        pred_weight = max(30, round(pred_norm * w_std + w_mean, 1))

        # ── Growth condition score ──
        temp_score = 100 * np.exp(-0.5 * ((temp - 27) / 3.5) ** 2)
        ph_score   = 100 * np.exp(-0.5 * ((ph   - 7.2) / 0.8) ** 2)
        condition  = round((temp_score + ph_score) / 2, 1)
        status     = "Optimal" if condition >= 75 else "Moderate" if condition >= 50 else "Poor"

        payload = {
            "temp":              round(temp, 2),
            "ph":                round(ph, 3),
            "oxygen":            round(oxygen, 2),
            "predicted_weight":  pred_weight,
            "condition_score":   condition,
            "growth_status":     status,
            "arduino_connected": arduino is not None,
        }

        requests.post(SERVER_URL, json=payload, timeout=0.5)
        print(f"  Temp:{temp:.1f}°C | pH:{ph:.2f} | O2:{oxygen:.1f} | Weight:{pred_weight}g | {status}")

    except Exception as e:
        print(f"Error: {e}")

    time.sleep(0.5)