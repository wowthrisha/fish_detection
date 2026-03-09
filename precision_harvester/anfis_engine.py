"""
hierarchical_anfis.py
Hierarchical ANFIS with Explainable AI
3 Sub-models → Master ANFIS → Final Score + Explanation

Run alongside arduino_reader.py:
  py server.py
  py main_yolo.py
  py hierarchical_anfis.py   ← replaces arduino_reader.py
"""

import numpy as np
import requests
import time
from itertools import product

# ══════════════════════════════════════════════
# MEMBERSHIP FUNCTIONS
# ══════════════════════════════════════════════
class GaussianMF:
    def __init__(self, center, sigma, label):
        self.c     = center
        self.sigma = sigma
        self.label = label  # e.g. "optimal", "cold", "acidic"

    def forward(self, x):
        return float(np.exp(-0.5 * ((x - self.c) / (self.sigma + 1e-8)) ** 2))

# ══════════════════════════════════════════════
# BASE ANFIS (single sub-model)
# ══════════════════════════════════════════════
class SubANFIS:
    def __init__(self, name, input_defs, n_mf=3, lr=0.005):
        """
        input_defs: list of (input_name, [MF defs])
        MF def: (center, sigma, label)
        """
        self.name       = name
        self.input_defs = input_defs
        self.n_inputs   = len(input_defs)
        self.n_mf       = n_mf
        self.n_rules    = n_mf ** self.n_inputs
        self.lr         = lr

        # Build MFs
        self.mfs = []
        for inp_name, mf_list in input_defs:
            row = [GaussianMF(c, s, lbl) for c, s, lbl in mf_list]
            self.mfs.append((inp_name, row))

        # Consequent parameters (Takagi-Sugeno)
        self.C = np.random.randn(self.n_rules, self.n_inputs + 1) * 0.05

        # Rule labels (for XAI)
        self.rule_labels = self._build_rule_labels()

    def _build_rule_labels(self):
        labels = []
        mf_labels = [[mf.label for mf in row] for _, row in self.mfs]
        for combo in product(*[range(self.n_mf)] * self.n_inputs):
            parts = []
            for inp_idx, mf_idx in enumerate(combo):
                inp_name  = self.mfs[inp_idx][0]
                mf_label  = self.mfs[inp_idx][1][mf_idx].label
                parts.append(f"{inp_name} is {mf_label}")
            labels.append(" AND ".join(parts))
        return labels

    def fuzzify(self, inputs):
        mu_all = []
        for i, (inp_name, mf_row) in enumerate(self.mfs):
            mu_all.append([mf.forward(inputs[i]) for mf in mf_row])
        return mu_all

    def forward(self, inputs):
        mu_all = self.fuzzify(inputs)

        # Rule firing strengths
        w = []
        for combo in product(*[range(self.n_mf)] * self.n_inputs):
            strength = 1.0
            for inp_idx, mf_idx in enumerate(combo):
                strength *= mu_all[inp_idx][mf_idx]
            w.append(strength)

        w     = np.array(w)
        w_sum = w.sum() + 1e-8
        w_bar = w / w_sum

        x_aug  = np.append(inputs, 1.0)
        f      = self.C @ x_aug
        output = float(np.dot(w_bar, f))

        return output, w, w_bar

    def train_step(self, inputs, target):
        output, w, w_bar = self.forward(inputs)
        error  = output - target
        x_aug  = np.append(inputs, 1.0)

        # Consequent update (least squares direction)
        self.C -= self.lr * np.outer(w_bar, x_aug) * error

        # MF update (backprop)
        f     = self.C @ x_aug
        w_sum = w.sum() + 1e-8
        for i, combo in enumerate(product(*[range(self.n_mf)] * self.n_inputs)):
            dE_dw = (f[i] - output) / w_sum * error
            for inp_idx, mf_idx in enumerate(combo):
                mf  = self.mfs[inp_idx][1][mf_idx]
                x_i = inputs[inp_idx]
                other_mu = 1.0
                for j, mf_j_idx in enumerate(combo):
                    if j != inp_idx:
                        other_mu *= self.mfs[j][1][mf_j_idx].forward(inputs[j])
                grad_c     = dE_dw * other_mu * (x_i - mf.c) / (mf.sigma ** 2 + 1e-8)
                grad_sigma = dE_dw * other_mu * (x_i - mf.c) ** 2 / (mf.sigma ** 3 + 1e-8)
                mf.c     -= self.lr * grad_c
                mf.sigma -= self.lr * grad_sigma
                mf.sigma  = max(0.01, mf.sigma)

        return abs(error)

    def explain(self, inputs, top_k=3):
        """
        Returns top-k fired rules with their contributions — XAI core
        """
        output, w, w_bar = self.forward(inputs)
        x_aug = np.append(inputs, 1.0)
        f     = self.C @ x_aug

        contributions = w_bar * f
        top_idx       = np.argsort(np.abs(contributions))[::-1][:top_k]

        explanations = []
        for idx in top_idx:
            explanations.append({
                "rule":         self.rule_labels[idx],
                "firing":       round(float(w_bar[idx]) * 100, 1),
                "contribution": round(float(contributions[idx]), 3),
                "impact":       "positive" if contributions[idx] > 0 else "negative",
            })

        return explanations, round(output, 4)

# ══════════════════════════════════════════════
# BUILD THE 3 SUB-MODELS
# ══════════════════════════════════════════════
def build_environment_anfis():
    """Inputs: temperature, dissolved_O2"""
    return SubANFIS(
        name = "Environment",
        input_defs = [
            ("Temperature", [
                (18, 3.0, "very_cold"),
                (27, 2.5, "optimal"),
                (34, 3.0, "very_hot"),
            ]),
            ("Dissolved_O2", [
                (3,  1.5, "critical"),
                (7,  1.5, "optimal"),
                (11, 1.5, "high"),
            ]),
        ]
    )

def build_chemistry_anfis():
    """Inputs: pH, ammonia (NH3 derived)"""
    return SubANFIS(
        name = "Chemistry",
        input_defs = [
            ("pH", [
                (5.5, 0.6, "acidic"),
                (7.2, 0.5, "neutral"),
                (9.0, 0.8, "alkaline"),
            ]),
            ("Ammonia_NH3", [
                (0.1, 0.08, "safe"),
                (0.8, 0.3,  "moderate"),
                (2.5, 0.8,  "dangerous"),
            ]),
        ]
    )

def build_population_anfis():
    """Inputs: fish_count, juvenile_percentage"""
    return SubANFIS(
        name = "Population",
        input_defs = [
            ("Fish_Count", [
                (1,  1.0, "sparse"),
                (5,  2.0, "normal"),
                (12, 3.0, "dense"),
            ]),
            ("Juvenile_Pct", [
                (5,  4.0, "low"),
                (25, 8.0, "moderate"),
                (60, 15., "high"),
            ]),
        ]
    )

def build_master_anfis():
    """Inputs: env_score, chem_score, pop_score"""
    return SubANFIS(
        name = "Master",
        input_defs = [
            ("Env_Score", [
                (0.2, 0.15, "poor"),
                (0.6, 0.18, "moderate"),
                (0.9, 0.12, "excellent"),
            ]),
            ("Chem_Score", [
                (0.2, 0.15, "poor"),
                (0.6, 0.18, "moderate"),
                (0.9, 0.12, "excellent"),
            ]),
            ("Pop_Score", [
                (0.2, 0.15, "stressed"),
                (0.6, 0.18, "stable"),
                (0.9, 0.12, "healthy"),
            ]),
        ]
    )

# ══════════════════════════════════════════════
# TRAIN ALL MODELS
# ══════════════════════════════════════════════
def train_all(epochs=120, n=800):
    print("=" * 55)
    print("  Hierarchical ANFIS — Training All Sub-Models")
    print("=" * 55)
    np.random.seed(42)

    env_model  = build_environment_anfis()
    chem_model = build_chemistry_anfis()
    pop_model  = build_population_anfis()
    mst_model  = build_master_anfis()

    # ── Generate training data ──
    temps      = np.random.uniform(15, 38, n)
    phs        = np.random.uniform(5.0, 10.0, n)
    o2s        = np.clip(14.6 - 0.38 * temps + np.random.normal(0, 0.5, n), 1, 14)
    nh3s       = np.clip((temps - 20) * 0.03 + np.random.normal(0, 0.05, n), 0.01, 5)
    fish_cnts  = np.random.randint(0, 15, n).astype(float)
    juv_pcts   = np.random.uniform(0, 80, n)

    # Ground truth scores (biological model)
    env_gt  = np.exp(-0.5*((temps-27)/4)**2) * np.exp(-0.5*((o2s-7)/2)**2)
    chem_gt = np.exp(-0.5*((phs-7.2)/0.8)**2) * np.exp(-0.5*((nh3s-0.1)/0.3)**2)
    pop_gt  = np.exp(-0.5*((juv_pcts-5)/15)**2) * np.clip(fish_cnts/8, 0.1, 1.0)
    mst_gt  = (0.4*env_gt + 0.35*chem_gt + 0.25*pop_gt)

    print(f"\n  Training Environment ANFIS...")
    for ep in range(epochs):
        idx = np.random.permutation(n)
        for i in idx:
            env_model.train_step([temps[i], o2s[i]], env_gt[i])
        if (ep+1) % 40 == 0:
            print(f"    Epoch {ep+1}/{epochs} ✓")

    print(f"  Training Chemistry ANFIS...")
    for ep in range(epochs):
        idx = np.random.permutation(n)
        for i in idx:
            chem_model.train_step([phs[i], nh3s[i]], chem_gt[i])
        if (ep+1) % 40 == 0:
            print(f"    Epoch {ep+1}/{epochs} ✓")

    print(f"  Training Population ANFIS...")
    for ep in range(epochs):
        idx = np.random.permutation(n)
        for i in idx:
            pop_model.train_step([fish_cnts[i], juv_pcts[i]], pop_gt[i])
        if (ep+1) % 40 == 0:
            print(f"    Epoch {ep+1}/{epochs} ✓")

    print(f"  Training Master ANFIS...")
    for ep in range(epochs):
        idx = np.random.permutation(n)
        for i in idx:
            env_s  = env_model.forward([temps[i], o2s[i]])[0]
            chem_s = chem_model.forward([phs[i], nh3s[i]])[0]
            pop_s  = pop_model.forward([fish_cnts[i], juv_pcts[i]])[0]
            mst_model.train_step([env_s, chem_s, pop_s], mst_gt[i])
        if (ep+1) % 40 == 0:
            print(f"    Epoch {ep+1}/{epochs} ✓")

    print("\n  All models trained!\n" + "=" * 55)
    return env_model, chem_model, pop_model, mst_model

# ══════════════════════════════════════════════
# EXPLAINABILITY ENGINE
# ══════════════════════════════════════════════
def generate_explanation(env_exp, chem_exp, pop_exp, mst_exp,
                         env_s, chem_s, pop_s, final_score,
                         temp, ph, o2, nh3, fish_count, juv_pct):
    """
    Generate human-readable explanation of why the score is what it is.
    This is the XAI layer.
    """

    # Dominant factor
    scores = {"Environment": env_s, "Chemistry": chem_s, "Population": pop_s}
    dominant = min(scores, key=scores.get)  # lowest score = biggest problem

    # Factor contributions (weighted)
    env_contrib  = round(env_s  * 0.40 * 100, 1)
    chem_contrib = round(chem_s * 0.35 * 100, 1)
    pop_contrib  = round(pop_s  * 0.25 * 100, 1)

    # Grade
    pct = final_score * 100
    grade = "A" if pct>=85 else "B" if pct>=70 else "C" if pct>=55 else "D" if pct>=40 else "F"

    # Build natural language reasons
    reasons = []
    if temp > 30:
        reasons.append(f"Temperature critically high ({temp:.1f}°C) — Aeromonas risk")
    elif temp < 20:
        reasons.append(f"Temperature too low ({temp:.1f}°C) — metabolism suppressed")
    if ph < 6.5:
        reasons.append(f"pH dangerously acidic ({ph:.1f}) — gill damage likely")
    elif ph > 8.5:
        reasons.append(f"pH too alkaline ({ph:.1f}) — ammonia toxicity increases")
    if o2 < 5:
        reasons.append(f"Dissolved O₂ critically low ({o2:.1f} mg/L) — suffocation risk")
    if nh3 > 0.5:
        reasons.append(f"Ammonia elevated ({nh3:.2f} ppm) — immune suppression")
    if juv_pct > 30:
        reasons.append(f"High juvenile catch ({juv_pct:.0f}%) — population stressed")
    if fish_count == 0:
        reasons.append("No fish detected in frame")

    if not reasons:
        reasons.append("All parameters within optimal range — pond is healthy")

    return {
        "final_score":     round(pct, 1),
        "grade":           grade,
        "dominant_factor": dominant,
        "env_score":       round(env_s * 100, 1),
        "chem_score":      round(chem_s * 100, 1),
        "pop_score":       round(pop_s * 100, 1),
        "env_contrib":     env_contrib,
        "chem_contrib":    chem_contrib,
        "pop_contrib":     pop_contrib,
        "env_rules":       env_exp,
        "chem_rules":      chem_exp,
        "pop_rules":       pop_exp,
        "master_rules":    mst_exp,
        "reasons":         reasons,
        "recommendation":  _recommend(dominant, temp, ph, o2, nh3, juv_pct),
    }

def _recommend(dominant, temp, ph, o2, nh3, juv_pct):
    if dominant == "Environment":
        if temp > 30: return "Activate cooling system or increase water circulation immediately"
        if temp < 20: return "Apply pond heater or reduce water depth to allow solar warming"
        if o2 < 5:   return "Activate aerators immediately — oxygen critical"
    if dominant == "Chemistry":
        if ph < 6.5: return "Add agricultural lime (CaCO3) to raise pH — target 7.0-7.5"
        if ph > 8.5: return "Add CO2 or organic matter to lower pH"
        if nh3 > 0.5: return "Perform 30% water exchange and reduce feeding immediately"
    if dominant == "Population":
        if juv_pct > 30: return "Release juvenile catch — do not harvest until size threshold met"
    return "Maintain current conditions — all parameters stable"

# ══════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════
env_model, chem_model, pop_model, mst_model = train_all()

# Simulation state
sim_temp = 27.0; sim_ph = 7.2; sim_o2 = 7.5

print("\n[XAI Engine] Running — sending to dashboard...\n")

while True:
    try:
        # ── Get sensor data ──
        temp = ph = o2 = None
        try:
            r    = requests.get("http://127.0.0.1:5050/sensor_data", timeout=0.3)
            d    = r.json()
            temp = d.get("temp"); ph = d.get("ph"); o2 = d.get("oxygen")
        except: pass

        # Simulation fallback
        if not temp:
            sim_temp += np.random.normal(0, 0.1)
            sim_ph   += np.random.normal(0, 0.02)
            sim_o2   += np.random.normal(0, 0.05)
            sim_temp  = float(np.clip(sim_temp, 15, 38))
            sim_ph    = float(np.clip(sim_ph, 5.0, 10.0))
            sim_o2    = float(np.clip(sim_o2, 1.0, 14.0))
            temp = sim_temp; ph = sim_ph; o2 = sim_o2

        # ── Get fish data ──
        fish_count = 0; juv_pct = 0
        try:
            r2         = requests.get("http://127.0.0.1:5050/data", timeout=0.3)
            d2         = r2.json()
            fish_count = d2.get("total_count", 0)
            juv_pct    = d2.get("juvenile_percentage", 0)
        except: pass

        # Derive NH3 biologically
        nh3 = float(np.clip((temp - 20) * 0.025 + (ph - 7) * 0.05
                            + np.random.normal(0, 0.02), 0.01, 5.0))

        # ── Run Hierarchical ANFIS ──
        env_exp,  env_s  = env_model.explain([temp, o2])
        chem_exp, chem_s = chem_model.explain([ph, nh3])
        pop_exp,  pop_s  = pop_model.explain([float(fish_count), float(juv_pct)])
        mst_exp,  final  = mst_model.explain([env_s, chem_s, pop_s])

        # ── Generate XAI explanation ──
        xai = generate_explanation(
            env_exp, chem_exp, pop_exp, mst_exp,
            env_s, chem_s, pop_s, final,
            temp, ph, o2, nh3, fish_count, juv_pct
        )

        # ── Send to dashboard ──
        payload = {
            "temp":              round(temp, 2),
            "ph":                round(ph, 3),
            "oxygen":            round(o2, 2),
            "predicted_weight":  round(env_s * 400 + 30, 1),
            "condition_score":   xai["final_score"],
            "growth_status":     xai["grade"],
            "arduino_connected": False,
            "xai":               xai,
        }
        requests.post("http://127.0.0.1:5050/sensor", json=payload, timeout=0.5)

        # ── Console output ──
        print(f"\n{'='*55}")
        print(f"  HIERARCHICAL ANFIS + XAI REPORT")
        print(f"{'='*55}")
        print(f"  Temp:{temp:.1f}°C  pH:{ph:.2f}  O2:{o2:.1f}  NH3:{nh3:.2f}")
        print(f"  Fish:{fish_count}  Juv%:{juv_pct:.0f}%")
        print(f"{'─'*55}")
        print(f"  Env  Score : {xai['env_score']:.1f}%  → {env_exp[0]['rule']}")
        print(f"  Chem Score : {xai['chem_score']:.1f}%  → {chem_exp[0]['rule']}")
        print(f"  Pop  Score : {xai['pop_score']:.1f}%  → {pop_exp[0]['rule']}")
        print(f"{'─'*55}")
        print(f"  FINAL SCORE : {xai['final_score']:.1f}%  GRADE: {xai['grade']}")
        print(f"  DOMINANT    : {xai['dominant_factor']}")
        print(f"  REASON      : {xai['reasons'][0]}")
        print(f"  ACTION      : {xai['recommendation']}")
        print(f"{'='*55}")

    except Exception as e:
        print(f"Error: {e}")

    time.sleep(1)