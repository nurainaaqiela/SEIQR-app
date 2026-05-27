import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="SEIQR HFMD Simulator", layout="wide")

st.title("🚀 SEIQR HFMD Epidemic Simulator")
st.write("Interactive model with R₀ feedback, presets, and outbreak prediction")

# =========================
# R0 FUNCTION (simplified SEIQR)
# R0 = beta / (gamma + delta + mu)
# =========================
def compute_R0(beta, gamma, delta, mu):
    return beta / (gamma + delta + mu + 1e-9)

# =========================
# PRESETS
# =========================
presets = {
    "🟢 Disease dies out (R₀ < 1)": {
        "beta": 0.002,
        "gamma": 0.3,
        "delta": 0.4,
        "mu": 0.01,
        "sigma": 0.2
    },
    "🔴 Outbreak scenario (R₀ > 1)": {
        "beta": 0.006,
        "gamma": 0.15,
        "delta": 0.1,
        "mu": 0.01,
        "sigma": 0.1
    },
    "⚖️ Balanced endemic": {
        "beta": 0.004,
        "gamma": 0.25,
        "delta": 0.2,
        "mu": 0.01,
        "sigma": 0.15
    }
}

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header("⚙️ Controls")

preset_choice = st.sidebar.selectbox("Choose preset", list(presets.keys()))

if st.sidebar.button("Apply preset"):
    st.session_state.update(presets[preset_choice])

# default values (safe init)
beta = st.sidebar.slider("β (infection rate)", 0.0, 0.01, st.session_state.get("beta", 0.003))
gamma = st.sidebar.slider("γ (recovery rate)", 0.0, 1.0, st.session_state.get("gamma", 0.2))
delta = st.sidebar.slider("δ (quarantine rate)", 0.0, 1.0, st.session_state.get("delta", 0.2))
mu = st.sidebar.slider("μ (natural loss)", 0.0, 0.1, st.session_state.get("mu", 0.01))
sigma = st.sidebar.slider("σ (incubation)", 0.0, 1.0, st.session_state.get("sigma", 0.1))

# =========================
# R0 CALCULATION
# =========================
R0 = compute_R0(beta, gamma, delta, mu)

# =========================
# OUTBREAK MESSAGE
# =========================
if R0 < 1:
    status = "🟢 Disease will DIE OUT"
elif R0 < 1.5:
    status = "🟡 Low-level transmission"
else:
    status = "🔴 ⚠️ OUTBREAK EXPECTED"

st.subheader("📊 Epidemic Status")
st.write(status)

# =========================
# R0 GAUGE (simple bar)
# =========================
st.subheader("📈 R₀ Indicator")

col1, col2, col3 = st.columns(3)
col1.metric("R₀ Value", f"{R0:.2f}")

progress = min(R0 / 3.0, 1.0)
st.progress(progress)

st.caption("Green < 1 | Yellow 1–1.5 | Red > 1.5")

# =========================
# REAL-WORLD INTERPRETATION
# =========================
st.subheader("🌍 Real-world interpretation")

st.write(f"""
- β (contact rate): school crowding, hygiene level  
- γ (recovery): healthcare efficiency  
- δ (quarantine): isolation effectiveness  
- σ (incubation): environmental / temperature influence  
""")

# =========================
# SIMPLE SEIQR SIMULATION
# =========================
st.subheader("📉 Infection Curve Simulation")

N = 1000
days = 160

S = np.zeros(days)
E = np.zeros(days)
I = np.zeros(days)
Q = np.zeros(days)
R = np.zeros(days)

S[0] = N - 1
E[0] = 0
I[0] = 1
Q[0] = 0
R[0] = 0

dt = 1

for t in range(1, days):
    dS = -beta * S[t-1] * I[t-1] / N
    dE = beta * S[t-1] * I[t-1] / N - sigma * E[t-1]
    dI = sigma * E[t-1] - (gamma + delta) * I[t-1]
    dQ = delta * I[t-1] - gamma * Q[t-1]
    dR = gamma * (I[t-1] + Q[t-1])

    S[t] = S[t-1] + dS * dt
    E[t] = E[t-1] + dE * dt
    I[t] = I[t-1] + dI * dt
    Q[t] = Q[t-1] + dQ * dt
    R[t] = R[t-1] + dR * dt

fig, ax = plt.subplots()
ax.plot(I, label="Infected")
ax.plot(E, label="Exposed")
ax.plot(Q, label="Quarantined")
ax.plot(R, label="Recovered")
ax.set_title("SEIQR Dynamics")
ax.legend()

st.pyplot(fig)

# =========================
# SUMMARY INSIGHT
# =========================
st.subheader("🧠 Automatic Insight")

if R0 < 1:
    st.success("No major outbreak. Control measures are effective.")
elif R0 < 1.5:
    st.warning("Endemic persistence possible. Improve quarantine or reduce contact rate.")
else:
    st.error("High outbreak risk. Immediate intervention required (reduce β or increase δ).")