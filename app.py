import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

st.set_page_config(page_title="SEIQR HFMD Simulator", layout="wide")

st.title("🚀 SEIQR HFMD Simulator")

# =========================
# SEIQR MODEL
# =========================
def seiqr(t, y, beta, sigma, gamma, gamma_q, delta, mu, Lambda):
    S, E, I, Q, R = y

    dS = Lambda - beta * S * I - mu * S
    dE = beta * S * I - (sigma + mu) * E
    dI = sigma * E - (gamma + delta + mu) * I
    dQ = delta * I - (gamma_q + mu) * Q
    dR = gamma * I + gamma_q * Q - mu * R

    return [dS, dE, dI, dQ, dR]

# =========================
# R0 (consistent approximation for your model)
# =========================
def compute_R0(beta, sigma, gamma, gamma_q, delta, mu):
    return (beta * sigma) / ((sigma + mu) * (gamma + delta + mu))

# =========================
# PRESETS (CONSISTENT WITH YOUR MODEL)
# =========================
presets = {
    "🟢 Controlled (R₀ < 1)": {
        "beta": 0.000002,
        "sigma": 0.2,
        "gamma": 0.3,
        "gamma_q": 0.25,
        "delta": 0.4,
        "mu": 0.01,
        "Lambda": 10
    },
    "🔴 Outbreak (R₀ > 1)": {
        "beta": 0.000006,
        "sigma": 0.15,
        "gamma": 0.15,
        "gamma_q": 0.1,
        "delta": 0.1,
        "mu": 0.01,
        "Lambda": 10
    },
    "⚖️ Medium": {
        "beta": 0.000004,
        "sigma": 0.18,
        "gamma": 0.25,
        "gamma_q": 0.2,
        "delta": 0.2,
        "mu": 0.01,
        "Lambda": 10
    }
}

# =========================
# SIDEBAR CONTROLS (MODEL UNCHANGED)
# =========================
st.sidebar.header("⚙️ Parameters")

preset = st.sidebar.selectbox("Choose preset", list(presets.keys()))

if st.sidebar.button("Apply preset"):
    st.session_state.update(presets[preset])

beta = st.sidebar.slider("β (infection rate)", 0.0, 0.00001, st.session_state.get("beta", 0.000003))
sigma = st.sidebar.slider("σ (incubation rate)", 0.0, 1.0, st.session_state.get("sigma", 0.2))
gamma = st.sidebar.slider("γ (recovery rate)", 0.0, 1.0, st.session_state.get("gamma", 0.3))
gamma_q = st.sidebar.slider("γq (quarantine recovery)", 0.0, 1.0, st.session_state.get("gamma_q", 0.25))
delta = st.sidebar.slider("δ (quarantine rate)", 0.0, 1.0, st.session_state.get("delta", 0.2))
mu = st.sidebar.slider("μ (natural death)", 0.0, 0.1, st.session_state.get("mu", 0.01))
Lambda = st.sidebar.slider("Λ (birth rate)", 0.0, 20.0, st.session_state.get("Lambda", 10))

# =========================
# R0 CALCULATION
# =========================
R0 = compute_R0(beta, sigma, gamma, gamma_q, delta, mu)

st.subheader("📊 R₀ Value")
st.metric("R₀", f"{R0:.3f}")

# =========================
# OUTBREAK CLASSIFICATION
# =========================
if R0 < 1:
    st.success("🟢 Disease dies out")
elif R0 < 1.5:
    st.warning("🟡 Endemic / slow spread")
else:
    st.error("🔴 OUTBREAK EXPECTED")

st.progress(min(R0 / 3.0, 1.0))

# =========================
# REAL-WORLD INTERPRETATION
# =========================
st.subheader("🌍 Parameter Meaning (HFMD Context)")
st.write("""
- β: contact rate (school transmission)
- σ: incubation rate
- γ: recovery rate (mild cases)
- γq: recovery in quarantine
- δ: isolation rate
- μ: natural death rate
- Λ: population inflow (births/entry)
""")

# =========================
# SIMULATION
# =========================
st.subheader("📉 SEIQR Dynamics")

t_span = (0, 160)
t_eval = np.linspace(0, 160, 400)

y0 = [990, 5, 5, 0, 0]  # initial conditions

sol = solve_ivp(
    seiqr,
    t_span,
    y0,
    t_eval=t_eval,
    args=(beta, sigma, gamma, gamma_q, delta, mu, Lambda)
)

S, E, I, Q, R = sol.y

fig, ax = plt.subplots()
ax.plot(S, label="Susceptible")
ax.plot(E, label="Exposed")
ax.plot(I, label="Infected")
ax.plot(Q, label="Quarantined")
ax.plot(R, label="Recovered")
ax.legend()
st.pyplot(fig)

# =========================
# AUTOMATIC INSIGHT
# =========================
st.subheader("🧠 Epidemic Insight")

if R0 < 1:
    st.info("Transmission will fade out naturally under current conditions.")
elif R0 < 1.5:
    st.warning("Disease may persist — improve quarantine or reduce contact.")
else:
    st.error("High outbreak risk — immediate intervention required.")