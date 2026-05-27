import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

st.set_page_config(page_title="SEIQR HFMD Simulator", layout="wide")

st.title("🚀 SEIQR HFMD Simulator")

# =========================
# ORIGINAL MODEL (UNCHANGED)
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
# SAFE DEFAULT INIT (IMPORTANT FIX)
# =========================
defaults = {
    "beta": 0.000003,
    "sigma": 0.2,
    "gamma": 0.3,
    "gamma_q": 0.25,
    "delta": 0.2,
    "mu": 0.01,
    "Lambda": 10
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Parameters")

beta = st.sidebar.slider("β", 0.0, 0.00001, float(st.session_state.beta))
sigma = st.sidebar.slider("σ", 0.0, 1.0, float(st.session_state.sigma))
gamma = st.sidebar.slider("γ", 0.0, 1.0, float(st.session_state.gamma))
gamma_q = st.sidebar.slider("γq", 0.0, 1.0, float(st.session_state.gamma_q))
delta = st.sidebar.slider("δ", 0.0, 1.0, float(st.session_state.delta))
mu = st.sidebar.slider("μ", 0.0, 0.1, float(st.session_state.mu))
Lambda = st.sidebar.slider("Λ", 0.0, 20.0, float(st.session_state.Lambda))

# =========================
# R0 (safe)
# =========================
R0 = (beta * sigma) / ((sigma + mu) * (gamma + delta + mu) + 1e-9)

st.metric("R₀", f"{R0:.3f}")

if R0 < 1:
    st.success("Disease dies out")
elif R0 < 1.5:
    st.warning("Endemic level")
else:
    st.error("Outbreak")

# =========================
# SIMULATION
# =========================
t_span = (0, 160)
t_eval = np.linspace(0, 160, 400)

y0 = [990, 5, 5, 0, 0]

sol = solve_ivp(
    seiqr,
    t_span,
    y0,
    t_eval=t_eval,
    args=(beta, sigma, gamma, gamma_q, delta, mu, Lambda)
)

S, E, I, Q, R = sol.y

fig, ax = plt.subplots()
ax.plot(S, label="S")
ax.plot(E, label="E")
ax.plot(I, label="I")
ax.plot(Q, label="Q")
ax.plot(R, label="R")
ax.legend()

st.pyplot(fig)