import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

st.set_page_config(page_title="SEIQR HFMD Simulator", layout="wide")

st.title("🚀 SEIQR HFMD Simulator (Stable Version)")

# =========================
# YOUR ORIGINAL MODEL (UNCHANGED)
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
# SAFE SLIDERS (NO SESSION STATE)
# =========================
st.sidebar.header("Parameters")

beta = st.sidebar.slider("β", 0.0, 0.00001, 0.000003)
sigma = st.sidebar.slider("σ", 0.0, 1.0, 0.2)
gamma = st.sidebar.slider("γ", 0.0, 1.0, 0.3)
gamma_q = st.sidebar.slider("γq", 0.0, 1.0, 0.25)
delta = st.sidebar.slider("δ", 0.0, 1.0, 0.2)
mu = st.sidebar.slider("μ", 0.0, 0.1, 0.01)
Lambda = st.sidebar.slider("Λ", 0.0, 20.0, 10.0)

# =========================
# R0 (SAFE)
# =========================
den = (sigma + mu) * (gamma + delta + mu)
R0 = (beta * sigma) / (den + 1e-9)

st.subheader("R₀")
st.metric("Value", f"{R0:.4f}")

if R0 < 1:
    st.success("🟢 Disease dies out")
elif R0 < 1.5:
    st.warning("🟡 Endemic level")
else:
    st.error("🔴 Outbreak expected")

# =========================
# INITIAL CONDITIONS
# =========================
y0 = [990, 5, 5, 0, 0]

# =========================
# TIME GRID (SAFE)
# =========================
t_span = (0, 160)
t_eval = np.linspace(0, 160, 300)

# =========================
# SOLVE SYSTEM (SAFE WRAPPER)
# =========================
try:
    sol = solve_ivp(
        seiqr,
        t_span,
        y0,
        t_eval=t_eval,
        args=(beta, sigma, gamma, gamma_q, delta, mu, Lambda),
        method="RK45"
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

except Exception as e:
    st.error("Simulation failed. Try adjusting parameters.")
    st.write(e)