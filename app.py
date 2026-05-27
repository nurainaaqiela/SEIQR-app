import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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
# SOLVER
# =========================
def run_model(beta, sigma, gamma, gamma_q, delta, mu, Lambda):

    y0 = [1000, 10, 5, 0, 0]
    t_eval = np.linspace(0, 160, 800)

    sol = solve_ivp(
        lambda t, y: seiqr(t, y, beta, sigma, gamma, gamma_q, delta, mu, Lambda),
        (0, 160),
        y0,
        t_eval=t_eval
    )

    return sol


# =========================
# R0 FUNCTION
# =========================
def compute_r0(beta, gamma, delta, mu):
    return beta / (gamma + delta + mu)


# =========================
# UI
# =========================
st.title("🦠 SEIQR Epidemiology Simulator (Fixed Version)")

st.markdown("""
This interactive model simulates disease spread and quarantine effects.
Now supports realistic R₀ < 1 and R₀ > 1 behavior.
""")

# =========================
# MODE
# =========================
mode = st.sidebar.radio(
    "Simulation Mode",
    ["Without Quarantine", "With Quarantine"]
)

st.sidebar.header("Model Parameters")

# =========================
# FIXED SLIDERS (IMPORTANT CHANGE)
# =========================

beta = st.sidebar.slider(
    "Transmission rate (β)",
    min_value=0.0,
    max_value=2.0,
    value=0.5,
    step=0.01
)

sigma = st.sidebar.slider("Incubation rate (σ)", 0.1, 1.0, 0.5, 0.1)
gamma = st.sidebar.slider("Recovery rate (γ)", 0.1, 1.0, 0.3, 0.1)
gamma_q = st.sidebar.slider("Quarantine recovery rate (γq)", 0.1, 1.0, 0.1, 0.1)
delta_input = st.sidebar.slider("Quarantine rate (δ)", 0.0, 0.5, 0.2, 0.01)

mu = st.sidebar.slider("Natural death rate (μ)", 0.0, 0.05, 0.01, 0.001)
Lambda = st.sidebar.slider("Birth rate (Λ)", 0.0, 20.0, 10.0, 0.5)

# =========================
# APPLY MODE
# =========================
delta = 0.0 if mode == "Without Quarantine" else delta_input

# =========================
# RUN SIMULATION
# =========================
sol = run_model(beta, sigma, gamma, gamma_q, delta, mu, Lambda)

# =========================
# METRICS
# =========================
I_peak = np.max(sol.y[2])
t_peak = sol.t[np.argmax(sol.y[2])]

R0 = compute_r0(beta, gamma, delta, mu)

st.subheader("📊 Key Epidemiological Indicators")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Peak Infected", f"{I_peak:.2f}")

with col2:
    st.metric("Time of Peak", f"{t_peak:.1f} days")

with col3:
    st.metric("R₀ Value", f"{R0:.2f}")

# =========================
# R0 INTERPRETATION
# =========================
if R0 < 1:
    st.success("🟢 Disease will die out (R₀ < 1)")
else:
    st.error("🔴 Outbreak likely (R₀ > 1)")

# =========================
# PLOT
# =========================
fig, ax = plt.subplots()

ax.plot(sol.t, sol.y[0], label="S")
ax.plot(sol.t, sol.y[1], label="E")
ax.plot(sol.t, sol.y[2], label="I")
ax.plot(sol.t, sol.y[3], label="Q")
ax.plot(sol.t, sol.y[4], label="R")

ax.set_xlabel("Time (days)")
ax.set_ylabel("Population")
ax.set_title(f"SEIQR Simulation - {mode}")
ax.legend()

st.pyplot(fig)