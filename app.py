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
    t_eval = np.linspace(0, 160, 1000)

    sol = solve_ivp(
        lambda t, y: seiqr(t, y, beta, sigma, gamma, gamma_q, delta, mu, Lambda),
        (0, 160),
        y0,
        t_eval=t_eval
    )

    return sol


# =========================
# UI
# =========================
st.title("🦠 SEIQR Disease Simulator (Multi-Mode)")

st.markdown("Choose simulation mode and explore disease dynamics interactively.")

# -------------------------
# MODE SELECTOR
# -------------------------
mode = st.sidebar.radio(
    "Select Simulation Mode",
    ["Without Quarantine", "With Quarantine"]
)

st.sidebar.header("Parameters")

beta = st.sidebar.number_input("Transmission rate (β)", 0.0001, 0.01, 0.002, 0.0001)
sigma = st.sidebar.slider("Incubation rate (σ)", 0.1, 1.0, 0.5, 0.1)
gamma = st.sidebar.slider("Recovery rate (γ)", 0.1, 1.0, 0.3, 0.1)
gamma_q = st.sidebar.slider("Quarantine recovery rate (γq)", 0.1, 1.0, 0.1, 0.1)
delta_input = st.sidebar.slider("Quarantine rate (δ)", 0.0, 0.5, 0.2, 0.01)

mu = st.sidebar.number_input("Natural death rate (μ)", 0.0, 0.05, 0.01, 0.001)
Lambda = st.sidebar.number_input("Birth rate (Λ)", 0.0, 20.0, 10.0, 0.5)


# =========================
# APPLY MODE LOGIC
# =========================
if mode == "Without Quarantine":
    delta = 0.0
else:
    delta = delta_input


# =========================
# RUN SIMULATION
# =========================
sol = run_model(beta, sigma, gamma, gamma_q, delta, mu, Lambda)


# =========================
# METRICS
# =========================
I_peak = np.max(sol.y[2])
t_peak = sol.t[np.argmax(sol.y[2])]

st.subheader("📊 Results")

col1, col2 = st.columns(2)

with col1:
    st.metric("Peak Infected", f"{I_peak:.2f}")

with col2:
    st.metric("Time of Peak", f"{t_peak:.1f} days")


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