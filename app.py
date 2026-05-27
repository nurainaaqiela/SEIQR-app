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
# SIMULATION FUNCTION
# =========================
def run_model(beta, sigma, gamma, gamma_q, delta, mu, Lambda):

    y0 = [1000, 10, 5, 0, 0]
    t_eval = np.linspace(0, 160, 1000)

    sol = solve_ivp(
        fun=lambda t, y: seiqr(t, y, beta, sigma, gamma, gamma_q, delta, mu, Lambda),
        t_span=(0, 160),
        y0=y0,
        t_eval=t_eval
    )

    return sol


# =========================
# UI
# =========================
st.title("🦠 SEIQR Disease Simulation App")

st.markdown("""
This simulator models disease spread using the SEIQR framework:
Susceptible → Exposed → Infectious → Quarantined → Recovered
""")


# =========================
# INPUT PARAMETERS
# =========================

st.sidebar.header("Model Parameters")

beta = st.sidebar.number_input(
    "Transmission rate (β)",
    min_value=0.0001,
    max_value=0.01,
    value=0.002,
    step=0.0001,
    format="%.4f"
)

sigma = st.sidebar.slider(
    "Incubation rate (σ)",
    0.1, 1.0, 0.5, 0.1
)

gamma = st.sidebar.slider(
    "Recovery rate (γ)",
    0.1, 1.0, 0.3, 0.1
)

gamma_q = st.sidebar.slider(
    "Quarantine recovery rate (γq)",
    0.1, 1.0, 0.1, 0.1
)

delta = st.sidebar.slider(
    "Quarantine rate (δ)",
    0.0, 0.5, 0.2, 0.01
)

mu = st.sidebar.number_input(
    "Natural death rate (μ)",
    min_value=0.0,
    max_value=0.05,
    value=0.01,
    step=0.001,
    format="%.3f"
)

Lambda = st.sidebar.number_input(
    "Birth/entry rate (Λ)",
    min_value=0.0,
    max_value=20.0,
    value=10.0,
    step=0.5
)


# =========================
# RUN MODEL
# =========================
sol = run_model(beta, sigma, gamma, gamma_q, delta, mu, Lambda)


# =========================
# RESULTS METRICS
# =========================
I_peak = np.max(sol.y[2])
t_peak = sol.t[np.argmax(sol.y[2])]


st.subheader("📊 Key Results")

col1, col2 = st.columns(2)

with col1:
    st.metric("Peak Infected", f"{I_peak:.2f}")

with col2:
    st.metric("Time of Peak", f"{t_peak:.1f} days")


# =========================
# PLOT
# =========================
fig, ax = plt.subplots()

ax.plot(sol.t, sol.y[0], label="Susceptible (S)")
ax.plot(sol.t, sol.y[1], label="Exposed (E)")
ax.plot(sol.t, sol.y[2], label="Infected (I)")
ax.plot(sol.t, sol.y[3], label="Quarantined (Q)")
ax.plot(sol.t, sol.y[4], label="Recovered (R)")

ax.set_xlabel("Time (days)")
ax.set_ylabel("Population")
ax.set_title("SEIQR Disease Dynamics")
ax.legend()

st.pyplot(fig)