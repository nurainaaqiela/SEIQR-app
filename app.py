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
        fun=lambda t, y: seiqr(t, y, beta, sigma, gamma, gamma_q, delta, mu, Lambda),
        t_span=(0, 160),
        y0=y0,
        t_eval=t_eval
    )

    return sol


# =========================
# UI
# =========================
st.title("🦠 SEIQR Disease Simulation Dashboard")

st.markdown("""
Interactive epidemiological model to simulate disease spread and evaluate quarantine effectiveness.
""")

st.sidebar.header("Model Parameters")

beta = st.sidebar.number_input(
    "Transmission rate (β)",
    min_value=0.0001,
    max_value=0.01,
    value=0.002,
    step=0.0001,
    format="%.4f"
)

sigma = st.sidebar.slider("Incubation rate (σ)", 0.1, 1.0, 0.5, 0.1)
gamma = st.sidebar.slider("Recovery rate (γ)", 0.1, 1.0, 0.3, 0.1)
gamma_q = st.sidebar.slider("Quarantine recovery rate (γq)", 0.1, 1.0, 0.1, 0.1)
delta = st.sidebar.slider("Quarantine rate (δ)", 0.0, 0.5, 0.2, 0.01)

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
# RUN SIMULATIONS
# =========================

# No quarantine scenario
sol_no_q = run_model(beta, sigma, gamma, gamma_q, 0.0, mu, Lambda)

# With quarantine scenario
sol_q = run_model(beta, sigma, gamma, gamma_q, delta, mu, Lambda)


# =========================
# METRICS
# =========================
I_peak_noq = np.max(sol_no_q.y[2])
I_peak_q = np.max(sol_q.y[2])

reduction = (I_peak_noq - I_peak_q) / I_peak_noq * 100 if I_peak_noq > 0 else 0


st.subheader("📊 Key Results Comparison")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Peak (No Quarantine)", f"{I_peak_noq:.2f}")

with col2:
    st.metric("Peak (With Quarantine)", f"{I_peak_q:.2f}")

with col3:
    st.metric("Reduction", f"{reduction:.1f}%")


# =========================
# PLOT
# =========================
fig, ax = plt.subplots()

ax.plot(sol_no_q.t, sol_no_q.y[2], "--r", label="Infected (No Quarantine)")
ax.plot(sol_q.t, sol_q.y[2], "-b", label="Infected (With Quarantine)")

ax.set_xlabel("Time (days)")
ax.set_ylabel("Infected Population")
ax.set_title("Impact of Quarantine on Disease Spread")
ax.legend()

st.pyplot(fig)