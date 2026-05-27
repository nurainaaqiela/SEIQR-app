import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# SEIQR model
def seiqr(t, y, beta, sigma, gamma, gamma_q, delta, mu, Lambda):
    S, E, I, Q, R = y

    dS = Lambda - beta*S*I - mu*S
    dE = beta*S*I - (sigma + mu)*E
    dI = sigma*E - (gamma + delta + mu)*I
    dQ = delta*I - (gamma_q + mu)*Q
    dR = gamma*I + gamma_q*Q - mu*R

    return [dS, dE, dI, dQ, dR]


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


st.title("SEIQR Disease Simulation App")

# sliders
beta = st.slider("Transmission rate (beta)", 0.0001, 0.01, 0.002)
sigma = st.slider("Incubation rate (sigma)", 0.1, 1.0, 0.5)
gamma = st.slider("Recovery rate (gamma)", 0.1, 1.0, 0.3)
gamma_q = st.slider("Quarantine recovery rate", 0.1, 1.0, 0.1)
delta = st.slider("Quarantine rate (delta)", 0.0, 0.5, 0.2)
mu = st.slider("Natural death rate (mu)", 0.0, 0.05, 0.01)
Lambda = st.slider("Birth rate (Lambda)", 0.0, 20.0, 10.0)

# run model
sol = run_model(beta, sigma, gamma, gamma_q, delta, mu, Lambda)

# plot
fig, ax = plt.subplots()

ax.plot(sol.t, sol.y[0], label="S")
ax.plot(sol.t, sol.y[1], label="E")
ax.plot(sol.t, sol.y[2], label="I")
ax.plot(sol.t, sol.y[3], label="Q")
ax.plot(sol.t, sol.y[4], label="R")

ax.legend()
st.pyplot(fig)
