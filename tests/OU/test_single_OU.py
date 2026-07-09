'''
File to visualize effect of OU parameter 
'''


"""
================================================================================
QUICK REFERENCE: 2D ORNSTEIN-UHLENBECK SIMULATION PARAMETERS
================================================================================

1. PARAMETER TUNING & SIGNAL BEHAVIOR:
   The structural envelope and power of the signal depend heavily on the
   relationship between parameters. Stationary variance is: Var = (sigma^2) / lambda

   * lambda (Decay Rate / Damping):
     - HIGH lambda: Strongly damped, short-lived, transient "bursts". 
     - LOW lambda: Weakly damped, long-lasting, fluid, and highly sustained oscillations.
   * sigma (Noise Intensity / Volatility):
     - Scales the absolute vertical amplitude (energy) of the signal without 
       altering the duration/timescale of the bursts.
   * omega (Angular Frequency):
     - Controls the rotation speed. Frequency in Hz = omega / (2 * pi)

--------------------------------------------------------------------------------

2. THE "dt" STABILITY GOLDILOCKS ZONE:
   Because this is a rotating system solved via an explicit Euler-Maruyama scheme,
   the time-step 'dt' has a strict mathematical upper bound. If dt is too large,
   the simulation will exponentially explode (blow up to infinity / NaN).

   To ensure numerical stability and avoid aliasing, dt must satisfy:
   
                dt < dt_max = (2 * lambda) / (lambda^2 + omega^2)

   EXAMPLES:
   * For 10 Hz (omega ~ 62.8) and lambda = 1:   dt_max ≈ 0.0005s  -> Use dt = 0.0002
   * For 30 Hz (omega ~ 188.5) and lambda = 1:  dt_max ≈ 0.000056s -> Use dt = 0.00002

   No longer true for the new code that uses backward Euler
================================================================================
"""


'''
Remarks: 
- Increasing dampening lbda with other parameters fixed augments the number of spindles and reduces their duration.

'''

import numpy as np
import matplotlib.pyplot as plt
from Functions.generate_OU import get_OU_signal

# --- Simulation Parameters ---
lambda_val = 1         # Decay rate
omega = 2 * np.pi * 30   # High frequency: 30 Hz!
sigma = 1.0              # Noise intensity

T = 20                  # Total simulation time (seconds)
dt = 0.001               # Notice dt is 0.001! (Explicit Euler would explode here)
N = int(T / dt)          
t = np.linspace(0, T, N)

# --- Set up the Implicit System Matrix ---
# Drift matrix A:
# [-lambda,  omega]
# [-omega,  -lambda]
A = np.array([[-lambda_val, omega],
              [-omega, -lambda_val]])

# Identity matrix
I = np.eye(2)

# Compute the inverse operator: (I - A * dt)^-1
# This matrix handles the "look-ahead" physics ensuring perfect stability.
implicit_operator = np.linalg.inv(I - A * dt)

# --- Initialize Arrays ---
x = np.zeros(N)
y = np.zeros(N)

# --- Backward Euler Simulation Loop ---
for i in range(1, N):
    # Independent Wiener increments
    dW_x = np.random.normal(0, np.sqrt(dt))
    dW_y = np.random.normal(0, np.sqrt(dt))
    
    # Current state vector + stochastic noise push
    current_state_with_noise = np.array([
        x[i-1] + np.sqrt(2) * sigma * dW_x,
        y[i-1] + np.sqrt(2) * sigma * dW_y
    ])
    
    # Solve implicitly for the next step via matrix multiplication
    next_state = implicit_operator @ current_state_with_noise
    
    x[i] = next_state[0]
    y[i] = next_state[1]

# --- Visualization ---
plt.figure(figsize=(12, 8))

# Time domain plot
plt.subplot(2, 1, 1)
plt.plot(t, x, label='$x(t)$ (Excitatory)', color='#1f77b4', alpha=0.8)
plt.plot(t, y, label='$y(t)$ (Inhibitory)', color='#ff7f0e', alpha=0.8)
plt.title(f'Implicit Backward Euler: Stable 2D OU Process ({int(omega/(2*np.pi))} Hz) at $dt={dt}$')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Phase Space Plot
plt.subplot(2, 1, 2)
plt.plot(x, y, color='#2ca02c', lw=0.6, alpha=0.7)
plt.scatter(x[0], y[0], color='red', zorder=5, label='Start')
plt.title('Phase Space Trajectory')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.axis('equal')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()