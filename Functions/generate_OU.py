import numpy as np

def get_OU_signal(T, dt, lbda, omega, sigma):
    """
    Generates a 2D Ornstein-Uhlenbeck signal using the unconditionally stable 
    Implicit Backward Euler scheme.
    """
    N = int(T / dt)          # Number of time steps
    t = np.linspace(0, T, N)
    x = np.zeros(N)
    y = np.zeros(N)

    # Initial conditions (starting from the origin)
    x[0] = 0.0
    y[0] = 0.0    

    # --- Pre-compute the Implicit Matrix Operator ---
    # Drift matrix A = [[-lbda,  omega],
    #                   [-omega, -lbda]]
    A = np.array([[-lbda,  omega],
                  [-omega, -lbda]])
    I = np.eye(2)
    
    # Calculate (I - A * dt)^-1 which defines the backward step geometry
    implicit_operator = np.linalg.inv(I - A * dt)

    # --- Backward Euler Simulation Loop ---
    for i in range(1, N):
        # Independent Wiener increments (Gaussian noise scaled by sqrt(dt))
        dW_x = np.random.normal(0, np.sqrt(dt))
        dW_y = np.random.normal(0, np.sqrt(dt))
        
        # Add current state to the random stochastic shock vector
        current_state_with_noise = np.array([
            x[i-1] + np.sqrt(2) * sigma * dW_x,
            y[i-1] + np.sqrt(2) * sigma * dW_y
        ])
        
        # Solve implicitly for the next step via a single matrix-vector dot product
        next_state = implicit_operator @ current_state_with_noise
        
        x[i] = next_state[0]
        y[i] = next_state[1]

    return t, x


def get_mixed_OU_signals(T, dt, lbda_list, omega_list, sigma_list, factor_list):
    """
    Inputs:
    - T: desired recording length (seconds)
    - dt: sampling period
    - lbda_list: list of lbda parameters
    - omega_list: list of omega parameters
    - sigma_list: list of sigma parameters
    - factor_list: scaling factor for each signal
    
    Outputs:
    - t: time (seconds)
    - mixed_OU: time-series of summed OU signals
    """
    N = int(T / dt)          # Number of time steps
    t = np.linspace(0, T, N)

    mixed_OU = np.zeros(N)

    for i in range(len(lbda_list)):
        # get_OU_signal returns (t, x), so indexing [-1] correctly grabs the x array
        x = get_OU_signal(T, dt, lbda_list[i], omega_list[i], sigma_list[i])[-1]
        mixed_OU += x * factor_list[i]

    return t, mixed_OU