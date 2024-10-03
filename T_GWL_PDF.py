import numpy as np 
import matplotlib.pyplot as plt

# Define the PDF function with handling for negative square root values
def pdf_f_safe(x, theta, beta, c, omega):
    epsilon = 1e-5
    term1 = (theta**2 * (1 + x)) / (1 + theta + theta * x + epsilon)
    term2 = (theta * x - np.log((1 + theta + theta * x) / (theta + 1) + epsilon)) / beta
    term2_power_c = np.maximum(term2**c, 0)
    exp_term = np.exp(-term2_power_c)
    log_term = -np.log(1 + (theta * x / (theta + 1) + epsilon) * np.exp(-theta * x) + epsilon)
    omega_term = 1 - omega + 2 * omega * np.exp(-np.maximum(log_term / beta, 0)**c)
    return (c / beta) * term1 * np.maximum(term2**(c - 1), 0) * exp_term * omega_term

# Generate x values
x = np.linspace(0, 10, 500)

param_sets_wide = [
    {'theta': 1, 'beta': 1, 'c': 1, 'omega': 0.5},
    {'theta': 1, 'beta': 2, 'c': 1, 'omega': -0.5},
    {'theta': 2, 'beta': 1, 'c': 2, 'omega': 0.3},
    {'theta': 0.5, 'beta': 1, 'c': 0.5, 'omega': -0.7},
    {'theta': 0.8, 'beta': 1.5, 'c': 1.5, 'omega': 0.6},
    {'theta': 1.5, 'beta': 0.8, 'c': 1.2, 'omega': -0.4},
    {'theta': 2.5, 'beta': 2.5, 'c': 1.8, 'omega': 0.2},
    {'theta': 0.3, 'beta': 0.7, 'c': 0.8, 'omega': -0.6},
    {'theta': 3, 'beta': 3, 'c': 2.5, 'omega': 0.7},
    {'theta': 0.2, 'beta': 0.5, 'c': 0.6, 'omega': -0.8},
    {'theta': 1.2, 'beta': 1.2, 'c': 1.2, 'omega': 0.4},
    {'theta': 2.2, 'beta': 1.7, 'c': 2.1, 'omega': -0.3},
]

# Plot the PDF for different parameter sets
plt.figure(figsize=(12, 8))

for i, params in enumerate(param_sets_wide):
    y = pdf_f_safe(x, **params)
    plt.plot(x, y, linewidth = 3,
             label=f"theta={params['theta']}, beta={params['beta']}, c={params['c']}, omega={params['omega']}")

plt.title("PDF of the Transmuted Generalized Weibull-Lindley (T-GWL) for Various Parameters")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend(loc='best', bbox_to_anchor=(1, 1))
plt.grid(False)
plt.savefig("PDF_T_GWL.png")
plt.show()