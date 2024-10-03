import numpy as np
from scipy.optimize import minimize
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Define the PDF function
def pdf_function(x, theta, beta, c, omega):
    term1 = (theta * x) / (theta + 1)
    term2 = np.exp(-theta * x)
    term3 = 1 + term1 * term2
    term4 = -np.log(term3)
    u = np.exp(-(-term4 / beta)**c)
    
    theta_x_log = theta * x - np.log(1 + theta + theta * x)
    beta_theta_x_log = theta_x_log / beta
    
    pdf = (c / beta) * (theta**2 * (1 + x) / (1 + theta + theta * x)) \
          * np.clip((beta_theta_x_log)**(c-1), 0, np.inf) \
          * np.clip(np.exp(-(beta_theta_x_log)**c), 0, np.inf) \
          * (1 - omega + 2 * omega * u)
    
    return pdf

# Define the log-likelihood function
def log_likelihood(params, x, y):
    theta, beta, c, omega = params
    pdf_values = pdf_function(x, theta, beta, c, omega)
    
    # To avoid log of zero, we use a small epsilon value
    epsilon = 1e-10
    pdf_values = np.clip(pdf_values, epsilon, None)
    
    log_likelihood_value = np.sum(np.log(pdf_values))
    return -log_likelihood_value  # Negative because we minimize in scipy

# Perform MLE
def perform_mle(x_data, y_data, initial_params=[1.0, 1.0, 1.0, 0.0]):
    result = minimize(log_likelihood, initial_params, args=(x_data, y_data), method='L-BFGS-B', 
                      bounds=[(0, None), (0, None), (0, None), (-1, 1)])
    return result.x

# Generate synthetic data
theta_true = 2.0
beta_true = 1.0
c_true = 1.5
omega_true = 0.5
n_samples = 1000

def generate_data(n_samples, theta, beta, c, omega):
    x = np.random.rand(n_samples)
    term1 = (theta * x) / (theta + 1)
    term2 = np.exp(-theta * x)
    term3 = 1 + term1 * term2
    term4 = -np.log(term3)
    u = np.exp(-(-term4 / beta)**c)
    pdf = (c / beta) * (theta**2 * (1 + x) / (1 + theta + theta * x)) \
          * ((theta * x - np.log((1 + theta + theta * x) / (theta + 1))) / beta)**(c-1) \
          * np.exp(-((theta * x - np.log((1 + theta + theta * x) / (theta + 1))) / beta)**c) \
          * (1 - omega + 2 * omega * u)
    return x, pdf

x_data, y_data = generate_data(n_samples, theta_true, beta_true, c_true, omega_true)

# Perform MLE on the original data
initial_params = [1.0, 1.0, 1.0, 0.0]
theta_mle, beta_mle, c_mle, omega_mle = perform_mle(x_data, y_data, initial_params)

# Bootstrap Resampling for HDI
n_bootstrap = 1000
bootstrap_estimates = np.zeros((n_bootstrap, 4))

np.random.seed(42)
for i in range(n_bootstrap):
    indices = np.random.choice(len(x_data), size=len(x_data), replace=True)
    x_sample = x_data[indices]
    y_sample = y_data[indices]
    bootstrap_estimates[i, :] = perform_mle(x_sample, y_sample, initial_params)

# Calculate HDIs
def compute_hdi(data, cred_mass=0.94):
    sorted_data = np.sort(data)
    ci_idx_inc = int(np.floor(cred_mass * len(sorted_data)))
    n_cis = len(sorted_data) - ci_idx_inc
    ci_width = sorted_data[ci_idx_inc:] - sorted_data[:n_cis]
    min_idx = np.argmin(ci_width)
    hdi_min = sorted_data[min_idx]
    hdi_max = sorted_data[min_idx + ci_idx_inc]
    return hdi_min, hdi_max

theta_hdi = compute_hdi(bootstrap_estimates[:, 0])
beta_hdi = compute_hdi(bootstrap_estimates[:, 1])
c_hdi = compute_hdi(bootstrap_estimates[:, 2])
omega_hdi = compute_hdi(bootstrap_estimates[:, 3])

# Save the results to a CSV file
results = {
    'Parameter': ['theta', 'beta', 'c', 'omega'],
    'True Value': [theta_true, beta_true, c_true, omega_true],
    'MLE': [theta_mle, beta_mle, c_mle, omega_mle],
    'HDI Lower': [theta_hdi[0], beta_hdi[0], c_hdi[0], omega_hdi[0]],
    'HDI Upper': [theta_hdi[1], beta_hdi[1], c_hdi[1], omega_hdi[1]]
}

results_df = pd.DataFrame(results)
results_df.to_csv('mle_hdi_results.csv', index=False)

print(results_df)