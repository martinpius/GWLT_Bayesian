import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

# Generate synthetic data
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

# Known parameters
theta_true = 2.0
beta_true = 1.0
c_true = 1.5
omega_true = 0.5

# Generate data
n_samples = 1000
x_data, y_data = generate_data(n_samples, theta_true, beta_true, c_true, omega_true)

# Ensure that data does not contain invalid values
y_data = np.clip(y_data, 1e-10, np.inf)

# Bayesian estimation with PyMC3
with pm.Model() as model:
    # Priors for the parameters
    theta = pm.Normal('theta', mu=2, sigma=1)
    beta = pm.HalfNormal('beta', sigma=1)
    c = pm.HalfNormal('c', sigma=0.5)
    omega = pm.Uniform('omega', lower=-1, upper=1)

    # Intermediate calculations with robust handling of potential numerical issues
    term1 = (theta * x_data) / (theta + 1)
    term2 = pm.math.exp(-theta * x_data)
    term3 = 1 + term1 * term2
    term4 = -pm.math.log(term3)
    u = pm.math.exp(-(-term4 / beta)**c)
    
    theta_x_log = theta * x_data - pm.math.log(1 + theta + theta * x_data)
    beta_theta_x_log = (theta_x_log / beta)
    
    pdf = (c / beta) * (theta**2 * (1 + x_data) / (1 + theta + theta * x_data)) \
          * pm.math.switch(pm.math.ge(beta_theta_x_log, 0), (beta_theta_x_log)**(c-1), 0) \
          * pm.math.switch(pm.math.ge(-(beta_theta_x_log)**c, 0), pm.math.exp(-(beta_theta_x_log)**c), 0) \
          * (1 - omega + 2 * omega * u)

    # Observed data likelihood
    y_obs = pm.Normal('y_obs', mu=pdf, sigma=0.1, observed=y_data)
    
    # Inference
    trace = pm.sample(2000, tune=1000, cores=1, return_inferencedata=False)

# Convert trace to InferenceData for ArviZ compatibility
idata = az.from_pymc3(trace)

# Summary of the posterior distribution
summary = az.summary(idata)

# Convert the summary to a DataFrame for better visualization
summary_df = summary[['mean', 'sd', 'hdi_3%', 'hdi_97%']]

# Display the results
summary_df.rename(columns={
    'mean': 'Estimated Value',
    'sd': 'SD',
    'hdi_3%': '3%',
    'hdi_97%': '97%'
}, inplace=True)
summary_df.index.name = 'Parameter'
print(summary_df)
#summary_df.to_csv("Bayessian.csv")

# Plot the posterior distributions
az.plot_posterior(idata)
#plt.savefig("bayesian.png")
plt.show()