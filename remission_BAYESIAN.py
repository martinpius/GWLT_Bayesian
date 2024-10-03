import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from scipy.stats import kstest

# Define the PDF function with improved numerical stability
def pdf_function(x, theta, beta, c, omega):
    epsilon = 1e-10  # Small value to avoid numerical issues
    term1 = (theta * x) / (theta + 1)
    term2 = np.exp(-theta * x)
    term3 = 1 + term1 * term2
    term4 = -np.log(np.maximum(term3, epsilon))  # Avoid log(0)
    u = np.exp(-np.maximum((-term4 / beta)**c, epsilon))
    pdf = (c / beta) * (theta**2 * (1 + x) / (1 + theta + theta * x)) \
          * np.maximum(((theta * x - np.log(np.maximum(1 + theta + theta * x, epsilon))) / beta)**(c-1), epsilon) \
          * np.exp(-np.maximum(((theta * x - np.log(np.maximum(1 + theta + theta * x, epsilon))) / beta)**c, epsilon)) \
          * (1 - omega + 2 * omega * u)
    return pdf

# Provided remission times dataset
data = np.array([4.50, 32.15, 19.13, 4.87, 14.24, 5.71, 7.87, 7.59, 5.49, 3.02, 2.02, 4.51, 9.22, 1.05, 3.82, 9.47, 
                 26.31, 79.05, 2.02, 2.62, 4.26, 0.90, 11.25, 21.73, 10.34, 10.66, 0.51, 12.03, 3.36, 2.64, 43.01, 
                 14.76, 0.81, 1.19, 3.36, 8.66, 1.46, 14.83, 5.62, 18.10, 17.14, 25.74, 15.96, 17.36, 1.35, 4.33, 
                 9.02, 22.69, 6.94, 2.46, 7.26, 3.48, 4.23, 3.70, 6.54, 3.64, 8.65, 3.57, 5.41, 11.64, 2.09, 2.23, 
                 6.25, 7.93, 4.34, 25.82, 12.02, 3.88, 13.80, 5.85, 7.09, 20.28, 5.32, 46.12, 5.17, 2.80, 0.20, 8.37, 
                 36.66, 14.77, 10.06, 8.53, 4.98, 11.98, 5.06, 1.76, 16.62, 4.40, 12.07, 34.26, 6.97, 2.07, 0.08, 
                 17.12, 1.40, 12.63, 2.75, 7.66, 7.32, 4.18, 1.26, 13.29, 6.76, 23.63, 3.25, 7.62, 7.63, 3.52, 2.87, 
                 9.74, 3.31, 0.40, 2.26, 5.41, 2.69, 2.54, 11.79, 2.69, 5.34, 8.26, 6.93, 0.50, 10.75, 5.32, 13.11, 
                 5.09, 7.39])

# Ensure that data does not contain invalid values
y_data = np.clip(data, 1e-10, np.inf)
x_data = np.arange(len(y_data))

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

    # Observed data likelihood using pm.Data to allow updating
    y_obs = pm.Data("y_obs", y_data)
    likelihood = pm.Normal('likelihood', mu=pdf, sigma=0.1, observed=y_obs)
    
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
summary_df.to_csv("Bayesian_estimation_results.csv")

# Plot the posterior distributions
az.plot_posterior(idata)
plt.savefig("bayesian_posteriorsREMISSIONDATA.png")
plt.show()

# Plot the predictions
with model:
    pm.set_data({"y_obs": y_data})
    posterior_predictive = pm.sample_posterior_predictive(trace)
    y_pred = posterior_predictive["likelihood"].mean(axis=0)

plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data, label='Actual Remission Times', color='blue')
plt.plot(x_data, y_pred, label='Predicted Remission Times', color='red', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Remission Time')
plt.title('Actual vs Predicted Remission Times')
plt.legend()
plt.savefig("predictions.png")
plt.show()

# Compute -2Log Likelihood, AIC, BIC, HQIC
log_likelihood_values = pm.sample_stats.log_likelihood(trace, model).sum(axis=1)
neg2log_likelihood = -2 * np.sum(log_likelihood_values.mean())

k = 4  # Number of parameters
n = len(data)  # Number of observations

AIC = 2 * k + neg2log_likelihood
BIC = np.log(n) * k + neg2log_likelihood
HQIC = 2 * k * np.log(np.log(n)) + neg2log_likelihood

# Perform the KS test
D, p_value = kstest(y_data, 'norm', args=(np.mean(y_data), np.std(y_data)))

# Print the results
print(f"-2Log Likelihood: {neg2log_likelihood:.4f}")
print(f"AIC: {AIC:.4f}")
print(f"BIC: {BIC:.4f}")
print(f"HQIC: {HQIC:.4f}")
print(f"KS Test D statistic: {D:.4f}")
print(f"KS Test p-value: {p_value:.4f}")

# Save the results to a CSV file
results = {
    'Metric': ['-2Log Likelihood', 'AIC', 'BIC', 'HQIC', 'KS Test D statistic', 'KS Test p-value'],
    'Value': [neg2log_likelihood, AIC, BIC, HQIC, D, p_value]}
df = pd.DataFrame(results)
df.to_csv("BAYESIANOUTREMISSION.csv")
