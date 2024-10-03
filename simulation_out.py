import pandas as pd

# True values
theta_true = 2.0
beta_true = 1.0
c_true = 1.5
omega_true = 0.5

# MLE results
mle_results = {
    'Parameter': ['theta', 'beta', 'c', 'omega'],
    'True Value': [theta_true, beta_true, c_true, omega_true],
    'MLE Estimate': [1.2570046708022777, 0.665927844119978, 0.07414069742451973, 0.22750848714320002],
    'MLE HDI Lower': [0.6396320373541342, 0.4503500147165278, 0.048570690959391254, 0.14129448237933212],
    'MLE HDI Upper': [1.358295007591361, 0.7905380297997265, 0.20022336614834393, 0.2578130768446696]
}

# Bayesian results
bayesian_results = {
    'Parameter': ['theta', 'beta', 'c', 'omega'],
    'Bayesian Estimate': [2.013, 0.796, 0.412, 0.012],
    'Bayesian SD': [0.974, 0.601, 0.298, 0.58],
    'Bayesian 3%': [0.157, 0.0, 0.002, -0.889],
    'Bayesian 97%': [3.747, 1.882, 0.968, 0.994]
}

# Neural Network results
nn_results = {
    'Parameter': ['theta', 'beta', 'c', 'omega'],
    'NN Estimate': [2.0000036, 0.99998826, 1.5000012, 0.5000119],
    'NN HDI Lower': [1.9997253, 0.9998593, 1.499851, 0.49991074],
    'NN HDI Upper': [2.000136, 1.0001781, 1.5001309, 0.5001273]
}

# Combine all results into a single DataFrame
combined_results = {
    'Parameter': mle_results['Parameter'],
    'True Value': mle_results['True Value'],
    'MLE Estimate': mle_results['MLE Estimate'],
    'MLE HDI Lower': mle_results['MLE HDI Lower'],
    'MLE HDI Upper': mle_results['MLE HDI Upper'],
    'Bayesian Estimate': bayesian_results['Bayesian Estimate'],
    'Bayesian SD': bayesian_results['Bayesian SD'],
    'Bayesian 3%': bayesian_results['Bayesian 3%'],
    'Bayesian 97%': bayesian_results['Bayesian 97%'],
    'NN Estimate': nn_results['NN Estimate'],
    'NN HDI Lower': nn_results['NN HDI Lower'],
    'NN HDI Upper': nn_results['NN HDI Upper']
}

df_combined_results = pd.DataFrame(combined_results)

# Save the combined results to a CSV file
df_combined_results.to_csv('parameter_estimation_comparison.csv', index=False)

print(df_combined_results)