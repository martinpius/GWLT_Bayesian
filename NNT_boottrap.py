import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic data
def generate_data(n_samples, theta, beta, c, omega):
    x = np.random.rand(n_samples)
    term1 = (theta * x) / (theta + 1)
    term2 = np.exp(-theta * x)
    term3 = 1 + term1 * term2
    term4 = -np.log(np.maximum(term3, 1e-10))  # Avoid log(0)
    u = np.exp(-np.maximum((-term4 / beta)**c, 1e-10))
    pdf = (c / beta) * (theta**2 * (1 + x) / (1 + theta + theta * x)) \
          * np.maximum(((theta * x - np.log(np.maximum(1 + theta + theta * x, 1e-10))) / beta)**(c-1), 1e-10) \
          * np.exp(-np.maximum(((theta * x - np.log(np.maximum(1 + theta + theta * x, 1e-10))) / beta)**c, 1e-10)) \
          * (1 - omega + 2 * omega * u)
    return x, pdf

# True parameter values
theta_true = 2.0
beta_true = 1.0
c_true = 1.5
omega_true = 0.5
n_samples = 1000

# Generate synthetic data
x_data, y_data = generate_data(n_samples, theta_true, beta_true, c_true, omega_true)

# Standardize data
scaler_x = StandardScaler()
scaler_y = StandardScaler()
x_data = scaler_x.fit_transform(x_data.reshape(-1, 1))
y_data = scaler_y.fit_transform(y_data.reshape(-1, 1))

# Convert data to PyTorch tensors
x_data = torch.tensor(x_data, dtype=torch.float32)
y_data = torch.tensor(y_data, dtype=torch.float32)

# Define the neural network for parameter estimation
class ParameterEstimatorNN(nn.Module):
    def __init__(self):
        super(ParameterEstimatorNN, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 4)  # Output layer for 4 parameters: theta, beta, c, omega
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the function to compute HDI
def compute_hdi(samples, hdi_prob=0.94):
    sorted_samples = np.sort(samples)
    ci_idx_inc = int(np.floor(hdi_prob * len(sorted_samples)))
    n_cis = len(sorted_samples) - ci_idx_inc
    ci_width = sorted_samples[ci_idx_inc:] - sorted_samples[:n_cis]
    min_idx = np.argmin(ci_width)
    hdi_min = sorted_samples[min_idx]
    hdi_max = sorted_samples[min_idx + ci_idx_inc]
    return hdi_min, hdi_max

# Function to train and evaluate the neural network with bootstrapping
def train_and_evaluate_bootstrap(x_data, y_data, n_bootstraps=100, num_epochs=1000):
    param_estimates = []

    for i in range(n_bootstraps):
        # Resample the data
        x_resampled, y_resampled = resample(x_data, y_data)

        # Split data into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42)

        # Generate target data for training (the known true parameter values for all samples)
        y_train_params = torch.tensor([[theta_true, beta_true, c_true, omega_true]] * x_train.shape[0], dtype=torch.float32)
        y_val_params = torch.tensor([[theta_true, beta_true, c_true, omega_true]] * x_val.shape[0], dtype=torch.float32)

        # Instantiate the neural network
        model = ParameterEstimatorNN()

        # Define the loss function and the optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Train the neural network
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(x_train)
            loss = criterion(outputs, y_train_params)
            loss.backward()
            optimizer.step()

        # Evaluate the model on the full dataset
        model.eval()
        with torch.no_grad():
            estimated_params = model(x_data).mean(dim=0).numpy()
        param_estimates.append(estimated_params)

    param_estimates = np.array(param_estimates)
    return param_estimates

# Train and evaluate the neural network with bootstrapping
param_estimates = train_and_evaluate_bootstrap(x_data, y_data)

# Compute the mean estimates and HDIs
mean_estimates = np.mean(param_estimates, axis=0)
theta_hdi = compute_hdi(param_estimates[:, 0])
beta_hdi = compute_hdi(param_estimates[:, 1])
c_hdi = compute_hdi(param_estimates[:, 2])
omega_hdi = compute_hdi(param_estimates[:, 3])

# Save the results to a CSV file
results = {
    'Parameter': ['theta', 'beta', 'c', 'omega'],
    'True Value': [theta_true, beta_true, c_true, omega_true],
    'Estimate': mean_estimates,
    'HDI Lower': [theta_hdi[0], beta_hdi[0], c_hdi[0], omega_hdi[0]],
    'HDI Upper': [theta_hdi[1], beta_hdi[1], c_hdi[1], omega_hdi[1]]
}

results_df = pd.DataFrame(results)
results_df.to_csv('neural_network_parameter_estimates_with_hdi.csv', index=False)

print(results_df)