import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

# Instantiate the neural network
model = ParameterEstimatorNN()

# Define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Generate target data for training (the known true parameter values for all samples)
y_train_params = torch.tensor([[theta_true, beta_true, c_true, omega_true]] * x_train.shape[0], dtype=torch.float32)
y_val_params = torch.tensor([[theta_true, beta_true, c_true, omega_true]] * x_val.shape[0], dtype=torch.float32)

# Train the neural network
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train_params)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        model.eval()
        val_outputs = model(x_val)
        val_loss = criterion(val_outputs, y_val_params)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Get the estimated parameters
model.eval()
with torch.no_grad():
    estimated_params = model(torch.tensor(x_data, dtype=torch.float32)).mean(dim=0).numpy()

# Print the estimated parameters
print(f"Estimated Parameters: theta={estimated_params[0]}, beta={estimated_params[1]}, c={estimated_params[2]}, omega={estimated_params[3]}")

# Save the results to a CSV file
results = {
    'Parameter': ['theta', 'beta', 'c', 'omega'],
    'True Value': [theta_true, beta_true, c_true, omega_true],
    'Estimate': estimated_params
}

results_df = pd.DataFrame(results)
results_df.to_csv('neural_network_parameter_estimates.csv', index=False)

print(results_df)