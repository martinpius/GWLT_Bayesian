import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kstest

# Set the seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

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
x_data = np.arange(len(y_data)).astype(np.float32)

# Normalize data
y_mean, y_std = np.mean(y_data), np.std(y_data)
y_data_normalized = (y_data - y_mean) / y_std

# Convert data to PyTorch tensors
x_data_tensor = torch.tensor(x_data, dtype=torch.float32).unsqueeze(1)
y_data_tensor = torch.tensor(y_data_normalized, dtype=torch.float32).unsqueeze(1)

# Define the neural network model
class ParameterEstimationNN(nn.Module):
    def __init__(self):
        super(ParameterEstimationNN, self).__init__()
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 4)  # Output layer for the parameters theta, beta, c, omega
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Instantiate the model, define the loss function and the optimizer
model = ParameterEstimationNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
epochs = 90000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_data_tensor)
    loss = criterion(outputs, y_data_tensor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10000 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Predict the parameters
model.eval()
with torch.no_grad():
    predicted_params = model(x_data_tensor).numpy()

# Extract the predicted parameters
theta_pred, beta_pred, c_pred, omega_pred = predicted_params.T

# Print the estimated parameters
print(f"Estimated theta: {np.mean(theta_pred)}")
print(f"Estimated beta: {np.mean(beta_pred)}")
print(f"Estimated c: {np.mean(c_pred)}")
print(f"Estimated omega: {np.mean(omega_pred)}")

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

# Compute the predicted remission times using the estimated parameters
predicted_remission_times = pdf_function(x_data, np.mean(theta_pred), np.mean(beta_pred), np.mean(c_pred), np.mean(omega_pred))

# Denormalize the predicted remission times
predicted_remission_times = predicted_remission_times * y_std + y_mean

# Plot the predicted vs actual remission times
plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data, label='Actual Remission Times', color='blue')
plt.plot(x_data, predicted_remission_times, label='Predicted Remission Times', color='red', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Remission Time')
plt.title('Actual vs Predicted Remission Times')
plt.legend()
#plt.savefig("predictions.png")
plt.show()

# Compute -2Log Likelihood, AIC, BIC, HQIC
log_likelihood_values = np.maximum(predicted_remission_times, 1e-10)  # Ensure no log(0)
neg2log_likelihood = -2 * np.sum(np.log(log_likelihood_values))

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