import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import kstest
import matplotlib.pyplot as plt

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

# Features (X) and target (y) variables
X = np.arange(len(data)).reshape(-1, 1)  # Use indices as features
y = data.reshape(-1, 1)  # Remission times

# Standardize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Define the neural network model
class ParameterEstimatorNN(nn.Module):
    def __init__(self):
        super(ParameterEstimatorNN, self).__init__()
        self.fc1 = nn.Linear(1, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 4)  # Output layer for 4 parameters: theta, beta, c, omega
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model, define the loss function and the optimizer
model = ParameterEstimatorNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the neural network
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train.repeat(1, 4))  # Assuming the true parameters are repeated for each sample
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        model.eval()
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val.repeat(1, 4))  # Assuming the true parameters are repeated for each sample
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Extract the learned parameters (weights and biases)
theta, beta, c, omega = model(X_tensor).mean(dim=0).detach().numpy()

# Print the estimated parameters
print(f"Estimated Parameters: theta={theta}, beta={beta}, c={c}, omega={omega}")

# Define the standardized PDF for the model
def pdf_model(x, theta, beta, c, omega):
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

# Calculate the likelihood
x_vals = scaler_X.inverse_transform(X_scaled).flatten()
y_vals = scaler_y.inverse_transform(y_scaled).flatten()
likelihoods = pdf_model(y_vals, theta, beta, c, omega)

# Calculate -2Log Likelihood
log_likelihood = np.log(np.maximum(likelihoods, 1e-10))  # Avoid log(0)
neg2log_likelihood = -2 * np.sum(log_likelihood)

# Number of parameters
k = 4

# Number of observations
n = len(data)

# Compute AIC, BIC, CAIC, HQIC
AIC = 2 * k + neg2log_likelihood
BIC = np.log(n) * k + neg2log_likelihood
CAIC = (np.log(n) + 1) * k + neg2log_likelihood
HQIC = 2 * k * np.log(np.log(n)) + neg2log_likelihood

# Perform the KS test
D, p_value = kstest(y_vals, 'norm', args=(np.mean(y_vals), np.std(y_vals)))

# Print the results
print(f"-2Log Likelihood: {neg2log_likelihood:.4f}")
print(f"AIC: {AIC:.4f}")
print(f"BIC: {BIC:.4f}")
print(f"CAIC: {CAIC:.4f}")
print(f"HQIC: {HQIC:.4f}")
print(f"KS Test p-value: {p_value:.4f}")

# Save the results to a CSV file
results = {
    'Metric': ['-2Log Likelihood', 'AIC', 'BIC', 'CAIC', 'HQIC', 'KS Test p-value'],
    'Value': [neg2log_likelihood, AIC, BIC, CAIC, HQIC, p_value]
}

results_df = pd.DataFrame(results)
results_df.to_csv('model_fit_statistics.csv', index=False)

print(results_df)

# Generate predictions using the model
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_tensor)[:, 0].numpy()

# Inverse transform the predictions and the actual values
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_actual = scaler_y.inverse_transform(y_tensor.numpy())

# Plot the actual vs predicted remission times
plt.figure(figsize=(10, 6))
plt.plot(y_actual, label='Actual Remission Times', color='blue')
plt.plot(y_pred, label='Predicted Remission Times', color='red', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Remission Time')
plt.title('Actual vs Predicted Remission Times')
plt.legend()
plt.show()