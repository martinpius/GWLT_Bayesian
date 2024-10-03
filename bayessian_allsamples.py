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
        self.fc4 = nn.Linear(32, 1)  # Output layer for the predicted remission times
    
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

# Predict the remission times
model.eval()
with torch.no_grad():
    predicted_remission_times_normalized = model(x_data_tensor).numpy()

# Denormalize the predicted remission times
predicted_remission_times = predicted_remission_times_normalized * y_std + y_mean
print(f"{[p for p in model.parameters()]}")
# Plot the predicted vs actual remission times
plt.figure(figsize=(10, 6))
plt.plot(x_data[:40], y_data[:40],
          label='Actual Remission Times', 
          color='fuchsia', linewidth = 8)
plt.plot(x_data[:40], predicted_remission_times[:40],
          label='Predicted Remission Times',
            color='gray', linestyle='--', linewidth = 8)
plt.xlabel('Sample Index')
plt.ylabel('Remission Time')
plt.title('Actual vs Predicted Remission Times')
plt.legend()
#plt.savefig("predictionsREMISSION.png")
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

# Save the results to a CSV file
results = {
    'Metric': ['-2Log Likelihood', 'AIC', 'BIC', 'HQIC', 'KS Test D statistic', 'KS Test p-value'],
    'Value': [neg2log_likelihood, AIC, BIC, HQIC, D, p_value]
}

results_df = pd.DataFrame(results)
#results_df.to_csv("Neural_Network_estimation_resultsREMISSION.csv", index=False)

print("Results saved to 'Neural_Network_estimation_results.csv'")