import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
np.random.seed(1290)
torch.manual_seed(1290)

# Relief time data for 20 patients
data = np.array([1.1, 1.4, 1.3, 1.7, 1.9, 1.8, 1.6, 2.2, 1.7, 2.7, 4.1, 1.8, 1.5, 1.2, 1.4, 3.0, 1.7, 2.3, 1.6, 2.0])

# Remove outliers using IQR method
df = pd.DataFrame(data, columns=['Relief Time'])
Q1 = df['Relief Time'].quantile(0.25)
Q3 = df['Relief Time'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_clean = df[(df['Relief Time'] >= lower_bound) & (df['Relief Time'] <= upper_bound)]
clean_data = df_clean['Relief Time'].values

print(f"Original data size: {len(data)}")
print(f"Cleaned data size: {len(clean_data)}")


# Define the neural network model with dropout regularization
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)  # Single output for prediction
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Train the model with cross-validation
def train_nn_cv(data, k_folds=5, epochs=1000, lr=0.01):
    data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)
    kf = KFold(n_splits=k_folds)
    all_losses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        train_data = data[train_idx]
        val_data = data[val_idx]

        model = SimpleNN()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            predictions = model(train_data)
            loss = criterion(predictions, train_data)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_predictions = model(val_data)
                val_loss = criterion(val_predictions, val_data).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

            if epoch % 100 == 0:
                print(f"Fold {fold}, Epoch {epoch}, Training Loss: {loss.item()}, Validation Loss: {val_loss}")

        all_losses.append(best_val_loss)

    print(f"Average cross-validation loss: {np.mean(all_losses)}")
    return model

# Example usage
bnn_model = train_nn_cv(clean_data)

# Obtain predictions
predictions = bnn_model(torch.tensor(clean_data, dtype=torch.float32).unsqueeze(1)).detach().numpy()

# Compute model metrics
def compute_metrics(data, predictions):
    n = len(data)
    k = 4  # Number of parameters estimated by the model
    mse = np.mean((data - predictions.flatten())**2)
    log_likelihood = -mse * n
    neg_2_log_l = -2 * log_likelihood

    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood
    caic = bic + k
    hqic = 2 * k * np.log(np.log(n)) - 2 * log_likelihood

    return neg_2_log_l, aic, bic, caic, hqic

neg_2_log_l, aic, bic, caic, hqic = compute_metrics(clean_data, predictions)

print(f"-2LogL: {neg_2_log_l}")
print(f"AIC: {aic}")
print(f"BIC: {bic}")
print(f"CAIC: {caic}")
print(f"HQIC: {hqic}")

# Plot the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(range(len(clean_data)), clean_data, label='Actual Remission Times',
          color='magenta', linewidth = 1.5)
plt.plot(range(len(clean_data)), predictions, 
         label='Predicted Remission Times', 
         color='grey', linestyle='dashed',linewidth = 1.5)
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Remission Time')
plt.title('Actual vs Predicted Remission Times of Patients Receiving an Analgesic')
plt.savefig("Analgesic_bnn_preds.png")
plt.show()