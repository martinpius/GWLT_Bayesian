import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Compute statistical summary
mean = np.mean(data)
median = np.median(data)
std_dev = np.std(data)
min_val = np.min(data)
max_val = np.max(data)
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)

summary = {
    'Mean': mean,
    'Median': median,
    'Standard Deviation': std_dev,
    'Minimum': min_val,
    'Maximum': max_val,
    '25th Percentile': q1,
    '75th Percentile': q3
}

summary_df = pd.DataFrame(summary, index=['Value'])
print(summary_df)

# Plot Histogram
plt.figure(figsize=(10, 6))
sns.histplot(data, bins=20, kde=True, color='blue')
plt.title('Histogram of Remission Times')
plt.xlabel('Remission Time')
plt.ylabel('Frequency')
plt.savefig('histogramRemission.png')
plt.show()

# Plot Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(data, color='green')
plt.title('Box Plot of Remission Times')
plt.xlabel('Remission Time')
plt.savefig('box_plotRemission.png')
plt.show()

# Plot Time Series Plot
plt.figure(figsize=(10, 6))
plt.plot(data, marker='o', linestyle='-', color='red')
plt.title('Time Series of Remission Times')
plt.xlabel('Sample Index')
plt.ylabel('Remission Time')
plt.savefig('time_series_Remission.png')
plt.show()