import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Relief time data for 20 patients
data = np.array([1.1, 1.4, 1.3, 1.7, 1.9, 1.8, 1.6, 2.2, 1.7, 2.7, 4.1, 1.8, 1.5, 1.2, 1.4, 3.0, 1.7, 2.3, 1.6, 2.0])

# Create a DataFrame for summary statistics
df = pd.DataFrame(data, columns=['Relief Time'])

# Display summary statistics
print(df.describe())

# Visualize the data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(data, kde=True)
plt.title('Distribution of Relief Time of Patients Receiving an Analgesic')

plt.subplot(1, 2, 2)
sns.boxplot(y=data)
plt.title('Boxplot of Relief Time of Patients Receiving an Analgesic')

plt.tight_layout()
plt.savefig("relief_GrossClark.png")
plt.show()