import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data from the xlsx file
data = pd.read_excel('data_pro.xlsx')

# Calculate the Spearman correlation matrix
correlation_matrix = data.corr(method='spearman')

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Spearman Correlation Heatmap')