import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Step 1: Load and Explore the Data
# Load the dataset (update the file path to where 'data.csv' is located)
data = pd.read_csv('Task_brainbeamy/DataScience_Task_3_Shabarish_B_L/Code/task3/breast_cancer/breast-cancer.csv')

plt.ioff()

# Display the first few rows of the dataset
print("Dataset Overview:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Basic statistics for understanding data distribution
print("\nBasic Statistics:")
print(data.describe())

# Step 2: Labeled Scatter Plot (mean radius vs. mean texture)
plt.figure(figsize=(10, 6))

# Color data points based on diagnosis (Malignant=1, Benign=0)
colors = data['diagnosis'].map({'M': 'red', 'B': 'blue'})
labels = data['diagnosis'].map({'M': 'Malignant', 'B': 'Benign'})

# Create the scatter plot
plt.scatter(data['radius_mean'], data['texture_mean'], c=colors, alpha=0.6)

# Highlight outliers based on a condition
for i, txt in enumerate(data['diagnosis']):
    if data['radius_mean'][i] > 20:  # Highlight data points with high mean radius
        plt.text(data['radius_mean'][i], data['texture_mean'][i], txt, fontsize=8, color='black')

plt.xlabel('Mean Radius')
plt.ylabel('Mean Texture')
plt.title('Mean Radius vs. Mean Texture (Labeled by Diagnosis)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(['Malignant (Red)', 'Benign (Blue)'])
plt.show()

# Step 3: Statistical Overlay - Regression Line (mean radius vs. mean texture)
x = data['radius_mean']
y = data['texture_mean']

# Calculate regression line
slope, intercept, r_value, _, _ = stats.linregress(x, y)
line = slope * x + intercept

# Plot with regression line
plt.figure(figsize=(10, 6))
plt.scatter(x, y, c=colors, alpha=0.6, label='Data Points')
plt.plot(x, line, color='green', label=f'Regression line (R²={r_value**2:.2f})')
plt.xlabel('Mean Radius')
plt.ylabel('Mean Texture')
plt.title('Mean Radius vs. Mean Texture with Regression Line')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Step 4: Box Plot to Compare Distributions (Malignant vs. Benign)
plt.figure(figsize=(10, 6))
data.boxplot(column='area_mean', by='diagnosis')
plt.title('Box Plot of Mean Area by Diagnosis')
plt.suptitle('')  # Remove the default title for better clarity
plt.xlabel('Diagnosis')
plt.ylabel('Mean Area')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Step 5: Subplots to Compare Multiple Features
fig, ax = plt.subplots(2, 2, figsize=(14, 12))

# Scatter plot: Mean Radius vs. Mean Perimeter
data.plot(kind='scatter', x='radius_mean', y='perimeter_mean', c=colors, alpha=0.5, ax=ax[0, 0])
ax[0, 0].set_title('Mean Radius vs. Mean Perimeter')

# Histogram: Mean Area for both diagnosis
data[data['diagnosis'] == 'M']['area_mean'].plot(kind='hist', bins=20, ax=ax[0, 1], alpha=0.5, color='red', label='Malignant')
data[data['diagnosis'] == 'B']['area_mean'].plot(kind='hist', bins=20, ax=ax[0, 1], alpha=0.5, color='blue', label='Benign')
ax[0, 1].set_title('Histogram of Mean Area by Diagnosis')
ax[0, 1].legend()

# Grouped bar chart: Mean Texture by Diagnosis
grouped_data = data.groupby('diagnosis').mean()
grouped_data.plot(kind='bar', y='texture_mean', ax=ax[1, 0], color=['blue', 'red'])
ax[1, 0].set_title('Average Mean Texture by Diagnosis')

# Line chart: Changes in Mean Smoothness over Time
data.plot(kind='line', x='id', y='smoothness_mean', ax=ax[1, 1], marker='o', color='purple', alpha=0.7)
ax[1, 1].set_title('Mean Smoothness Over Observations')

plt.tight_layout()
plt.show()
