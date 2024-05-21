# Principal Component Analysis (PCA) for Data Visualization

## Introduction

Principal Component Analysis (PCA) is a widely used dimensionality reduction technique that transforms high-dimensional datasets into lower-dimensional spaces while retaining the most important information. It identifies the principal components, which are linear combinations of the original features that capture the maximum variance in the data.

In this project, we have a dataset (`data_samples`) consisting of 40 samples, each represented by an array of 5 values. The goal is to visualize and analyze the patterns and relationships in this dataset using PCA.

## Prerequisites

To run this code, you need the following libraries:
- NumPy
- scikit-learn
- Matplotlib

You can install them using pip:

```
pip install numpy scikit-learn matplotlib
```

## Data Preprocessing

Before applying PCA, it is common to preprocess the data using standardization. Standardization scales the features to have zero mean and unit variance, ensuring that all features contribute equally to the analysis. This is achieved using the `StandardScaler` from scikit-learn:

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## Applying PCA

After standardization, PCA is applied to the scaled data using the `PCA` class from scikit-learn. The `n_components` parameter specifies the desired number of principal components to retain. In this case, we choose to keep 3 components:

```python
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
```

The `fit_transform` method fits the PCA model to the scaled data and transforms the data into the principal component space. The resulting `X_pca` array contains the transformed samples in the lower-dimensional space.

## Visualization

To visualize the samples in the principal component space, we create scatter plots using the first two principal components (PC1 and PC2) as the axes. PC1 represents the direction of maximum variance in the dataset, capturing the most significant patterns or structures. PC2 represents the direction of the second-highest variance, orthogonal to PC1, capturing the next most significant patterns.

In the code, we create a single figure with different markers and colors for each sample:

```python
plt.figure(figsize=(10, 6))
markers = ['o', 's', 'v', '^', 'D', 'p', '*', 'h', 'H', 'd']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:blue', 'tab:orange', 'tab:purple']

for i in range(len(data_samples)):
    marker = markers[i % len(markers)]
    color = colors[i % len(colors)]
    label = f'Sample {i+1}'
    plt.scatter(X_pca[i, 0], X_pca[i, 1], marker=marker, color=color, label=label)
```

The positions of the samples in the scatter plot are determined by their coordinates in the principal component space. Samples with similar coordinates along PC1 and PC2 are considered to be similar in terms of the dominant patterns captured by these components.

## Interpretation

The interpretation of the principal components depends on the specific dataset and the context of the analysis. In this case, PC1 and PC2 capture the most important patterns and variations in the data. By visualizing the samples in this lower-dimensional space, you can identify clusters, trends, or relationships that may not be apparent in the original high-dimensional space.

PCA is a powerful tool for exploratory data analysis and dimensionality reduction. It helps in understanding the underlying structure of the data, identifying patterns, and visualizing the relationships between samples. However, it is important to note that the interpretation of the principal components requires domain knowledge and should be done in the context of the specific problem at hand.

## Complete Code

Here's the complete code for this project:

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random

# Generate random data samples
data_samples = [
    # Sample 1
    [0, 0, 100, 0, 0,],
    # Sample 2
    [0, 30, 92, 88, 12,],
    # Sample 3
    [0, 45, 80, 75, 25,],
    # Sample 4
    [60, 60, 68, 62, 35],
    # Sample 5
    [50, 55, 58, 52, 45],
    # Sample 6
    [40, 45, 48, 42, 55],
    # Sample 7
    [30, 35, 38, 32, 65],
    # Sample 8
    [20, 25, 28, 22, 75],
    # Sample 9
    [10, 15, 18, 12, 85],
    # Sample 10
    [0, 5, 8, 2, 95,],
    # Sample 11
    [90, 92, 95, 85, 5],
    # Sample 12
    [80, 82, 85, 75, 15],
    # Sample 13
    [70, 72, 75, 65, 25],
    # Sample 14
    [60, 62, 65, 55, 35],
    # Sample 15
    [50, 52, 55, 45, 45],
    # Sample 16
    [40, 42, 45, 35, 55],
    # Sample 17
    [30, 32, 35, 25, 65],
    # Sample 18
    [20, 22, 25, 15, 75],
    # Sample 19
    [10, 12, 15, 5, 85],
    # Sample 20
    [0, 2, 5, 0, 95],
    # Sample 21
    [95, 98, 100, 90, 0],
    # Sample 22
    [85, 88, 90, 80, 10],
    # Sample 23
    [75, 78, 80, 70, 20],
    # Sample 24
    [65, 68, 70, 60, 30],
    # Sample 25
    [55, 58, 60, 50, 40],
    # Sample 26
    [45, 48, 50, 40, 50],
    # Sample 27
    [35, 38, 40, 30, 60],
    # Sample 28
    [25, 28, 30, 20, 70],
    # Sample 29
    [15, 18, 20, 10, 80],
    # Sample 30
    [5, 8, 10, 0, 90],
    # Sample 31
    [90, 95, 100, 85, 0],
    # Sample 32
    [80, 85, 90, 75, 10],
    # Sample 33
    [70, 75, 80, 65, 20],
    # Sample 34
    [60, 65, 70, 55, 30],
    # Sample 35
    [50, 55, 60, 45, 40],
    # Sample 36
    [40, 45, 50, 35, 50],
    # Sample 37
    [30, 35, 40, 25, 60],
    # Sample 38
    [20, 25, 30, 15, 70],
    # Sample 39
    [10, 15, 20, 5, 80],
    # Sample 40
    [0, 5, 10, 0, 90]
]

# Stack all samples into a single 2D array
X = np.array(data_samples)

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Plot all samples in one figure with different markers and colors
plt.figure(figsize=(10, 6))
markers = ['o', 's', 'v', '^', 'D', 'p', '*', 'h', 'H', 'd']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:blue', 'tab:orange', 'tab:purple']

for i in range(len(data_samples)):
    marker = markers[i % len(markers)]
    color = colors[i % len(colors)]
    label = f'Sample {i+1}'
    plt.scatter(X_pca[i, 0], X_pca[i, 1], marker=marker, color=color, label=label)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.title('Principal Component Analysis (PCA) Visualization')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
<img width="1023" alt="Screenshot 2024-05-21 at 7 48 39 AM" src="https://github.com/rajrohit10/pca/assets/42783861/657d298f-9f54-4b26-94e6-e0896f767925">


## Conclusion

By applying PCA to the given dataset and visualizing the samples in the principal component space, you can gain insights into the underlying structure of the data and identify potential patterns or relationships. This information can be used for further analysis or decision-making processes.

