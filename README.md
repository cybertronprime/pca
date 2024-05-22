# Principal Component Analysis (PCA) for Data Visualization

## Introduction

Principal Component Analysis (PCA) is a widely used dimensionality reduction technique that transforms high-dimensional datasets into lower-dimensional spaces while retaining the most important information. It identifies the principal components, which are linear combinations of the original features that capture the maximum variance in the data.

In this project, we have a dataset consisting of 20 treatments (T0 to T19), each represented by 5 variables: HgCl2 concentration (%), EtOH exposure time (seconds), and contamination percentages for three replicates (R1, R2, R3). The goal is to visualize and analyze the patterns and relationships in this dataset using PCA.

## Data Preprocessing

Before applying PCA, the data is directly defined as a NumPy array in the script. The treatment names are separated from the feature matrix `X` using NumPy indexing. `X` contains the numeric values of the variables, while `treatments` contains the corresponding treatment names.

## Applying PCA

PCA is applied to the feature matrix `X` using the `PCA` class from scikit-learn. The `n_components` parameter is set to 3, indicating that we want to retain 3 principal components.

```python
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)
```

The `fit_transform` method fits the PCA model to the data and transforms the data into the principal component space. The resulting `X_pca` array contains the transformed samples in the lower-dimensional space.

## Principal Component Calculation

PCA calculates the principal components by performing an eigendecomposition of the covariance matrix of the data. The covariance matrix captures the relationships between the variables and their variances.

The first principal component (PC1) is the eigenvector corresponding to the largest eigenvalue of the covariance matrix. It represents the direction in the feature space along which the data varies the most. PC1 is a linear combination of the original variables, with weights determined by the eigenvector.

The second principal component (PC2) is the eigenvector corresponding to the second-largest eigenvalue, and so on. Each subsequent principal component captures the maximum remaining variance in the data, while being orthogonal to the previous components.

In this case, PC1 captures the most significant patterns and variations in the data across the 5 variables (HgCl2 concentration, EtOH exposure time, and contamination percentages for R1, R2, R3).

## Visualization

The script creates visualizations of the data in both 3D and 2D spaces using the principal components.

### 3D Plot

The 3D plot shows the transformed samples in the space of all three principal components (PC1, PC2, PC3). Each point represents a treatment, with color indicating its position along PC1.

```python
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], cmap='plasma', c=X_pca[:,0])
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')
ax.set_title('PCA 3D Plot')
plt.show()
```

The 3D plot provides an overview of the data distribution across all three principal components. However, it may not always be the most effective visualization for identifying patterns and variations, especially if the data is primarily captured by the first two principal components.

### 2D Plot

The 2D plot focuses on the first two principal components (PC1 and PC2) and provides a clearer visualization of the data patterns and relationships.

```python
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], cmap='plasma', c=X_pca[:,0])
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('PCA 2D Plot')
plt.colorbar()

for i, txt in enumerate(treatments):
    plt.annotate(txt, (X_pca[i,0], X_pca[i,1]))
    
plt.tight_layout()
plt.show()
```

In the 2D plot, each point represents a treatment, and the axes correspond to PC1 and PC2. The position of each point in the plot is determined by its coordinates in the principal component space. Points that are close together in the plot have similar profiles across the measured variables.

The 2D plot also labels each point with its treatment name, making it easier to identify specific treatments and their relationships.

## Variance Explanation

The script prints the percentage of variance explained by each principal component.

```python
print(f"PC1 variance: {pca.explained_variance_ratio_[0]*100:.1f}%")
print(f"PC2 variance: {pca.explained_variance_ratio_[1]*100:.1f}%")
print(f"PC3 variance: {pca.explained_variance_ratio_[2]*100:.1f}%")
```

The variance explanation provides insights into how much of the total variance in the data is captured by each principal component. If a high percentage of the variance is explained by the first two principal components (e.g., >70-80%), then the 2D plot is likely to be a good representation of the data patterns and relationships.

In this case, if PC1 and PC2 capture a significant portion of the total variance, the 2D plot will be more informative and accurate in visualizing the data compared to the 3D plot.

## Interpretation

The interpretation of the PCA results depends on the specific dataset and the context of the analysis. In this case, the 2D plot of PC1 vs. PC2 can reveal patterns, clusters, or trends among the treatments based on their HgCl2 concentration, EtOH exposure time, and contamination percentages.

Some possible interpretations could include:

- Treatments that cluster together in the plot may have similar profiles across the measured variables, indicating potential similarities in their effects or responses.
- Treatments that are far apart in the plot may have distinct profiles, suggesting differences in their effects or responses.
- The direction and magnitude of the principal components can provide insights into the relative importance and relationships of the original variables.

To fully interpret the results, it is important to consider the specific context of the study, the meaning of the variables, and any prior knowledge about the treatments and their expected effects.

## Conclusion

PCA is a powerful technique for dimensionality reduction and visualization of high-dimensional data. In this project, we applied PCA to a dataset of 20 treatments characterized by 5 variables.

The 3D plot provided an overview of the data distribution across all three principal components, while the 2D plot focused on PC1 and PC2, offering a clearer visualization of the data patterns and relationships.

The variance explanation revealed the percentage of total variance captured by each principal component, helping to determine the effectiveness of the 2D plot in representing the data accurately.

The interpretation of the PCA results requires domain knowledge and should be done in the context of the specific problem at hand. By analyzing the patterns, clusters, and trends in the principal component space, insights can be gained into the similarities, differences, and potential effects of the treatments.

Overall, PCA is a valuable tool for exploratory data analysis, pattern recognition, and data visualization, enabling researchers to uncover underlying structures and relationships in complex datasets.

## Complete Code

Here's the complete code for this project:

```import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Data as a NumPy array
data = np.array([
    ['T0', 0.0, 0, 100.0, 100.0, 100.0],
    ['T1', 0.0, 30, 77.8, 83.4, 77.8],
    ['T2', 0.0, 45, 66.7, 72.3, 72.3],
    ['T3', 0.0, 60, 33.4, 50.0, 50.0],
    ['T4', 0.0, 75, 16.7, 10.5, 16.7],
    ['T5', 0.01, 0, 72.3, 61.2, 72.3],
    ['T6', 0.01, 30, 55.5, 61.2, 55.5],
    ['T7', 0.01, 45, 50.0, 44.4, 50.0],
    ['T8', 0.01, 60, 27.7, 22.7, 16.7],
    ['T9', 0.01, 75, 11.2, 11.2, 11.2],
    ['T10', 0.1, 0, 33.3, 44.4, 44.4],
    ['T11', 0.1, 30, 33.3, 22.7, 33.3],
    ['T12', 0.1, 45, 16.6, 22.7, 11.1],
    ['T13', 0.1, 60, 0.0, 0.0, 0.0],
    ['T14', 0.1, 75, 0.0, 0.0, 0.0],
    ['T15', 1.0, 0, 22.7, 22.7, 16.7],
    ['T16', 1.0, 30, 11.2, 11.2, 16.7],
    ['T17', 1.0, 45, 0.0, 0.0, 0.0],
    ['T18', 1.0, 60, 0.0, 0.0, 0.0],
    ['T19', 1.0, 75, 0.0, 0.0, 0.0]
])

# Separate features and treatment names 
X = data[:, 1:].astype(float)
treatments = data[:, 0]

# PCA
pca = PCA(n_components=3)  
X_pca = pca.fit_transform(X)

# 3D Plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], cmap='plasma', c=X_pca[:,0]) 
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')
ax.set_title('PCA 3D Plot')
plt.show()

print(f"PC1 variance: {pca.explained_variance_ratio_[0]*100:.1f}%")  
print(f"PC2 variance: {pca.explained_variance_ratio_[1]*100:.1f}%")
print(f"PC3 variance: {pca.explained_variance_ratio_[2]*100:.1f}%")

# 2D Plot 
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], cmap='plasma', c=X_pca[:,0])
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')  
plt.title('PCA 2D Plot')
plt.colorbar()

for i, txt in enumerate(treatments):
    plt.annotate(txt, (X_pca[i,0], X_pca[i,1]))
    
plt.tight_layout()
plt.show()
```

<img width="657" alt="Screenshot 2024-05-22 at 22 47 44" src="https://github.com/rajrohit10/pca/assets/42783861/af26e779-5f20-4025-ad3f-10a1d53abb99">
<img width="799" alt="Screenshot 2024-05-22 at 22 47 59" src="https://github.com/rajrohit10/pca/assets/42783861/fa76779c-0ec2-4f6e-a9bc-3a8c5a1ebaa8">


## Conclusion

By applying PCA to the given dataset and visualizing the samples in the principal component space, you can gain insights into the underlying structure of the data and identify potential patterns or relationships. This information can be used for further analysis or decision-making processes.

