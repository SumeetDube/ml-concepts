### Example 1: Using the Iris Dataset

The Iris dataset is a classic example used to demonstrate clustering techniques. Below is a step-by-step implementation using Agglomerative Hierarchical Clustering.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data

# Create a dendrogram
linkage_data = linkage(X, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linkage_data)
plt.title('Dendrogram for the Iris Dataset')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()

# Apply Agglomerative Hierarchical Clustering
agg_clustering = AgglomerativeClustering(n_clusters=3)
labels = agg_clustering.fit_predict(X)

# Visualize the clusters
plt.figure(figsize=(10, 7))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
plt.title('Agglomerative Clustering of Iris Dataset')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()
```

### Example 2: Custom Data Points

This example demonstrates hierarchical clustering on a custom dataset using the `scipy` library to create a dendrogram.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Sample data points
data = np.array([[4, 21], [5, 19], [10, 24], [4, 17], [3, 16],
                 [11, 25], [14, 24], [6, 22], [10, 21], [12, 21]])

# Create a dendrogram
linkage_data = linkage(data, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linkage_data)
plt.title('Dendrogram for Custom Data Points')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()

# Apply Agglomerative Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering

agg_clustering = AgglomerativeClustering(n_clusters=2)
labels = agg_clustering.fit_predict(data)

# Visualize the clusters
plt.figure(figsize=(10, 7))
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='rainbow')
plt.title('Agglomerative Clustering of Custom Data Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
```

### Example 3: Using Scikit-Learn for Marketing Data

In a marketing context, you might want to cluster customers based on their purchasing behavior. Here’s how you can implement it.

```python
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Sample marketing data
data = {
    'Annual Income (k$)': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    'Spending Score (1-100)': [39, 81, 6, 77, 40, 76, 6, 94, 3, 72]
}
df = pd.DataFrame(data)

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Apply Agglomerative Hierarchical Clustering
agg_clustering = AgglomerativeClustering(n_clusters=3)
labels = agg_clustering.fit_predict(scaled_data)

# Visualize the clusters
plt.figure(figsize=(10, 7))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=labels, cmap='rainbow')
plt.title('Agglomerative Clustering of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
```

Citations:
[1] https://blog.quantinsti.com/hierarchical-clustering-python/
[2] https://vitalflux.com/hierarchical-clustering-explained-with-python-example/
[3] https://www.datacamp.com/tutorial/introduction-hierarchical-clustering-python
[4] https://www.w3schools.com/python/python_ml_hierarchial_clustering.asp
[5] https://stackabuse.com/hierarchical-clustering-with-python-and-scikit-learn/


--------
Hierarchical clustering is a widely used unsupervised machine learning technique that organizes data points into a hierarchical structure, allowing for the exploration of relationships and similarities among the data. This method is particularly useful when the **number of clusters is not predetermined**, as it can dynamically reveal the underlying structure of the data.

## Types of Hierarchical Clustering

Hierarchical clustering can be categorized into two main types:

### 1. Agglomerative Hierarchical Clustering (AHC)

Agglomerative clustering follows a **bottom-up approach**. Initially, each data point is treated as an individual cluster. The algorithm then iteratively merges the closest pairs of clusters based on a specified distance metric until only one cluster remains. The key steps involved include:

- **Distance Measurement**: The distance between clusters can be calculated using various metrics, such as Euclidean distance, Manhattan distance, or others.

- **Linkage Criteria**: This defines how the distance between clusters is calculated. Common methods include:
  - **Single Linkage**: Distance between the closest points in two clusters.
  - **Complete Linkage**: Distance between the farthest points in two clusters.
  - **Average Linkage**: Average distance between all points in two clusters.
  - **Ward’s Method**: Minimizes the total within-cluster variance.

The results of agglomerative clustering are often visualized using a **dendrogram**, a tree-like diagram that illustrates the merging process of clusters. The height of the branches in the dendrogram indicates the distance at which clusters are merged, allowing users to decide the optimal number of clusters by "cutting" the dendrogram at a desired level[1][2][3].

### 2. Divisive Hierarchical Clustering

In contrast, divisive clustering employs a **top-down approach**. It begins with all data points in a single cluster and recursively splits the clusters into smaller ones. This method is less commonly used due to its higher computational complexity but can be beneficial when a clear hierarchical structure is needed from the outset[3][5].

## Advantages of Hierarchical Clustering

- **No Predefined Number of Clusters**: Unlike methods like K-means, hierarchical clustering does not require the user to specify the number of clusters in advance. The dendrogram allows for flexible cluster selection based on the analysis needs.

- **Hierarchical Structure**: The resulting dendrogram provides a clear visual representation of the data structure, making it easier to understand relationships among data points.

- **Versatility**: Hierarchical clustering can handle various types of data, including those with non-convex shapes and different sizes and densities[3][4].

## Disadvantages of Hierarchical Clustering

- **Computational Complexity**: The algorithm can be computationally expensive, especially for large datasets, as it requires calculating distances between all pairs of clusters.

- **Sensitivity to Noise**: The results can be affected by outliers and noisy data, which may lead to misleading cluster formations.

- **Linkage and Distance Metric Dependence**: The choice of distance metric and linkage method can significantly influence the clustering outcome, requiring careful consideration based on the specific dataset and analysis goals[3][4][5].

## Applications

Hierarchical clustering is applied in various fields, including:

- **Bioinformatics**: Classifying genes or proteins based on their expression levels.
- **Market Research**: Segmenting customers into distinct groups for targeted marketing strategies.
- **Image Processing**: Grouping similar images or features in computer vision tasks.
- **Social Network Analysis**: Identifying communities within a network based on user interactions[2][5].

In summary, hierarchical clustering is a powerful tool in machine learning for uncovering the structure of data without prior knowledge of the number of clusters. Its flexibility and visual output make it a valuable method for exploratory data analysis.

Citations:
[1] https://www.javatpoint.com/hierarchical-clustering-in-machine-learning
[2] https://www.w3schools.com/python/python_ml_hierarchial_clustering.asp
[3] https://www.geeksforgeeks.org/hierarchical-clustering-in-data-mining/
[4] https://www.simplilearn.com/tutorials/data-science-tutorial/hierarchical-clustering-in-r
[5] https://www.learndatasci.com/glossary/hierarchical-clustering/