Here are some examples of implementing K-Means Clustering in Python using different approaches:

### 1. Using Scikit-Learn

Scikit-Learn provides a straightforward implementation of K-Means Clustering. Here's a simple example using the Iris dataset:

```python
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.utils import shuffle

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Shuffle the data
X, y = shuffle(X, y, random_state=42)

# Create a KMeans model with 3 clusters
model = KMeans(n_clusters=3, random_state=42)

# Fit the model to the data
iris_kmeans = model.fit(X)

# Get the labels assigned to each data point
labels = iris_kmeans.labels_
print(labels)
```

This code initializes the KMeans model, fits it to the data, and retrieves the cluster labels for each data point[4].

### 2. Custom Implementation from Scratch

You can also implement K-Means Clustering from scratch. Here’s a simple version:

```python
import numpy as np

def kmeans(X, k, num_iterations=100):
    # Randomly initialize centroids
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for _ in range(num_iterations):
        # Assign clusters based on closest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Recalculate centroids
        centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    
    return labels, centroids

# Example usage
data = np.random.rand(100, 2)  # Random dataset
labels, centroids = kmeans(data, k=3)
print(labels, centroids)
```

This custom implementation initializes centroids randomly, assigns data points to the nearest centroid, and recalculates centroids iteratively[2].

### 3. K-Means with Real-World Data

In a practical scenario, you might want to cluster a dataset like housing prices. Here’s a brief example:

```python
import pandas as pd
from sklearn.cluster import KMeans

# Load a dataset (e.g., California housing dataset)
data = pd.read_csv('california_housing.csv')
X = data[['median_income', 'households']]  # Selecting features

# Create and fit the model
model = KMeans(n_clusters=5)
model.fit(X)

# Get cluster labels
labels = model.labels_
data['Cluster'] = labels
print(data.head())
```

This example demonstrates how to cluster real-world data based on selected features, allowing for analysis of housing patterns[5].

These examples illustrate different ways to implement K-Means Clustering in Python, from using established libraries like Scikit-Learn to creating a custom algorithm from scratch.

Citations:
[1] https://realpython.com/k-means-clustering-python/
[2] https://towardsdatascience.com/create-your-own-k-means-clustering-algorithm-in-python-d7d4c9077670?gi=9e090749e6b9
[3] https://www.kaggle.com/code/prashant111/k-means-clustering-with-python
[4] https://domino.ai/blog/getting-started-with-k-means-clustering-in-python
[5] https://www.datacamp.com/tutorial/k-means-clustering-python

---------
K-Means Clustering is a widely used unsupervised machine learning algorithm that partitions an unlabeled dataset into a specified number of clusters, denoted as $k$. The primary objective of K-Means is to group similar data points together while minimizing the variance within each cluster.

## How K-Means Clustering Works

The K-Means algorithm operates through the following iterative steps:

1.  **Initialization**: Randomly select $k$ initial centroids, which serve as the centers of the clusters.
2.  **Assignment Step**: Each data point is assigned to the nearest centroid based on a distance metric, typically Euclidean distance. This assignment creates $k$ clusters.
3.  **Update Step**: After all points are assigned, the centroids are recalculated by taking the mean of all points in each cluster.
4.  **Iteration**: Steps 2 and 3 are repeated until the centroids no longer change significantly or a predefined number of iterations is reached.

The algorithm aims to minimize the within-cluster variance, which is the sum of squared distances between each point and its corresponding centroid. This process continues until the centroids stabilize, indicating that the clusters are well-formed.

## Key Characteristics

-   **Unsupervised Learning**: K-Means does not require labeled data, making it suitable for exploratory data analysis.
-   **Distance-Based**: The algorithm relies on distance metrics to determine cluster membership, which means it works best with numerical data where distance calculations are meaningful.
-   **Sensitivity to Initial Conditions**: The final clusters can vary depending on the initial placement of centroids, leading to different outcomes across multiple runs. This sensitivity can sometimes result in the algorithm converging to a local minimum rather than the global optimum.
-   **Choosing $k$**: The number of clusters $k$ must be defined before running the algorithm. Techniques like the Elbow Method can help determine the optimal number of clusters by plotting the within-cluster sum of squares against different values of $k$ and identifying the point where adding more clusters yields diminishing returns.

## Applications

K-Means clustering is applied in various domains, including:

-   **Customer Segmentation**: Grouping customers based on purchasing behavior to tailor marketing strategies.
-   **Image Compression**: Reducing the number of colors in an image by clustering similar colors together.
-   **Anomaly Detection**: Identifying outliers in data by observing which points do not fit well into any cluster.

K-Means clustering is favored for its simplicity and speed, making it a popular choice for many clustering tasks in machine learning and data analysis.

