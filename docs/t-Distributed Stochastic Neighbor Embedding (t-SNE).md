
### Example 1: Basic t-SNE on the Iris Dataset

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
X_embedded = tsne.fit_transform(X)

# Plotting the results
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis')
plt.title('t-SNE visualization of Iris Dataset')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(scatter, label='Species')
plt.show()
```

### Example 2: t-SNE on the MNIST Dataset

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data[:1000]  # Using a subset for faster computation
y = mnist.target[:1000]

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
X_embedded = tsne.fit_transform(X)

# Plotting the results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y.astype(int), cmap='jet', alpha=0.5)
plt.title('t-SNE visualization of MNIST Dataset')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(scatter, label='Digit')
plt.show()
```

### Example 3: t-SNE with PCA Preprocessing

For large datasets, it is often beneficial to apply PCA before t-SNE to reduce the dimensionality and improve performance.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# First apply PCA to reduce dimensions to 50
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X)

# Then apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
X_embedded = tsne.fit_transform(X_pca)

# Plotting the results
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis')
plt.title('t-SNE visualization of Iris Dataset after PCA')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(scatter, label='Species')
plt.show()
```

These examples illustrate how to use t-SNE for visualizing high-dimensional data in Python, highlighting its application in different datasets and the potential benefits of preprocessing with PCA.

Citations:
[1] https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding
[2] https://lvdmaaten.github.io/tsne/
[3] https://www.geeksforgeeks.org/t-distributed-stochastic-neighbor-embedding-t-sne-using-r/
[4] https://builtin.com/data-science/tsne-python
[5] https://towardsdatascience.com/t-distributed-stochastic-neighbor-embedding-t-sne-bb60ff109561?gi=43c94e5bc9d7


----------
t-Distributed Stochastic Neighbor Embedding (t-SNE) is a powerful non-linear dimensionality reduction technique primarily used for data visualization. Developed by Laurens van der Maaten and Geoffrey Hinton in 2008, t-SNE is particularly effective for embedding high-dimensional data into a two or three-dimensional space while preserving the local structure of the data.

## Overview of t-SNE

t-SNE operates by modeling the similarity between data points in a high-dimensional space and projecting these similarities into a lower-dimensional space. The algorithm achieves this by minimizing the divergence between two distributions: one that measures pairwise similarities of the input objects in the high-dimensional space and another that measures pairwise similarities of the corresponding low-dimensional points. This process involves:

1. **Calculating Pairwise Similarities**: t-SNE uses a Gaussian distribution to compute the similarity between points in high-dimensional space. Points that are closer together have a higher probability of being selected as neighbors.

2. **Mapping to Lower Dimensions**: The algorithm attempts to preserve these pairwise similarities when mapping the high-dimensional data to a lower-dimensional space. It focuses on maintaining local relationships, meaning that points that are close in high-dimensional space remain close in the lower-dimensional representation.

3. **Optimization**: t-SNE employs gradient descent to minimize the difference between the two distributions, effectively creating a lower-dimensional representation that captures the structure of the data[1][3][4].

## Comparison with PCA

While both t-SNE and Principal Component Analysis (PCA) are used for dimensionality reduction, they differ significantly in their approach:

- **PCA** is a linear technique that seeks to maximize variance and is best suited for data with a linear structure. It preserves large pairwise distances among points.

- **t-SNE**, on the other hand, is non-linear and focuses on preserving small pairwise distances, making it particularly useful for visualizing complex datasets where relationships are not linear. This characteristic allows t-SNE to reveal clusters and patterns that PCA might miss[2][3].

## Applications

t-SNE is widely used in various fields, including:

- **Image Processing**: Visualizing high-dimensional image data.
- **Natural Language Processing**: Analyzing word embeddings.
- **Biological Data Analysis**: Exploring genomic data.
- **Anomaly Detection**: Identifying outliers in datasets.
- **Social Network Analysis**: Understanding relationships in complex networks[4][5].

## Conclusion

t-SNE is a crucial tool in machine learning for visualizing high-dimensional data. By effectively preserving the local structure of the data, it enables researchers and data scientists to uncover hidden patterns and relationships, facilitating better insights and interpretations of complex datasets.

Citations:
[1] https://www.geeksforgeeks.org/ml-t-distributed-stochastic-neighbor-embedding-t-sne-algorithm/
[2] https://towardsdatascience.com/t-distributed-stochastic-neighbor-embedding-t-sne-bb60ff109561?gi=43c94e5bc9d7
[3] https://www.datacamp.com/tutorial/introduction-t-sne
[4] https://www.javatpoint.com/t-sne-in-machine-learning
[5] https://www.geeksforgeeks.org/t-distributed-stochastic-neighbor-embedding-t-sne-using-r/