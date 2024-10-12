## Dimensionality Reduction in Machine Learning: PCA and t-SNE

Dimensionality reduction is a crucial technique in machine learning that simplifies complex datasets by reducing the number of features (or dimensions) while preserving essential information. Two of the most widely used methods for dimensionality reduction are **Principal Component Analysis (PCA)** and **t-distributed Stochastic Neighbor Embedding (t-SNE)**. Each technique has its unique approach and applications, making them suitable for different scenarios.

### Principal Component Analysis (PCA)

***Overview of PCA***

Principal Component Analysis (PCA) is a linear dimensionality reduction technique that transforms the original dataset into a new coordinate system. In this new system, the axes (or principal components) are ordered by the amount of *variance* they capture from the data. The first principal component captures the most variance, the second captures the second most, and so on. This allows for the reduction of dimensions while retaining the most significant features of the data.

- **Mathematical Basis**: PCA involves the computation of the covariance matrix of the data, followed by eigendecomposition to find the eigenvalues and eigenvectors. The eigenvectors corresponding to the largest eigenvalues define the principal components, which are orthogonal to each other[1][4].

- **Applications**: PCA is commonly used in exploratory data analysis, data visualization, and preprocessing for other machine learning algorithms. It is particularly effective when dealing with high-dimensional datasets where many features are correlated, helping to reduce noise and improve model performance[3][5].

- **Limitations**: While PCA is powerful, it assumes linear relationships among features. This means it may not perform well with datasets that have complex, nonlinear structures[2].
*maximizing variance*
### t-distributed Stochastic Neighbor Embedding (t-SNE)

****Overview of t-SNE****

t-SNE is a non-linear dimensionality reduction technique primarily used for visualizing high-dimensional data in *two or three dimensions*. Unlike PCA, t-SNE focuses on preserving the local structure of the data, meaning that similar data points remain close together in the lower-dimensional space.

- **Mechanism**: t-SNE works by converting the high-dimensional Euclidean distances between points into conditional probabilities that represent similarities. It then attempts to minimize the divergence between these probabilities in the high-dimensional space and the corresponding probabilities in the lower-dimensional space[2][4].

- **Applications**: t-SNE is particularly useful for visualizing complex datasets, such as images or text, where the relationships between data points are not linear. It is widely used in fields like bioinformatics, image processing, and natural language processing to explore and identify patterns within the data[2].

- **Limitations**: t-SNE can be computationally intensive and may struggle with very large datasets. Additionally, it does not preserve global structures as effectively as PCA, which can lead to misleading interpretations if the overall data distribution is important[4].

### Comparison of PCA and t-SNE

| Feature                       | PCA                                  | t-SNE                                      |
| ----------------------------- | ------------------------------------ | ------------------------------------------ |
| **Type**                      | Linear                               | Non-linear                                 |
| **Preservation of Structure** | Global structure (variance)          | Local structure (similarity)               |
| **Computational Complexity**  | Relatively low                       | Higher, especially with large datasets     |
| **Use Cases**                 | Preprocessing, exploratory analysis  | Visualization of complex datasets          |
| **Interpretability**          | Easier to interpret due to linearity | More challenging due to non-linear mapping |


### Conclusion

Both PCA and t-SNE serve important roles in dimensionality reduction within machine learning. PCA is ideal for linear relationships and high-dimensional datasets where variance is a key focus, while t-SNE excels in visualizing complex, non-linear relationships among data points. Choosing between these techniques depends on the specific requirements of the analysis, such as the nature of the data and the desired outcomes.

Citations:
[1] https://en.wikipedia.org/wiki/Principal_component_analysis
[2] https://www.enjoyalgorithms.com/blog/tsne-algorithm-in-ml/
[3] https://www.keboola.com/blog/pca-machine-learning
[4] https://www.ibm.com/topics/dimensionality-reduction
[5] https://builtin.com/data-science/step-step-explanation-principal-component-analysis
[6] https://www.scdiscoveries.com/blog/knowledge/what-is-t-sne-plot/
[7] https://www.displayr.com/using-t-sne-to-visualize-data-before-prediction/
[8] https://en.wikipedia.org/wiki/T-**distributed_stochastic_neighbor_embedding**

----------------
## Examples of Dimensionality Reduction in Python

Dimensionality reduction techniques like PCA and t-SNE can be easily implemented in Python using libraries such as **scikit-learn** and **matplotlib**. Below are examples of how to perform both PCA and t-SNE on sample datasets.

### Example 1: Principal Component Analysis (PCA)

****Using the Iris Dataset****

The Iris dataset is a classic dataset for demonstrating PCA. Here’s how to implement PCA using scikit-learn:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions
X_reduced = pca.fit_transform(X_scaled)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, edgecolor='k', cmap='viridis')
plt.title('PCA of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
plt.show()
```

In this example, PCA reduces the four dimensions of the Iris dataset to two dimensions, allowing for visualization of the data.

### Example 2: t-distributed Stochastic Neighbor Embedding (t-SNE)

****Using the MNIST Dataset****

t-SNE is particularly effective for visualizing high-dimensional data like images. Here’s an example using the MNIST dataset:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data
y = mnist.target

# Reduce dimensions with PCA first for better performance
pca = PCA(n_components=50)  # Reduce to 50 dimensions
X_reduced = pca.fit_transform(X)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_reduced)

# Plot the results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y.astype(int), cmap='Spectral', alpha=0.5)
plt.title('t-SNE of MNIST Dataset')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(scatter)
plt.show()
```

In this example, we first reduce the dimensionality of the MNIST dataset to 50 dimensions using PCA, which helps speed up the t-SNE computation. The resulting two-dimensional representation is then plotted, showing how t-SNE clusters similar digits together.

### Conclusion

Both PCA and t-SNE are powerful tools for dimensionality reduction in Python. PCA is effective for linear dimensionality reduction and is often used as a preprocessing step, while t-SNE excels in visualizing complex, non-linear relationships in high-dimensional data. These examples demonstrate how to implement these techniques using popular Python libraries.

Citations:
[1] https://towardsdatascience.com/dimensionality-reduction-with-pca-and-t-sne-in-python-c80c680221d
[2] https://www.datacamp.com/tutorial/principal-component-analysis-in-python
[3] https://www.geeksforgeeks.org/implementing-pca-in-python-with-scikit-learn/
[4] https://towardsdatascience.com/t-sne-from-scratch-ft-numpy-172ee2a61df7
[5] https://pieriantraining.com/machine-learning-in-python-principal-component-analysis-pca/
[6] https://www.javatpoint.com/principal-component-analysis-with-python
[7] https://towardsdatascience.com/a-step-by-step-implementation-of-principal-component-analysis-5520cc6cd598?gi=53c5a08b5369
[8] https://www.geeksforgeeks.org/principal-component-analysis-with-python/
[9] https://www.geeksforgeeks.org/difference-between-pca-vs-t-sne/
[10] https://www.datacamp.com/tutorial/introduction-t-sne
[11] https://builtin.com/data-science/tsne-python
[12] https://towardsdatascience.com/understanding-t-sne-by-implementing-2baf3a987ab3?gi=f2fad751910e
[13] https://www.geeksforgeeks.org/ml-t-distributed-stochastic-neighbor-embedding-t-sne-algorithm/
[14] https://stackoverflow.com/questions/52849890/how-to-implement-t-sne-in-a-model
[15] https://www.kaggle.com/code/patrickparsa/dimensionality-reduction-pca-and-tsne
[16] https://www.kaggle.com/code/tilii7/dimensionality-reduction-pca-tsne

-----------
# Viewing an image from the minst dataset 
```python 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Fetch the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
x = mnist.data
y = mnist.target

# Function to plot an MNIST image
def plot_mnist_image(index):
    image = x[index].reshape(28, 28)  # Reshape the image to 28x28 pixels
    label = y[index]  # Get the corresponding label
    
    plt.imshow(image, cmap='gray')  # Plot the image with a gray colormap
    plt.title(f'Label: {label}')
    plt.axis('off')  # Turn off the axis
    plt.show()

# Plot an example image
plot_mnist_image(0)  # Change the index to view different images

```