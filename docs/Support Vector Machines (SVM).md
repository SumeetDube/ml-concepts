---
~
---
### 1. SVM for Classification

In this example, we will use the SVM classifier to classify a synthetic dataset created with the `make_blobs` function.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create a synthetic dataset, with specified characteris like clusters, etc 
X, y = make_blobs(n_samples=500, centers=2, random_state=0, cluster_std=0.40)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SVM classifier
clf = SVC(kernel='linear')

# Fit the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Measure accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Plotting decision boundary
# c = is the colour label , s - size of the points, cmap - colour map to the poitns
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='spring')
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to plot decision boundary
xx = np.linspace(xlim[0], xlim[1], 100)
yy = np.linspace(ylim[0], ylim[1], 100)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# Plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5)
plt.title("SVM Decision Boundary")
plt.show()
```

### 2. SVM for Regression (Support Vector Regression - SVR)

In this example, we will use Support Vector Regression to fit a model to a dataset.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# Create sample data
X = np.array([[i] for i in range(10)])
y = np.array([1, 2, 1.5, 3, 2.5, 4, 3.5, 5, 4.5, 6])

# Fit SVR model
svr = SVR(kernel='linear', C=1.0)
svr.fit(X, y)

# Predict
X_test = np.array([[i] for i in range(0, 10, 1)])
y_pred = svr.predict(X_test)

# Plotting the results
plt.scatter(X, y, color='red', label='Data Points')
plt.plot(X_test, y_pred, color='blue', label='SVR Prediction')
plt.title("Support Vector Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
```

### 3. SVM with Kernel Trick

This example illustrates how to use the RBF kernel to classify non-linearly separable data.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create a non-linearly separable dataset
X, y = make_circles(n_samples=500, noise=0.1, factor=0.2, random_state=42)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SVM classifier with RBF kernel
clf = SVC(kernel='rbf', gamma='scale')

# Fit the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Measure accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Plotting decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='spring')
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to plot decision boundary
xx = np.linspace(xlim[0], xlim[1], 100)
yy = np.linspace(ylim[0], ylim[1], 100)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# Plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5)
plt.title("SVM with RBF Kernel Decision Boundary")
plt.show()
```

These examples demonstrate how to implement SVM for both classification and regression tasks using Python. The `scikit-learn` library provides a straightforward interface for working with SVMs, making it easy to apply these powerful algorithms to various datasets.

Citations:
[1] https://metana.io/blog/support-vector-machine-svm-classifier-in-python-svm-classifier-python-code/
[2] https://vitalflux.com/classification-model-svm-classifier-python-example/
[3] https://www.shiksha.com/online-courses/articles/support-vector-machines-python-code/
[4] https://www.geeksforgeeks.org/classifying-data-using-support-vector-machinessvms-in-python/
[5] https://www.datacamp.com/tutorial/svm-classification-scikit-learn-python


--------------
Support Vector Machines (SVM) are a class of supervised machine learning algorithms primarily used for classification tasks, although they can also be applied to regression problems. Developed by Vladimir Vapnik and his colleagues in the 1990s, SVMs are particularly effective in high-dimensional spaces and are known for their ability to handle both linear and non-linear classification tasks.

## Core Concepts of SVM

### 1. Decision Boundary and Hyperplane

At the heart of SVM is the concept of a **hyperplane**, which acts as a decision boundary that separates different classes in the feature space. In a two-dimensional space, this hyperplane is simply a line, while in higher dimensions, it becomes a plane or a hyperplane. The goal of the SVM algorithm is to find the optimal hyperplane that maximizes the margin between the closest data points from each class, known as **support vectors**. 

The mathematical representation of the hyperplane can be expressed as:

$$
wx + b = 0
$$

where $$w$$ is the weight vector, $$x$$ is the input feature vector, and $$b$$ is the bias term. The margin is defined as the distance between the hyperplane and the nearest data points from either class.

### 2. Support Vectors

Support vectors are the data points that lie closest to the decision boundary. These points are crucial because they directly influence the position and orientation of the hyperplane. If these points were removed, the hyperplane could change, which is why SVMs focus on these specific points for constructing the model.

### 3. Linear vs. Non-Linear SVM

- **Linear SVM**: This is used when the data is linearly separable, meaning that a straight line (or hyperplane in higher dimensions) can perfectly separate the classes.

- **Non-Linear SVM**: When data is not linearly separable, SVMs utilize **kernel functions** to transform the data into a higher-dimensional space where a linear separation is possible. This transformation is often referred to as the **kernel trick**.

### 4. Kernel Functions

Kernel functions enable SVMs to operate in high-dimensional spaces without explicitly computing the coordinates of the transformed space. Common kernel functions include:

- **Linear Kernel**: Suitable for linearly separable data.
  
$$
K(x_i, x_j) = x_i^T x_j
$$

- **Polynomial Kernel**: Useful for data that can be separated by polynomial boundaries.

$$
K(x_i, x_j) = (x_i^T x_j + c)^d
$$

- **Radial Basis Function (RBF) Kernel**: Effective for non-linear data, it measures the distance between points in the feature space.

$$
K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)
$$

- **Sigmoid Kernel**: Similar to the RBF but uses the hyperbolic tangent function.

$$
K(x_i, x_j) = \tanh(\alpha x_i^T x_j + c)
$$

## Advantages of SVM

1. **High Dimensionality**: SVMs perform well in high-dimensional spaces, making them suitable for applications like text classification and image recognition.

2. **Robustness to Overfitting**: The margin maximization principle helps SVMs generalize better to unseen data, reducing the risk of overfitting.

3. **Versatility**: SVMs can be applied to both classification and regression tasks, adapting to various types of data.

4. **Effective with Limited Data**: SVMs can achieve good performance even with a small amount of training data due to their reliance on support vectors.

5. **Non-linear Classification**: With the use of kernel functions, SVMs can effectively classify non-linearly separable data.

## Applications of SVM

SVMs are widely used in various fields, including:

- **Text Classification**: For tasks like spam detection or sentiment analysis.
  
- **Image Classification**: In facial recognition and object detection.

- **Bioinformatics**: For gene classification and protein structure prediction.

- **Anomaly Detection**: Identifying outliers in datasets.

## Conclusion

Support Vector Machines are a powerful and versatile tool in machine learning, particularly effective for classification tasks. Their ability to handle both linear and non-linear data through the use of kernel functions, combined with their robustness to overfitting and effectiveness in high-dimensional spaces, makes them a popular choice for many real-world applications.

Citations:
[1] https://www.ibm.com/topics/support-vector-machine
[2] https://monkeylearn.com/blog/introduction-to-support-vector-machines-svm/
[3] https://www.techtarget.com/whatis/definition/support-vector-machine-SVM
[4] https://www.geeksforgeeks.org/support-vector-machine-algorithm/
[5] https://en.wikipedia.org/wiki/Support_vector_machine