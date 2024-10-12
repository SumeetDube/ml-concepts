## Example 1: Iris Flower Classification

In this example, we'll use the classic iris flower dataset to classify iris flowers into three species based on their sepal and petal dimensions. We'll use scikit-learn to load the dataset and implement KNN.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a KNN classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model on the training data
knn.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn.predict(X_test)

# Evaluate the model's accuracy
accuracy = knn.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
```

## Example 2: Handwritten Digit Recognition

In this example, we'll use the MNIST dataset of handwritten digits to train a KNN model to recognize digits from 0 to 9.

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a KNN classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model on the training data
knn.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn.predict(X_test)

# Evaluate the model's accuracy
accuracy = knn.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
```

## Example 3: Breast Cancer Classification

In this example, we'll use the Wisconsin Breast Cancer dataset to classify tumors as benign or malignant based on various features.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load the breast cancer dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a KNN classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model on the training data
knn.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn.predict(X_test)

# Evaluate the model's accuracy
accuracy = knn.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
```

These examples demonstrate the versatility of KNN in solving classification problems across different domains. The key steps are:

1. Load the dataset and split it into training and testing sets.
2. Create a KNeighborsClassifier with the desired value of k.
3. Train the model on the training data using the `fit()` method.
4. Make predictions on the test data using the `predict()` method.
5. Evaluate the model's accuracy using the `score()` method.

The choice of k and the distance metric can significantly impact the model's performance. Experimenting with different values of k and using techniques like cross-validation can help find the optimal hyperparameters for a given problem.

Citations:
[1] https://www.almabetter.com/bytes/articles/knn-algorithm-python
[2] https://realpython.com/knn-python/
[3] https://domino.ai/blog/knn-with-examples-in-python
[4] https://www.w3schools.com/python/python_ml_knn.asp
[5] https://www.digitalocean.com/community/tutorials/k-nearest-neighbors-knn-in-python


--------
## KNN Regression Example

In this example, we will generate a simple dataset and use KNN to predict the target variable.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Generate synthetic data
np.random.seed(0)
X = np.random.rand(100, 1) * 10  # 100 data points, single feature
y = 2.5 * X.squeeze() + np.random.randn(100) * 2  # Linear relationship with noise

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a KNN regressor with k=5
knn = KNeighborsRegressor(n_neighbors=5)

# Train the model on the training data
knn.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Plotting the results
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='green', label='Test data')
plt.scatter(X_test, y_pred, color='red', label='Predictions')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('KNN Regression Example')
plt.legend()
plt.show()
```

---------------
K-Nearest Neighbors (K-NN) is a fundamental supervised machine learning algorithm used for both classification and regression tasks. It operates on the principle of similarity, where the classification or prediction of a new data point is determined based on the labels or values of its K nearest neighbors from a training dataset. This algorithm is particularly valued for its simplicity and effectiveness in various applications, including pattern recognition, data mining, and recommendation systems.

## Historical Background

The K-NN algorithm was initially developed by Evelyn Fix and Joseph Hodges in 1951, with significant contributions from Thomas Cover later on. It is classified as a non-parametric method, meaning it does not assume any specific distribution for the underlying data, which allows for greater flexibility in handling diverse datasets[1][3].

## How K-NN Works

### Steps Involved

1. **Choosing the Value of K**: The parameter $$ K $$ represents the number of nearest neighbors to consider when making predictions. The choice of $$ K $$ can significantly affect the model's performance; a smaller $$ K $$ may lead to noise sensitivity, while a larger $$ K $$ can smooth out the decision boundary.

2. **Calculating Distance**: To determine similarity, K-NN typically employs distance metrics such as Euclidean distance. The distance between the new data point and each point in the training dataset is calculated[1][3].

   $$
   \text{distance}(x, X_i) = \sqrt{\sum_{j=1}^{d} (x_j - X_{ij})^2}
   $$

   where $$ x $$ is the new data point, $$ X_i $$ is a training data point, and $$ d $$ is the number of features.

3. **Finding Nearest Neighbors**: The algorithm identifies the $$ K $$ data points in the training set that are closest to the new data point based on the calculated distances.

4. **Making Predictions**:
   - **For Classification**: The algorithm performs a majority voting among the $$ K $$ nearest neighbors. The class label that appears most frequently among the neighbors is assigned to the new data point.
   - **For Regression**: The predicted value is computed as the average of the values of the $$ K $$ nearest neighbors[1][3][4].

### Example Application

Consider a scenario where a company wants to target advertisements for a new SUV based on user data, including age and estimated salary. Using K-NN, the algorithm can classify users into categories (e.g., interested vs. not interested) based on their proximity to existing classified data points, thus optimizing marketing efforts[2].

## Advantages and Disadvantages

### Advantages

- **Simplicity**: K-NN is easy to understand and implement, making it accessible for beginners in machine learning.
- **Non-parametric**: It does not assume any underlying data distribution, allowing it to be applied to various datasets.
- **Versatile**: K-NN can handle both numerical and categorical data, making it suitable for a wide range of applications[1][3].

### Disadvantages

- **Computationally Intensive**: As K-NN stores the entire training dataset, it can be slow, especially with large datasets, as it requires distance calculations for every prediction.
- **Sensitive to Irrelevant Features**: The presence of irrelevant features can negatively impact the distance calculations and, consequently, the predictions.
- **Choice of K**: The performance of the algorithm heavily depends on the selection of $$ K $$, which may require experimentation to optimize[3][4].

## Applications of K-NN

K-NN is widely used across various domains, including:

- **Healthcare**: For predicting disease risks and patient outcomes.
- **Finance**: To assess credit risk and predict bankruptcies.
- **Marketing**: In customer segmentation and targeted advertising.
- **Agriculture**: For predicting crop yields and managing resources[3][4].

In summary, K-Nearest Neighbors is a powerful and flexible algorithm that leverages the concept of proximity to make predictions based on known data, making it a staple in the machine learning toolkit.

Citations:
[1] https://www.geeksforgeeks.org/k-nearest-neighbours/
[2] https://www.javatpoint.com/k-nearest-neighbor-algorithm-for-machine-learning
[3] https://www.pinecone.io/learn/k-nearest-neighbor/
[4] https://www.techopedia.com/definition/32066/k-nearest-neighbor-k-nn
[5] https://towardsdatascience.com/a-simple-introduction-to-k-nearest-neighbors-algorithm-b3519ed98e