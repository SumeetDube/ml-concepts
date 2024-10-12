## Grid Search Example

Let's say we want to tune the hyperparameters of a Support Vector Machine (SVM) classifier on the Iris dataset using Grid Search:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for the grid search
param_grid = {
    'C': [0.1, 1, 10], # Values of the regularization parameter
    'kernel': ['linear', 'rbf'], # Types of kernel functions
    'gamma': ['scale', 'auto'] # Kernel coefficient for 'rbf' kernel
}

# Create the SVM classifier
svm = SVC()

# Create the GridSearchCV object
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')

# Perform the grid search on the training data
grid_search.fit(X_train, y_train)

# Print the best parameters and the corresponding accuracy
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

# Evaluate the best model on the test data
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print("Test Accuracy of Best Model:", test_accuracy)
```

This will perform a grid search over the specified hyperparameters for the SVM classifier and report the best parameters and accuracy[2].

## Randomized Search Example

Randomized Search is similar to Grid Search, but instead of evaluating all possible combinations, it randomly samples a fixed number of combinations from the specified distributions. Here's an example using the Iris dataset:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from scipy.stats import uniform, randint

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter distributions for randomized search
param_distributions = {
    'C': uniform(loc=0.1, scale=10.0), # Uniform distribution for C
    'kernel': ['linear', 'rbf'], # Discrete values for kernel
    'gamma': uniform(loc=0.01, scale=0.99) # Uniform distribution for gamma
}

# Create the SVM classifier
svm = SVC()

# Create the RandomizedSearchCV object
random_search = RandomizedSearchCV(svm, param_distributions, n_iter=20, cv=5, scoring='accuracy', random_state=42)

# Perform the randomized search on the training data
random_search.fit(X_train, y_train)

# Print the best parameters and the corresponding accuracy
print("Best Parameters:", random_search.best_params_)
print("Best Accuracy:", random_search.best_score_)

# Evaluate the best model on the test data
best_model = random_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print("Test Accuracy of Best Model:", test_accuracy)
```

Citations:
[1] https://towardsdatascience.com/grid-search-in-python-from-scratch-hyperparameter-tuning-3cca8443727b?gi=cd4a43d1e697
[2] https://spotintelligence.com/2023/08/17/grid-search/
[3] https://drbeane.github.io/python_dsci/pages/grid_search.html
[4] https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
[5] https://www.geeksforgeeks.org/grid-searching-from-scratch-using-python/


---------
Hyperparameter tuning is a critical step in optimizing machine learning models, as it involves finding the best settings for hyperparameters—parameters that are not learned from the data but set prior to training. Two widely used methods for hyperparameter tuning are **Grid Search** and **Randomized Search**.

## Grid Search

Grid search is an exhaustive search method that evaluates every possible combination of hyperparameter values specified in a grid. The process includes the following steps:

1.  **Define Hyperparameters**: A grid is created by specifying a list of values for each hyperparameter. For example, if optimizing hyperparameters $\alpha$ and $\beta$ , you might specify:
    
    -   $\alpha$ : \[0.01, 0.1, 1.0, 10.0\]
    -   $\beta$ : \[0.01, 0.1, 1.0, 10.0\]
    
2.  **Model Training**: The model is trained for every combination of these values, typically using cross-validation to evaluate performance. This helps ensure that the model's performance is not overly optimistic due to overfitting.
3.  **Performance Evaluation**: After training, the model's performance is assessed, and the combination of hyperparameters that yields the best performance is selected.

## Advantages and Disadvantages

-   **Advantages**:
    
    -   Guarantees finding the optimal combination of hyperparameters within the specified grid, making it suitable for smaller search spaces.
    
-   **Disadvantages**:
    
    -   Computationally expensive, especially with many hyperparameters or large value ranges, as it requires training a model for every combination, which can be infeasible for complex models.
    

## Randomized Search

Randomized search, on the other hand, samples a fixed number of hyperparameter combinations from specified distributions rather than evaluating all possible combinations. Here’s how it works:

1.  **Define Distributions**: Instead of a grid, you specify a distribution for each hyperparameter. For example, you might use a uniform distribution for $\alpha$ and $\beta$ .
2.  **Sampling**: Randomized search randomly samples a set number of combinations from these distributions and trains the model using these sampled values.
3.  **Performance Evaluation**: As with grid search, the performance of each sampled model is evaluated to determine the best hyperparameter settings.

## Advantages and Disadvantages

-   **Advantages**:
    
    -   More efficient than grid search, especially in high-dimensional hyperparameter spaces, as it can explore a wider range of values without the exhaustive computation of grid search.
    -   Reduces the risk of overfitting by not exhaustively searching all combinations.
    
-   **Disadvantages**:
    
    -   There is no guarantee that the best combination will be found, as it only samples a subset of the possible combinations.
    

## Conclusion

Both grid search and randomized search are valuable techniques for hyperparameter tuning in machine learning. The choice between them typically depends on the specific problem, the size of the hyperparameter space, and computational resources. For smaller hyperparameter spaces, grid search may be preferable, while randomized search is often more suitable for larger, more complex spaces due to its efficiency.
