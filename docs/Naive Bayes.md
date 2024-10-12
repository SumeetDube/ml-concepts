### Example 1: Naive Bayes from Scratch

```python
import numpy as np
import pandas as pd

# Load dataset
data = pd.read_csv('data.csv')  # Replace with your dataset

# Separate by class
def separate_by_class(data):
    separated = {}
    for i in range(len(data)):
        vector = data[i, :-1]  # All but last column
        label = data[i, -1]    # Last column
        if label not in separated:
            separated[label] = []
        separated[label].append(vector)
    return separated

# Calculate mean and standard deviation
def summarize_dataset(dataset):
    summaries = [(np.mean(attribute), np.std(attribute)) for attribute in zip(*dataset)]
    return summaries

# Calculate probabilities
def calculate_probability(x, mean, std):
    exponent = np.exp(-(np.power(x - mean, 2) / (2 * np.power(std, 2))))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

# Make predictions
def predict(summaries, input_data):
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = 1
        for i in range(len(class_summaries)):
            mean, std = class_summaries[i]
            x = input_data[i]
            probabilities[class_value] *= calculate_probability(x, mean, std)
    return max(probabilities, key=probabilities.get)

# Example usage
data = np.array([[1.0, 2.0, 0],
                 [1.5, 1.8, 0],
                 [5.0, 8.0, 1],
                 [6.0, 9.0, 1]])
separated = separate_by_class(data)
summaries = {label: summarize_dataset(np.array(rows)) for label, rows in separated.items()
prediction = predict(summaries, np.array([1.2, 1.9]))
print(f'Predicted class: {prediction}')
```

### Example 2: Using Scikit-learn

This example shows how to use the Scikit-learn library to implement the Naive Bayes classifier, which is simpler and more efficient for practical applications.

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
```

### Example 3: Text Classification with Scikit-learn

This example illustrates how to use Naive Bayes for text classification, such as spam detection.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Sample data
data = [
    ('Free money now!!!', 1),  # Spam
    ('Hi Bob, how about a game of golf tomorrow?', 0),  # Not spam
    ('Limited time offer!', 1),  # Spam
    ('Are we still on for lunch?', 0)  # Not spam
]

# Split data into features and labels
X, y = zip(*data)

# Create a model pipeline
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
model.fit(X, y)

# Make a prediction
predicted = model.predict(['Win a free ticket to the concert!'])
print(f'Predicted class: {"Spam" if predicted[0] else "Not Spam"}')
```

Citations:
[1] https://www.geeksforgeeks.org/ml-naive-bayes-scratch-implementation-using-python/
[2] https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
[3] https://www.kaggle.com/code/prashant111/naive-bayes-classifier-in-python
[4] https://www.datacamp.com/tutorial/naive-bayes-scikit-learn
[5] https://www.javatpoint.com/machine-learning-naive-bayes-classifier


----------
Naive Bayes is a family of probabilistic algorithms used primarily for classification tasks in machine learning. Its foundation lies in Bayes' Theorem, which relates the conditional and marginal probabilities of random events. Despite its simplicity and the "naive" assumption of feature independence, Naive Bayes classifiers are widely appreciated for their speed and effectiveness, particularly in high-dimensional datasets such as text classification.

## Bayes' Theorem

Bayes' Theorem provides a way to update the probability estimate for a hypothesis as more evidence or information becomes available. The theorem is mathematically expressed as:

$P(c|x)=\frac{P(x|c)\cdot P(c)}{P(x)}$

Where:

-   $P(c|x)$ is the posterior probability of class $c$ given feature vector $x$.
-   $P(x|c)$ is the likelihood of feature vector $x$ given class $c$.
-   $P(c)$ is the prior probability of class $c$.
-   $P(x)$ is the prior probability of feature vector $x$.

In Naive Bayes, the key assumption is that all features are conditionally independent given the class label. This simplifies the computation of $P(x|c)$ to the product of individual probabilities for each feature:

$P(x|c)=P(x_1|c)\cdot P(x_2|c)\cdots P(x_n|c)$

## Types of Naive Bayes Classifiers

There are several variations of Naive Bayes classifiers, each suited for different types of data:

1.  **Gaussian Naive Bayes**: Assumes that the continuous features follow a normal distribution. This is commonly used when dealing with real-valued data.
2.  **Multinomial Naive Bayes**: Particularly effective for discrete data, such as word counts in text classification. It models the feature counts based on a multinomial distribution.
 <iframe width="560" height="315" src="https://www.youtube.com/embed/BZxCIkSkMgo?si=KR3vXbhkuJM3k7pK" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
4.  **Bernoulli Naive Bayes**: Similar to Multinomial Naive Bayes but assumes binary features (0s and 1s). It is useful for binary/boolean features.<iframe width="560" height="315" src="https://www.youtube.com/embed/nl9WiZMZnYs?si=CcDrPiph98mRUu0o" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Advantages and Disadvantages

## Advantages:

-   **Simplicity and Speed**: Naive Bayes is easy to implement and computationally efficient, making it suitable for large datasets.
-   **Performance with High Dimensional Data**: It performs well even when the number of features is large relative to the number of observations.
-   **Robustness**: It often outperforms more complex models when the dataset is small or when the independence assumption approximately holds.

## Disadvantages:

-   **Independence Assumption**: The assumption that all features are independent is rarely true in real-world scenarios, which can lead to inaccurate probability estimates.
-   **Zero Probability Problem**: If a feature value is not present in the training data for a particular class, it can lead to zero probabilities. This is often addressed using techniques like Laplace smoothing.

## Applications

Naive Bayes classifiers are widely used in various applications, including:

-   **Text Classification**: Such as spam detection in emails, sentiment analysis, and document categorization.
-   **Medical Diagnosis**: Assisting in predicting the likelihood of diseases based on patient data.
-   **Recommendation Systems**: Helping to classify items based on user preferences.
-   **Real-time Prediction**: Due to its speed, it is suitable for applications requiring quick predictions, such as online news categorization.

In summary, Naive Bayes classifiers leverage the simplicity of Bayes' theorem and the assumption of feature independence to provide a robust and efficient method for classification tasks, especially in domains with high-dimensional data. Despite its limitations, its effectiveness in many practical scenarios makes it a popular choice among data scientists and machine learning practitioners.