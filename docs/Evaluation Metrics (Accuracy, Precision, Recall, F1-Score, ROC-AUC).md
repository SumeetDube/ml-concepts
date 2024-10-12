### Example Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# For simplicity, we'll convert this to a binary classification problem
# Let's classify whether the species is Iris-Versicolor (1) or not (0)
y_binary = (y == 1).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Create and train the SVM classifier
svm = SVC(probability=True)  # Set probability=True for ROC-AUC
svm.fit(X_train, y_train)

# Make predictions
y_pred = svm.predict(X_test)
y_pred_proba = svm.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print the evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"ROC-AUC: {roc_auc:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Visualizing the Confusion Matrix
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Iris-Versicolor', 'Iris-Versicolor']).plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
```


------------
Evaluation metrics are essential for assessing the performance of machine learning models, especially in classification tasks. Here’s an overview of key evaluation metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

## Accuracy

Accuracy is the simplest evaluation metric, defined as the ratio of correctly predicted instances to the total instances in the dataset. It is calculated as:

$$
\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}
$$
$$accuracy = \frac{TP + TN} {Total}$$ 
While accuracy provides a quick snapshot of model performance, it can be misleading, especially in imbalanced datasets. For example, in a dataset where 95% of instances belong to one class, a model could achieve 95% accuracy by predicting only that class, failing to identify the minority class effectively[2][4].
![[Pasted image 20240802012045.png]]
pred +ve && acutally +ve = TP
pred +ve && actually -ve = FP
pred -ve && actually +ve = FN
pred -ve && actually -ve = TN

## Precision

Precision measures the accuracy of the positive predictions made by the model. It is defined as the ratio of true positive predictions to the total predicted positives (true positives + false positives):

$$
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
$$

$$TP/ Total predicted postive$$
High precision indicates that the model has a low false positive rate, which is crucial in scenarios where false positives are costly, such as in medical diagnoses[2][4].

## Recall

Recall, also known as Sensitivity or True Positive Rate, measures the model's ability to identify all relevant instances. It is defined as the ratio of true positive predictions to the total actual positives (true positives + false negatives):

$$
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
$$
$$TP/Total Ground Truth Postive$$

High recall is important in applications where missing a positive instance (like a disease diagnosis) is critical[2][3].

## F1-Score

The F1-Score is the harmonic mean of precision and recall, providing a balance between the two metrics. It is particularly useful in situations where there is an uneven class distribution. The F1-Score is calculated as:

$$
\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

An F1-Score close to 1 indicates a good balance between precision and recall, making it a preferred metric in many classification tasks[1][3].

## ROC-AUC

The Receiver Operating Characteristic (ROC) curve is a graphical representation of a model's performance across different threshold settings. The Area Under the Curve (AUC) measures the entire two-dimensional area underneath the ROC curve, providing an aggregate measure of performance across all classification thresholds.

AUC values range from 0 to 1, where:
- 1 indicates a perfect model,
- 0.5 indicates a model with no discrimination capability (random guessing),
- Values below 0.5 suggest that the model is performing worse than random guessing.

ROC-AUC is particularly useful for evaluating binary classifiers and is robust to class imbalance[1][3].

## Conclusion

In summary, these evaluation metrics provide valuable insights into the performance of machine learning models, helping to identify strengths and weaknesses. The choice of metric often depends on the specific application and the consequences of different types of errors.

Citations:
[1] https://www.aiacceleratorinstitute.com/evaluating-machine-learning-models-metrics-and-techniques/
[2] https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234?gi=00283f401208
[3] https://www.geeksforgeeks.org/metrics-for-machine-learning-model/
[4] https://www.javatpoint.com/performance-metrics-in-machine-learning
[5] https://www.shiksha.com/online-courses/articles/evaluating-a-machine-learning-algorithm/