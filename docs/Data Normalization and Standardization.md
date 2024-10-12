## Normalization vs Standardization in Machine Learning

In machine learning, normalization and standardization are two common techniques used for feature scaling. Both methods aim to transform features to a common scale, but they differ in their approach and the resulting distribution of the scaled data.

### Normalization (Min-Max Scaling)

Normalization, also known as min-max scaling, transforms features to a common range, typically between 0 and 1. It is calculated using the formula:

$$X_{new} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

where $X_{min}$ and $X_{max}$ are the minimum and maximum values of the feature, respectively[1][2].

Normalization is useful when:
- The data does not follow a normal distribution[1]
- The features have varying ranges[2]
- The algorithm is sensitive to the magnitude of the features[4]

However, normalization is sensitive to outliers, as they can significantly affect the minimum and maximum values[3].

### Standardization (Z-Score Normalization)

Standardization, also known as z-score normalization, transforms features to have a mean of 0 and a standard deviation of 1. It is calculated using the formula:

$$X_{new} = \frac{X - \mu}{\sigma}$$

where $\mu$ is the mean and $\sigma$ is the standard deviation of the feature[1][2].

Standardization is useful when:
- The data follows a normal distribution[1]
- The algorithm assumes a Gaussian distribution of the features[2]
- The features have different scales[4]

Standardization is less affected by outliers compared to normalization, as it does not have a predefined range[3].

## Key Differences

1. **Normalization** uses the minimum and maximum values for scaling, while **standardization** uses the mean and standard deviation[5].
2. **Normalization** scales values between [0, 1] or [-1, 1], while **standardization** has no bounded range[1][2].
3. **Normalization** is sensitive to outliers, while **standardization** is less affected by them[3].
4. **Normalization** is useful when the data does not follow a normal distribution, while **standardization** is useful when the data follows a Gaussian distribution[1][2].
5. **Normalization** is often used in algorithms that do not assume any distribution of data, such as k-nearest neighbors and neural networks, while **standardization** is used in algorithms that assume a Gaussian distribution, such as linear regression[1].
-----------

### Normalization
Normalization can be done using the `MinMaxScaler` from Scikit-learn, which scales the data to a specified range, typically [0, 1].

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Example data
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

# Initialize the scaler
scaler = MinMaxScaler()

# Fit and transform the data
normalized_data = scaler.fit_transform(data)

print("Normalized Data:\n", normalized_data)
```

### Standardization
Standardization can be done using the `StandardScaler` from Scikit-learn, which scales the data to have a mean of 0 and a standard deviation of 1.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

# Example data
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the data
standardized_data = scaler.fit_transform(data)

print("Standardized Data:\n", standardized_data)
```

### Explanation
1. **Data Preparation**: 
   - The example data is a 3x3 NumPy array for simplicity.
   
2. **Normalization**:
   - `MinMaxScaler` is initialized.
   - `fit_transform` method is used to normalize the data to the range [0, 1].
   
3. **Standardization**:
   - `StandardScaler` is initialized.
   - `fit_transform` method is used to standardize the data to have a mean of 0 and a standard deviation of 1.

These examples illustrate how to scale data using Scikit-learn, which is crucial for many machine learning algorithms.