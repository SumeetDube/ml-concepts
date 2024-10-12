Feature encoding is a process used to convert categorical data into a format that can be provided to machine learning algorithms to improve predictions. Two common methods of feature encoding are One-Hot Encoding and Label Encoding.

### One-Hot Encoding
One-Hot Encoding converts categorical values into a format where each unique value is represented as a binary vector. This method is suitable when categorical variables are nominal (no ordinal relationship).

**Example:**
If we have a categorical feature `color` with values `red`, `green`, `blue`, one-hot encoding will convert it into:
- `red` -> [1, 0, 0]
- `green` -> [0, 1, 0]
- `blue` -> [0, 0, 1]

**Code Example:**
```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Example data
data = pd.DataFrame({'color': ['red', 'green', 'blue', 'green', 'red']})

# Initialize the encoder
encoder = OneHotEncoder(sparse=False)

# Fit and transform the data
one_hot_encoded_data = encoder.fit_transform(data[['color']])

print("One-Hot Encoded Data:\n", one_hot_encoded_data)
```

```python
import pandas as pd 
# Step 1: Create a sample DataFrame 
data = { 'team': ['A', 'A', 'B', 'B', 'B', 'B', 'C', 'C'], 'points': [25, 12, 15, 14, 19, 23, 25, 29] } 
df = pd.DataFrame(data) 
# Step 2: Perform one-hot encoding 
df_encoded = pd.get_dummies(df, columns=['team']) 
# Display the one-hot encoded DataFrame 
print(df_encoded)
```
### Label Encoding
Label Encoding converts categorical values into numeric values. Each unique category is assigned an integer value. This method is suitable for ordinal categorical variables (where the order matters).

**Example:**
If we have a categorical feature `color` with values `red`, `green`, `blue`, label encoding will convert it into:
- `red` -> 2
- `green` -> 1
- `blue` -> 0

**Code Example:**
```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Example data
data = pd.DataFrame({'color': ['red', 'green', 'blue', 'green', 'red']})

# Initialize the encoder
encoder = LabelEncoder()

# Fit and transform the data
label_encoded_data = encoder.fit_transform(data['color'])

print("Label Encoded Data:\n", label_encoded_data)
```

### Explanation
1. **One-Hot Encoding**:
   - `OneHotEncoder` is initialized with `sparse=False` to return a dense array.
   - The data is fit and transformed using `fit_transform` to get the one-hot encoded format.

2. **Label Encoding**:
   - `LabelEncoder` is initialized.
   - The data is fit and transformed using `fit_transform` to get the label encoded format.

### When to Use
- **One-Hot Encoding**: Use when the categorical variable is nominal and there is no intrinsic ordering (e.g., color, product type).
- **Label Encoding**: Use when the categorical variable is ordinal and there is a meaningful ordering (e.g., ratings, rankings).

These encoding techniques are essential for preparing categorical data for machine learning models, ensuring that the algorithms can process and learn from the data effectively.