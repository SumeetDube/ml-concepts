Time series feature engineering is a crucial process in preparing time-dependent data for machine learning models. It involves creating new features from existing time series data to improve the model's predictive performance. This process typically includes generating lag features, rolling window statistics, and time-based features.

## Key Concepts in Time Series Feature Engineering

1. **Lag Features**: These are previous values of the time series that can help capture trends and patterns. For instance, if predicting today's sales, using yesterday's sales as a feature can provide valuable insights.

2. **Window Features**: These summarize values over a fixed window of prior time steps. For example, calculating the average sales over the last week can help smooth out noise and highlight underlying trends.

3. **Time-based Features**: These include components of the date and time, such as the day of the week, month, or whether a date falls on a holiday. These features can capture seasonality and other temporal patterns that are significant for forecasting.

## Python Code Examples

Here are some practical Python code examples demonstrating how to create these features using the `pandas` library.

### 1. Creating Lag Features

```python
import pandas as pd

# Sample time series data
data = {'date': pd.date_range(start='2022-01-01', periods=10, freq='D'),
        'sales': [100, 120, 130, 110, 150, 160, 170, 180, 190, 200]}
df = pd.DataFrame(data)

# Set the date as the index
df.set_index('date', inplace=True)

# Create lag features
df['lag_1'] = df['sales'].shift(1)  # Lag of 1 day
df['lag_2'] = df['sales'].shift(2)  # Lag of 2 days

print(df)
```

### 2. Creating Rolling Window Features

```python
# Create rolling window features
df['rolling_mean_3'] = df['sales'].rolling(window=3).mean()  # 3-day rolling mean
df['rolling_std_3'] = df['sales'].rolling(window=3).std()    # 3-day rolling standard deviation

print(df)
```

### 3. Creating Time-based Features

```python
# Create time-based features
df['day_of_week'] = df.index.dayofweek  # Day of the week (0=Monday, 6=Sunday)
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)  # Weekend indicator

print(df)
```

## Conclusion

Time series feature engineering is essential for transforming raw time series data into a format suitable for machine learning. By creating lag features, rolling statistics, and time-based features, data scientists can uncover patterns that enhance model performance. The provided Python code examples illustrate how to implement these techniques using the `pandas` library, making it easier to prepare time series data for analysis and forecasting tasks[1][2][4].

Citations:
[1] https://machinelearningmastery.com/basic-feature-engineering-time-series-data-python/
[2] https://dotdata.com/blog/practical-guide-for-feature-engineering-of-time-series-data/
[3] https://www.trainindata.com/p/feature-engineering-for-forecasting
[4] https://towardsdatascience.com/automate-time-series-feature-engineering-in-a-few-lines-of-python-code-f28fe52e4704?gi=17f2c8058c9c
[5] https://www.timescale.com/blog/how-to-work-with-time-series-in-python/