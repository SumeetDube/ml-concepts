Here are some examples of how to implement the ARIMA (AutoRegressive Integrated Moving Average) model in Python using the `statsmodels` library:

### Example 1: Basic ARIMA Implementation

This example demonstrates the basic steps to fit an ARIMA model to a time series dataset.

```python
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot as plt

# Load dataset
data = pd.read_csv('shampoo-sales.csv', header=0, parse_dates=True, index_col=0)

# Fit ARIMA model
model = ARIMA(data, order=(5, 1, 0))  # p=5, d=1, q=0
model_fit = model.fit()

# Print model summary
print(model_fit.summary())

# Generate forecasts
forecast = model_fit.forecast(steps=10)
print(forecast)

# Plot the results
plt.plot(data, label='Observed')
plt.plot(forecast, label='Forecast', color='red')
plt.legend()
plt.show()
```

### Example 2: Walk-Forward Validation

This example shows how to perform walk-forward validation with ARIMA to evaluate model performance.

```python
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot as plt

# Load dataset
series = pd.read_csv('shampoo-sales.csv', header=0, parse_dates=True, index_col=0)
X = series.values

# Split into train and test sets
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()

# Walk-forward validation
for t in range(len(test)):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

# Evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

# Plot forecasts against actual outcomes
plt.plot(test, label='Test Data')
plt.plot(predictions, color='red', label='Predictions')
plt.legend()
plt.show()
```

### Example 3: Using ACF and PACF for Parameter Selection

This example illustrates how to determine the parameters $$p$$, $$d$$, and $$q$$ using ACF and PACF plots.

```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Load dataset
data = pd.read_csv('shampoo-sales.csv', header=0, parse_dates=True, index_col=0)

# Plot ACF and PACF
plot_acf(data)
plt.title('ACF Plot')
plt.show()

plot_pacf(data)
plt.title('PACF Plot')
plt.show()

# Fit ARIMA model based on ACF and PACF analysis
model = ARIMA(data, order=(2, 1, 1))  # Example parameters based on analysis
model_fit = model.fit()
print(model_fit.summary())
```

These examples provide a foundation for implementing ARIMA models in Python. Each example focuses on different aspects of the modeling process, from basic fitting to evaluation and parameter selection.

Citations:
[1] https://www.projectpro.io/article/how-to-build-arima-model-in-python/544
[2] https://www.kdnuggets.com/2023/08/times-series-analysis-arima-models-python.html
[3] https://www.javatpoint.com/arima-model-in-python
[4] https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
[5] https://corporatefinanceinstitute.com/resources/data-science/autoregressive-integrated-moving-average-arima/

-----
![[Pasted image 20240808071951.png]]
The **Autoregressive Integrated Moving Average (ARIMA)** model is a widely used statistical method for time series forecasting, integrating three key components: Autoregression (AR), Integration (I), and Moving Average (MA). Each of these components plays a crucial role in analyzing and predicting future values based on historical data.

## Components of ARIMA

1. **Autoregressive (AR)**: This part of the model indicates that the current value of the series is regressed on its own previous values. The parameter $p$ represents the number of lagged observations included in the model.
2. **Integrated (I)**: This component involves differencing the data to achieve stationarity, meaning that the statistical properties of the series do not change over time. The parameter $d$ denotes the number of times the data has been differenced.
3. **Moving Average (MA)**: This aspect of the model captures the relationship between an observation and a residual error from a moving average model applied to lagged observations. The parameter $q$ indicates the size of the moving average window.

The ARIMA model is typically expressed as ARIMA($p$, $d$, $q$), where $p$, $d$, and $q$ are non-negative integers that define the model's structure.

## Application and Importance

ARIMA is particularly effective for time series data that exhibit non-stationarity, where trends or seasonal patterns may be present. By applying differencing, ARIMA can stabilize the mean of the time series, allowing for more accurate forecasting. The model is extensively used in various fields, including economics, finance, and environmental studies, to predict future trends based on historical data patterns.

## Estimation and Model Fitting

To fit an ARIMA model, one typically uses historical time series data to estimate the coefficients of the model. This process often involves techniques like Maximum Likelihood Estimation (MLE) to determine the best-fitting parameters. Additionally, the selection of $p$, $d$, and $q$ can be challenging and may require trial and error or automated methods such as Auto-ARIMA, which helps streamline the model selection process by evaluating multiple configurations. In summary, ARIMA is a powerful tool for time series analysis that combines autoregressive and moving average components while addressing non-stationarity through integration, making it a cornerstone technique in statistical forecasting.