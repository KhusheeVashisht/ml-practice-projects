import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
data = pd.read_csv("house_prices.csv")

# Multiple features
X = data[["sqft_living","bedrooms","bathrooms"]]
y = data["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Calculate MSE
mse = mean_squared_error(y_test, pred)
print("Mean Squared Error:", mse)

# Scatter plot of actual data
plt.scatter(data["sqft_living"], y, color="red", label="Actual Prices")

# Create regression line
sqft_range = np.linspace(data["sqft_living"].min(), data["sqft_living"].max(), 100)

bedrooms_avg = data["bedrooms"].mean()
bathrooms_avg = data["bathrooms"].mean()

X_line = pd.DataFrame({
    "sqft_living": sqft_range,
    "bedrooms": bedrooms_avg,
    "bathrooms": bathrooms_avg
})

price_pred = model.predict(X_line)

# Plot regression line
plt.plot(sqft_range, price_pred, color="blue", label="Regression Line")

plt.xlabel("sqft_living")
plt.ylabel("Price")
plt.title("Multiple Linear Regression")
plt.legend()

plt.show()