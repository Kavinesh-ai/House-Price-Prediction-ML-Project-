import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load Dataset
data = pd.read_csv("housing.csv")

# Features
X = data[["area", "bedrooms"]]

# Target
y = data["price"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
score = r2_score(y_test, y_pred)

# Output
print("House Price Prediction")
print("-" * 35)

print("R2 Score:", round(score, 2))

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred.round(2)
})

print("\nSample Predictions")
print(results)