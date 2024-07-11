import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib


# Load dataset from CSV file
data = pd.read_csv('klipper_training_data.csv')

# Features and target
X = data[['distance', 'angle', 'object_width', 'object_height', 'object_depth', 'layer_height']]
y = data['threshold']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print("Mean Squared Error on Training Data:", mse_train)
print("Mean Squared Error on Test Data:", mse_test)

# Save model
joblib.dump(model, 'threshold_model.pkl')
