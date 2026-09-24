# src/train.py

from preprocess import load_data, clean_data, preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import joblib

# 1. Load and preprocess data
df = load_data('../data/insurance_dataset.csv')
df = clean_data(df)
df_encoded = preprocess_data(df)

# 2. Split features and target
X = df_encoded.drop('expenses', axis=1)
y = df_encoded['expenses']

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

print("Train R²:", r2_score(y_train, y_train_pred))
print("Test R²:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# 6. Save model and column structure
joblib.dump(model, '../models/insurance_model.pkl')
joblib.dump(X_train.columns.tolist(), '../models/model_columns.pkl')
print("Model saved successfully.")