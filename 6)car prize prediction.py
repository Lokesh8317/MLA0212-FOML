import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
data = {
    'Brand': ['Toyota', 'Honda', 'Ford', 'Toyota', 'Honda'],
    'Model': ['Corolla', 'Civic', 'Focus', 'Camry', 'Accord'],
    'Year': [2010, 2011, 2012, 2013, 2014],
    'Mileage': [150000, 120000, 130000, 140000, 110000],
    'FuelType': ['Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol'],
    'Transmission': ['Manual', 'Automatic', 'Manual', 'Automatic', 'Manual'],
    'Price': [5000, 6000, 5500, 7000, 7200]
}
df = pd.DataFrame(data)
label_encoders = {col: LabelEncoder() for col in ['Brand', 'Model', 'FuelType', 'Transmission']}
for col, le in label_encoders.items():
    df[col] = le.fit_transform(df[col])

X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)
