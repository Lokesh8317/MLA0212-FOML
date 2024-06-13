import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
np.random.seed(42)
num_samples = 1000
income = np.random.rand(num_samples) * 100
age = np.random.rand(num_samples) * 100
debt = np.random.rand(num_samples) * 100
X = np.vstack((income, age, debt)).T
y = np.random.randint(2, size=num_samples)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
data = pd.DataFrame(X, columns=['Income', 'Age', 'Debt'])
data['CreditScore'] = y
print("\nSample data:")
print(data.head())
