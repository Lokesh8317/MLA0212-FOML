

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
data = {
    "Day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
    "Sales": [100, 120, 90, 110, 130, 150]
}
df = pd.DataFrame(data)
day_mapping = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}
df["Day_Num"] = df["Day"].map(day_mapping)
threshold = 110
df["High_Sales"] = (df["Sales"] > threshold).astype(int)
X = df["Day_Num"].values.reshape(-1, 1)
y = df["High_Sales"].values
model = LogisticRegression()
model.fit(X, y)
next_week_days = np.array([0, 1, 2, 3, 4, 5, 6]).reshape(-1, 1)
predicted_classes = model.predict(next_week_days)
days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
for day, prediction in zip(days_of_week, predicted_classes):
    sales_category = "High Sales" if prediction == 1 else "Low Sales"
    print(f"{day}: {sales_category}")
plt.figure(figsize=(10, 5))
plt.scatter(df["Day"], df["Sales"], color='r', label='Historical Sales')
plt.axhline(y=threshold, color='g', linestyle='--', label='Threshold')
plt.xlabel('Day')
plt.ylabel('Sales')
plt.title('Sales Classification for the Upcoming Week')
plt.legend()
plt.grid(True)
plt.show()
