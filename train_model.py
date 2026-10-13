import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
data = pd.read_csv("data/sales_data.csv")
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date")
data["Month_Number"] = np.arange(len(data))
X = data[["Month_Number"]]
y = data["Sales"]
model = LinearRegression()
model.fit(X, y)
with open("sales_model.pkl", "wb") as file:
    pickle.dump(model, file)
print("================================")
print("MODEL TRAINED SUCCESSFULLY!")
print("================================")
print("R2 Score:", model.score(X, y))
future_months = np.arange(
    len(data),
    len(data) + 6
).reshape(-1, 1)
predictions = model.predict(future_months)
print("\nFUTURE SALES PREDICTIONS")
print("------------------------")
for i, prediction in enumerate(predictions, start=1):
    print(f"Month {i}: {prediction:.2f}")