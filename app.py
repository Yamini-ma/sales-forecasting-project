import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Sales Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sales Forecasting Using Machine Learning")
st.write("Analyze historical sales and predict future demand.")

uploaded = st.file_uploader(
    "Upload a CSV file (columns required: Date, Sales)",
    type=["csv"]
)

if uploaded is not None:
    data = pd.read_csv(uploaded)
else:
    data = pd.read_csv("sales_data.csv")

required = {"Date", "Sales"}

if not required.issubset(data.columns):
    st.error("CSV must contain Date and Sales columns.")
    st.stop()

data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
data["Sales"] = pd.to_numeric(data["Sales"], errors="coerce")

data = data.dropna(
    subset=["Date", "Sales"]
).sort_values("Date").drop_duplicates("Date")

if len(data) < 30:
    st.error("Please provide at least 30 days of historical sales.")
    st.stop()

# Feature engineering
data["day_index"] = (
    data["Date"] - data["Date"].min()
).dt.days

data["dayofweek"] = data["Date"].dt.dayofweek
data["month"] = data["Date"].dt.month
data["dayofyear"] = data["Date"].dt.dayofyear

features = [
    "day_index",
    "dayofweek",
    "month",
    "dayofyear"
]

X = data[features]
y = data["Sales"]

# Train-test split
split = int(len(data) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# Machine Learning model
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    max_depth=12
)

model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, pred)

rmse = np.sqrt(
    mean_squared_error(y_test, pred)
)

r2 = r2_score(y_test, pred)

# Dashboard
c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Sales",
    f"₹{data['Sales'].sum():,.0f}"
)

c2.metric(
    "Average Daily Sales",
    f"₹{data['Sales'].mean():,.0f}"
)

c3.metric(
    "MAE",
    f"{mae:,.0f}"
)

c4.metric(
    "R² Score",
    f"{r2:.2f}"
)

# Historical sales
st.subheader("Historical Sales")

fig, ax = plt.subplots(figsize=(11, 4))

ax.plot(
    data["Date"],
    data["Sales"]
)

ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.grid(alpha=0.25)

st.pyplot(fig)

# Actual vs predicted
st.subheader("Actual vs Predicted Sales")

test_dates = data["Date"].iloc[split:]

fig2, ax2 = plt.subplots(figsize=(11, 4))

ax2.plot(
    test_dates,
    y_test.values,
    label="Actual"
)

ax2.plot(
    test_dates,
    pred,
    label="Predicted"
)

ax2.set_xlabel("Date")
ax2.set_ylabel("Sales")

ax2.legend()
ax2.grid(alpha=0.25)

st.pyplot(fig2)

# Future forecasting
days = st.slider(
    "Forecast horizon (days)",
    7,
    90,
    30
)

future_dates = pd.date_range(
    data["Date"].max() + pd.Timedelta(days=1),
    periods=days
)

future = pd.DataFrame({
    "Date": future_dates
})

future["day_index"] = (
    future["Date"] - data["Date"].min()
).dt.days

future["dayofweek"] = future["Date"].dt.dayofweek
future["month"] = future["Date"].dt.month
future["dayofyear"] = future["Date"].dt.dayofyear

future["Forecast_Sales"] = model.predict(
    future[features]
).round(0)

st.subheader(
    f"Future Demand Forecast – {days} Days"
)

st.dataframe(
    future[["Date", "Forecast_Sales"]],
    use_container_width=True
)

# Forecast graph
fig3, ax3 = plt.subplots(figsize=(11, 4))

ax3.plot(
    data["Date"].tail(60),
    data["Sales"].tail(60),
    label="Historical"
)

ax3.plot(
    future["Date"],
    future["Forecast_Sales"],
    label="Forecast"
)

ax3.set_xlabel("Date")
ax3.set_ylabel("Sales")

ax3.legend()
ax3.grid(alpha=0.25)

st.pyplot(fig3)

# Download forecast
csv = future[
    ["Date", "Forecast_Sales"]
].to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇️ Download Forecast CSV",
    csv,
    "sales_forecast.csv",
    "text/csv"
)