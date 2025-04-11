# Streamlit App for NYC Green Taxi March 2022
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Set page configuration
st.set_page_config(page_title="NYC Green Taxi - March 2022", layout="wide")

st.title("🚖 NYC Green Taxi Trip Analysis - March 2022")

@st.cache_data

def load_data():
    df = pd.read_parquet("green_tripdata_2022-03.parquet")
    df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
    df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'])

    df['trip_duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60
    df['weekday'] = df['lpep_dropoff_datetime'].dt.day_name()
    df['hour'] = df['lpep_dropoff_datetime'].dt.hour

    drop_cols = ["fare_amount", "ehail_fee", "extra", "mta_tax", "tip_amount", 
                 "tolls_amount", "improvement_surcharge", "congestion_surcharge"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    df = df.dropna()
    return df

# Load data
df = load_data()
st.sidebar.header("Filter Trips")

# Sidebar filters
selected_day = st.sidebar.selectbox("Select Weekday", ["All"] + sorted(df["weekday"].unique()))
selected_hour = st.sidebar.selectbox("Select Hour", ["All"] + sorted(df["hour"].unique()))

filtered_df = df.copy()
if selected_day != "All":
    filtered_df = filtered_df[filtered_df["weekday"] == selected_day]
if selected_hour != "All":
    filtered_df = filtered_df[filtered_df["hour"] == int(selected_hour)]

# Metrics
st.subheader("🔢 Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Trips", len(filtered_df))
col2.metric("Avg Duration (min)", round(filtered_df["trip_duration"].mean(), 2))
col3.metric("Avg Distance (mi)", round(filtered_df["trip_distance"].mean(), 2))

# Visualizations
st.subheader("📊 Visualizations")

# Trips per weekday
fig1, ax1 = plt.subplots()
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
df["weekday"].value_counts().reindex(order).plot(kind="bar", ax=ax1, color="skyblue")
ax1.set_title("Trips by Weekday")
st.pyplot(fig1)

# Trip duration vs. distance
fig2, ax2 = plt.subplots()
sample = filtered_df.sample(min(1000, len(filtered_df)))
sns.scatterplot(x="trip_distance", y="trip_duration", data=sample, ax=ax2, alpha=0.4)
ax2.set_title("Trip Duration vs. Distance")
st.pyplot(fig2)

# Regression Model Section
st.subheader("🧠 Predictive Modeling")

if st.button("Run Regression Models"):
    if "total_amount" not in filtered_df.columns:
        st.error("Error: 'total_amount' column is missing in the dataset.")
    elif filtered_df.empty:
        st.warning("No data to train the model after filtering. Try changing the filters.")
    else:
        try:
            X = filtered_df.drop(columns=["total_amount", "lpep_pickup_datetime", "lpep_dropoff_datetime"], errors='ignore')
            y = filtered_df["total_amount"]
            X = pd.get_dummies(X, drop_first=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            def show_results(model, name):
                try:
                    model.fit(X_train, y_train)
                    preds = np.ravel(model.predict(X_test))
                    y_eval = np.ravel(y_test)
                    r2 = r2_score(y_eval, preds)
                    rmse = mean_squared_error(y_eval, preds) ** 0.5  # manually compute RMSE
                    st.write(f"**{name}**")
                    st.write(f"R2 Score: {r2:.3f}, RMSE: {rmse:.2f}")
                except Exception as e:
                    st.error(f"{name} failed: {e}")

            show_results(LinearRegression(), "Linear Regression")
            show_results(DecisionTreeRegressor(max_depth=10), "Decision Tree (max_depth=10)")
            show_results(RandomForestRegressor(n_estimators=100, max_depth=10), "Random Forest")
            show_results(GradientBoostingRegressor(n_estimators=100, max_depth=3), "Gradient Boosting")

        except Exception as e:
            st.error(f"Unexpected error during model execution: {e}")
