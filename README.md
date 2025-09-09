# NYC-Green-Taxi-March-2022
🚖 NYC Green Taxi Trip Analysis – March 2022
📌 Overview

This project analyzes the NYC Green Taxi dataset (March 2022) using Python, Streamlit, and Machine Learning.
It provides:

Interactive visualizations of trip patterns (by day, hour, duration, and distance).

Key metrics such as total trips, average duration, and average distance.

A predictive modeling module to estimate taxi fares (total_amount) using regression models.

The goal is to demonstrate data preprocessing, exploratory data analysis (EDA), and machine learning in a user-friendly dashboard.

📂 Dataset

Source: NYC TLC Trip Record Data

File used: green_tripdata_2022-03.parquet

Contains pickup/dropoff times, trip distance, passenger count, fare details, and more.

⚙️ Features
🔎 Exploratory Analysis

Filter trips by weekday and hour.

View trip counts by weekday.

Explore relationship between trip distance vs. duration.

📊 Key Metrics

Total trips (after filters).

Average trip duration (minutes).

Average trip distance (miles).

🧠 Predictive Modeling

Run multiple ML models to predict total fare amount:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Gradient Boosting Regressor

Each model outputs:

R² Score

RMSE (Root Mean Squared Error)

🚀 Getting Started
🔧 Requirements

Install dependencies with:

pip install -r requirements.txt

▶️ Run Streamlit App
streamlit run app.py


The app will open in your browser (default: http://localhost:8501).

📓 Run the Notebook

You can also explore the analysis in Jupyter:

jupyter notebook "NYC_Green_Taxi_March_2022 (1).ipynb"

📷 Screenshots

Trips by Weekday


Trip Duration vs Distance



👩‍💻 Tech Stack

Frontend: Streamlit

Data Analysis: Pandas, NumPy, Matplotlib, Seaborn

Machine Learning: Scikit-learn (Regression Models)

Data Format: Parquet (via PyArrow)

🙌 Contributors

Thrisha Reddy J
