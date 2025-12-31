# World_Happiness_Report
🌍 World Happiness Report 2021 – Data Analysis & Machine Learning

This project performs Exploratory Data Analysis (EDA) and Linear Regression modeling on the World Happiness Report 2021 dataset.
The goal is to understand which socio-economic factors contribute most to a country’s happiness score and to build a predictive model using machine learning.

📁 Dataset Information

Dataset: World Happiness Report 2021

File: world-happiness-report-2021.csv

Observations: Countries worldwide

Target Variable: happiness_score

Key Features Used

gdp_per_capita

social_support

healthy_life_expectancy

freedom

perceptions_of_corruption

🛠️ Technologies & Libraries

Python

Pandas – Data cleaning & manipulation

NumPy – Numerical operations

Matplotlib & Seaborn – Data visualization

Scikit-learn – Machine learning (Linear Regression)

🧹 Data Cleaning & Preprocessing

Standardized column names (lowercase, underscores)

Renamed important columns for clarity

Removed rows with missing happiness scores

Converted numerical features to proper numeric types

Filled missing values using median imputation

Verified dataset integrity after cleaning

📊 Exploratory Data Analysis (EDA)
1️⃣ Distribution of Happiness Scores

Histogram with KDE

Mean and median reference lines

Output: happiness_score_distribution.png

2️⃣ Top & Bottom 10 Countries by Happiness

Horizontal bar charts

Value annotations for clarity

Outputs:

top10_happiest_countries.png

bottom10_least_happy_countries.png

3️⃣ Correlation Heatmap

Shows relationships between happiness and socio-economic factors

Output: correlation_heatmap.png

4️⃣ Scatter & Regression Plots

Happiness score vs each numeric feature

Includes best-fit regression line

Outputs:

happiness_vs_gdp_per_capita.png

happiness_vs_social_support.png

happiness_vs_healthy_life_expectancy.png

happiness_vs_freedom.png

happiness_vs_perceptions_of_corruption.png

🤖 Machine Learning Model
Model Used

Linear Regression

Train-Test Split

80% training

20% testing

random_state = 42

📈 Model Evaluation Metrics

R² Score – Measures goodness of fit

Mean Squared Error (MSE) – Measures prediction error

Model performance is printed in the console after training.

🔍 Feature Importance

Coefficients extracted from the regression model

Shows how strongly each factor impacts happiness

Output:

feature_importance.png

📉 Actual vs Predicted Values

Scatter plot comparing real vs predicted happiness scores

Perfect prediction reference line included

Output:

actual_vs_predicted.png

▶️ How to Run the Project

Install dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn


Update dataset path if needed:

file_path = "world-happiness-report-2021.csv"


Run the script:

python world_happiness_analysis.py

📌 Outputs

High-resolution .png visualizations saved automatically

Console output includes:

Dataset info

Summary statistics

Model performance metrics

Feature coefficients

🎯 Project Objectives

Perform real-world EDA

Understand global happiness indicators

Apply machine learning regression

Interpret feature importance

Practice end-to-end data science workflow

✨ Author

Umer Shaikh
