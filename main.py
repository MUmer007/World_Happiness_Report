
# 🌍 World Happiness Report 2021 – Data Analysis & ML
# Author: Umer |

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# --- Load Dataset ---
file_path = r"D:\WORLD HAPPINESS REPORT\world-happiness-report-2021.csv"
data = pd.read_csv(file_path)


print("\n Dataset Loaded Successfully!\n")
print(data.head(5))
print("\nSummary Statistics:\n", data.describe())
print("\nDataset Info:\n")
data.info()


data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")

data.rename(columns={
    "ladder_score": "happiness_score",
    "country_name": "country",
    "logged_gdp_per_capita": "gdp_per_capita",
    "freedom_to_make_life_choices": "freedom",
    "perceptions_of_corruption": "perceptions_of_corruption",
    "healthy_life_expectancy": "healthy_life_expectancy",
    "social_support": "social_support"
}, inplace=True)


data.dropna(subset=["happiness_score"], inplace=True)

numeric_features = [
    "gdp_per_capita", "social_support", "healthy_life_expectancy",
    "freedom", "perceptions_of_corruption"
]

for col in numeric_features:
    data[col] = pd.to_numeric(data[col], errors="coerce")
    data[col].fillna(data[col].median(), inplace=True)
TabError
print("\n Missing Values After Cleaning:\n", data.isnull().sum())

#  Visualize Distribution of Happiness Scores


sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))

sns.histplot(
    data=data,
    x="happiness_score",
    bins=20,
    kde=True,
    color="#4A90E2",
    edgecolor="white",
    alpha=0.8
)

mean_score = data["happiness_score"].mean()
median_score = data["happiness_score"].median()

plt.axvline(mean_score, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_score:.2f}")
plt.axvline(median_score, color="green", linestyle=":", linewidth=2, label=f"Median: {median_score:.2f}")

plt.title("Distribution of World Happiness Scores (2021)", fontsize=16, fontweight="bold", pad=20)
plt.xlabel("Happiness Score")
plt.ylabel("Number of Countries")
plt.legend(title="Reference Lines")
plt.tight_layout()
plt.savefig("happiness_score_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

#  Visualize Top & Bottom 10 Countries


top10 = data.nlargest(10, "happiness_score")
bottom10 = data.nsmallest(10, "happiness_score")

# Top 10 Happiest Countries
plt.figure(figsize=(10, 6))
sns.barplot(
    data=top10,
    x="happiness_score",
    y="country",
    hue="country",       
    palette="Greens_r",
    edgecolor="black",
    legend=False        
)

plt.title("Top 10 Happiest Countries (2021)", fontsize=16, fontweight="bold", pad=15)
plt.xlabel("Happiness Score")
plt.ylabel("Country")

for i, val in enumerate(top10["happiness_score"]):
    plt.text(val + 0.1, i, f"{val:.2f}", va="center")

plt.tight_layout()
plt.savefig("top10_happiest_countries.png", dpi=300)
plt.show()

# Bottom 10 Least Happy Countries 
plt.figure(figsize=(10, 6))
sns.barplot(
    data=bottom10,
    x="happiness_score",
    y="country",
    hue="country",
    palette="Reds_r",
    edgecolor="black",
    legend=False
)

plt.title("Bottom 10 Least Happy Countries (2021)", fontsize=16, fontweight="bold", pad=15)
plt.xlabel("Happiness Score")
plt.ylabel("Country")

for i, val in enumerate(bottom10["happiness_score"]):
    plt.text(val + 0.1, i, f"{val:.2f}", va="center")

plt.tight_layout()
plt.savefig("bottom10_least_happy_countries.png", dpi=300)
plt.show()

# Visualize Correlation Heatmap


selected_cols = ["happiness_score"] + numeric_features
corr = data[selected_cols].corr()

plt.figure(figsize=(9, 7))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    cbar_kws={"label": "Correlation Coefficient"}
)
plt.title("Correlation Heatmap – World Happiness Report 2021", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=300)
plt.show()


#  Visualize Scatter & Regression Plots


for factor in numeric_features:
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=data, x=factor, y="happiness_score", color="royalblue", s=60, alpha=0.7, edgecolor="white")
    sns.regplot(data=data, x=factor, y="happiness_score", scatter=False, color="red", line_kws={"lw": 2})
    plt.title(f"Happiness vs {factor.replace('_', ' ').title()}", fontsize=15, fontweight="bold", pad=15)
    plt.xlabel(factor.replace('_', ' ').title())
    plt.ylabel("Happiness Score")
    plt.tight_layout()
    plt.savefig(f"happiness_vs_{factor}.png", dpi=300)
    plt.show()


# Linear Regression Model

X = data[numeric_features]
y = data["happiness_score"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

# --- Evaluate Model ---
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"\n📊 Model Performance:")
print(f"   R² Score: {r2:.3f}")
print(f"   Mean Squared Error: {mse:.3f}")

# --- Coefficients ---
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values("Coefficient", ascending=False)

print("\nFeature Importance:\n", coef_df)

# --- Visualize Coefficients ---
plt.figure(figsize=(8, 5))
sns.barplot(
    data=coef_df,
    x="Coefficient",
    y="Feature",
    hue="Feature",
    palette="coolwarm",
    edgecolor="black",
    legend=False
)

plt.title("Feature Importance (Linear Regression)", fontsize=14, fontweight="bold", pad=15)
plt.axvline(0, color="black", linewidth=1)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.show()

# --- Actual vs Predicted ---
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred, color="royalblue", s=70)
sns.lineplot(x=y_test, y=y_test, color="red", label="Perfect Prediction")
plt.title("Actual vs Predicted Happiness Scores", fontsize=14, fontweight="bold")
plt.xlabel("Actual Happiness Score")
plt.ylabel("Predicted Happiness Score")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300)
plt.show()

print("\n✅ Analysis Completed Successfully!")
