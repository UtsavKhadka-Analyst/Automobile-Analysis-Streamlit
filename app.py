# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set Streamlit page title
st.title("Comprehensive Data Cleaning, Preprocessing, and Analysis")

# Load the dataset
st.header("Step 1: Load the Dataset")
filename = "https://raw.githubusercontent.com/ncit17153/Exam1_streamlit/refs/heads/main/Exam1_clean_df.csv"
headers = [
    "symboling", "normalized-losses", "make", "fuel-type", "aspiration",
    "num-of-doors", "body-style", "drive-wheels", "engine-location",
    "wheel-base", "length", "width", "height", "curb-weight", "engine-type",
    "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke",
    "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"
]

try:
    df = pd.read_csv(filename, names=headers)
    st.write("Raw dataset loaded:")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Replace '?' with NaN and ensure numeric columns
st.header("Step 2: Handle Missing Values")
df.replace("?", np.nan, inplace=True)
numeric_cols = ["normalized-losses", "bore", "stroke", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Count missing values
st.write("Missing values in each column:")
missing_data = df.isnull().sum()
st.write(missing_data)

# Fill missing values
st.write("Replacing missing values:")
df["normalized-losses"].fillna(df["normalized-losses"].mean(), inplace=True)
df["bore"].fillna(df["bore"].mean(), inplace=True)
df["stroke"].fillna(df["stroke"].mean(), inplace=True)
df["horsepower"].fillna(df["horsepower"].mean(), inplace=True)
df["peak-rpm"].fillna(df["peak-rpm"].mean(), inplace=True)
df["num-of-doors"].fillna("four", inplace=True)
df.dropna(subset=["price"], inplace=True)
df.reset_index(drop=True, inplace=True)

st.write("Dataset after handling missing values:")
st.dataframe(df.head())

# Data transformation
st.header("Step 3: Data Transformation")
df["city-L/100km"] = 235 / df["city-mpg"]
df["highway-L/100km"] = 235 / df["highway-mpg"]

st.write("Transformed dataset (mpg to L/100km):")
st.dataframe(df[["city-L/100km", "highway-L/100km"]].head())

# Data normalization
st.header("Step 4: Data Normalization")
df["length"] = df["length"] / df["length"].max()
df["width"] = df["width"] / df["width"].max()
df["height"] = df["height"] / df["height"].max()

st.write("Normalized columns (length, width, height):")
st.dataframe(df[["length", "width", "height"]].head())

# Binning
st.header("Step 5: Binning Horsepower")
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
group_names = ["Low", "Medium", "High"]
df["horsepower-binned"] = pd.cut(df["horsepower"], bins, labels=group_names, include_lowest=True)

st.write("Horsepower binned:")
st.bar_chart(df["horsepower-binned"].value_counts())

# Indicator variables
st.header("Step 6: Create Indicator Variables")
fuel_dummies = pd.get_dummies(df["fuel-type"], prefix="fuel-type")
aspiration_dummies = pd.get_dummies(df["aspiration"], prefix="aspiration")
df = pd.concat([df, fuel_dummies, aspiration_dummies], axis=1)
df.drop(["fuel-type", "aspiration"], axis=1, inplace=True)

st.write("Dataset with indicator variables:")
st.dataframe(df.head())

# Save cleaned dataset
st.download_button(
    label="Download Cleaned Dataset",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="cleaned_dataset.csv",
    mime="text/csv"
)

# Load cleaned dataset for analysis
st.header("Step 7: Analysis")
analysis_url = 'https://raw.githubusercontent.com/klamsal/Fall2024Exam/refs/heads/main/CleanedAutomobile.csv'

try:
    df_analysis = pd.read_csv(analysis_url)
except Exception as e:
    st.error(f"Error loading analysis dataset: {e}")
    st.stop()

# Correlation analysis
st.subheader("Correlation Analysis")
correlation_matrix = df_analysis[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr()
st.write("Correlation matrix:")
st.dataframe(correlation_matrix)

# Scatter plots
st.subheader("Scatter Plots")
try:
    fig1, ax1 = plt.subplots()
    sns.regplot(x="engine-size", y="price", data=df_analysis, ax=ax1)
    plt.title("Engine Size vs. Price")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.regplot(x="highway-mpg", y="price", data=df_analysis, ax=ax2)
    plt.title("Highway MPG vs. Price")
    st.pyplot(fig2)
except Exception as e:
    st.error(f"Error generating scatter plots: {e}")

# Box plot
st.subheader("Box Plot: Body Style vs. Price")
try:
    fig3, ax3 = plt.subplots()
    sns.boxplot(x="body-style", y="price", data=df_analysis, palette="Set2", ax=ax3)
    plt.xticks(rotation=45)
    st.pyplot(fig3)
except Exception as e:
    st.error(f"Error generating box plot: {e}")

# Heatmap
st.subheader("Heatmap: Drive Wheels and Body Style vs. Price")
try:
    grouped_data = df_analysis.groupby(['drive-wheels', 'body-style'])['price'].mean().reset_index()
    pivot_table = grouped_data.pivot(index='drive-wheels', columns='body-style', values='price').fillna(0)

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap='RdBu', linewidths=.5, ax=ax4)
    plt.title("Heatmap: Drive Wheels & Body Style vs. Price")
    st.pyplot(fig4)
except Exception as e:
    st.error(f"Error generating heatmap: {e}")

# Pearson correlation
st.subheader("Pearson Correlation")
try:
    pearson_coef, p_value = stats.pearsonr(df_analysis['engine-size'], df_analysis['price'])
    st.write(f"Pearson Correlation (Engine Size vs. Price): {pearson_coef:.2f}, P-value: {p_value:.2e}")
except Exception as e:
    st.error(f"Error calculating Pearson correlation: {e}")
