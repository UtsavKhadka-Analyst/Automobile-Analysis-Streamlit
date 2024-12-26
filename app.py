import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to load the dataset
def load_data():
    filename = "https://raw.githubusercontent.com/UtsavKhadka-Analyst/Automobile-Analysis-Streamlit/refs/heads/main/imports_85.csv"
    headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style",
               "drive-wheels", "engine-location", "wheel-base", "length", "width", "height", "curb-weight", "engine-type",
               "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower",
               "peak-rpm", "city-mpg", "highway-mpg", "price"]
    df = pd.read_csv(filename, names=headers)
    df.replace("?", np.nan, inplace=True)
    return df

# Load data
df = load_data()

# Streamlit App UI
st.title('Car Data Cleaning and Transformation')

# Show first 5 rows of data
st.subheader('Original Data')
st.dataframe(df.head())

# Missing values
st.subheader('Missing Values')
missing_data = df.isnull()
st.write(missing_data.head())

# Handle missing data - replacing with mean
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)

avg_bore = df['bore'].astype('float').mean(axis=0)
df["bore"].replace(np.nan, avg_bore, inplace=True)

avg_stroke = df['stroke'].astype('float').mean(axis=0)
df["stroke"].replace(np.nan, avg_stroke, inplace=True)

avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)

avg_peakrpm = df['peak-rpm'].astype('float').mean(axis=0)
df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)

df['num-of-doors'].replace(np.nan, "four", inplace=True)

# Drop rows without price data
df.dropna(subset=["price"], axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)

# Correct data formats
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

# Convert mpg to L/100km
df['city-L/100km'] = 235 / df["city-mpg"]
df['highway-L/100km'] = 235 / df["highway-mpg"]

# Show transformed data
st.subheader('Transformed Data')
st.dataframe(df.head())

# Plot data
st.subheader('Data Visualization: Price Distribution')
plt.figure(figsize=(10, 6))
plt.hist(df['price'], bins=20, color='skyblue', edgecolor='black')
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')
st.pyplot()

