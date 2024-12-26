import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Title of the app
st.title("Automobile Price Prediction Analysis")

# Load Data
path = 'https://raw.githubusercontent.com/UtsavKhadka-Analyst/Automobile-Analysis-Streamlit/refs/heads/main/Automobile_clean_df.csv'
df = pd.read_csv(path)

# Show data preview
st.header("Dataset Preview")
st.write(df.head())

# Show column data types
st.subheader("Data Types of Columns")
st.write(df.dtypes)

# Question 1: Data type of "peak-rpm"
st.subheader("Question 1: Data Type of 'peak-rpm'")
st.write(f"The data type of the 'peak-rpm' column is: {df['peak-rpm'].dtypes}")

# Question 2: Correlation between selected columns
st.subheader("Question 2: Correlation between Bore, Stroke, Compression-Ratio, and Horsepower")
correlation = df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr()
st.write(correlation)

# Continuous Numerical Variables - Engine size vs Price
st.subheader("Engine Size vs Price")
sns.regplot(x="engine-size", y="price", data=df)
plt.xlabel('Engine size')
plt.ylabel('Price of car')
plt.ylim(0,)
st.pyplot()

# Highway mpg vs Price
st.subheader("Highway MPG vs Price")
sns.regplot(x="highway-mpg", y="price", data=df)
plt.xlabel("Highway MPG")
plt.ylabel("Price")
plt.title("Relationship between Highway MPG and Price")
st.pyplot()

# Weak Relationship - Peak RPM vs Price
st.subheader("Peak RPM vs Price")
sns.regplot(x="peak-rpm", y="price", data=df)
plt.xlabel("Peak RPM")
plt.ylabel("Price")
plt.title("Relationship between Peak RPM and Price")
st.pyplot()

# Question 3a: Correlation between stroke and price
st.subheader("Question 3a: Correlation between Stroke and Price")
correlation_stroke_price = df[['stroke', 'price']].corr()
st.write(correlation_stroke_price)

# Question 3b: Regression plot for stroke vs price
st.subheader("Question 3b: Regression Plot for Stroke vs Price")
sns.regplot(x="stroke", y="price", data=df)
st.pyplot()

# Categorical Variables - Boxplot for Body Style vs Price
st.subheader("Body Style vs Price")
sns.boxplot(x="body-style", y="price", data=df, palette="Set2")
plt.xlabel("Body Style")
plt.ylabel("Price")
plt.title("Price Distribution by Body Style")
plt.xticks(rotation=45)
st.pyplot()

# Boxplot for Engine Location vs Price
st.subheader("Engine Location vs Price")
sns.boxplot(x="engine-location", y="price", data=df, palette="Set3")
plt.xlabel("Engine Location")
plt.ylabel("Price")
plt.title("Price Distribution by Engine Location")
st.pyplot()

# Boxplot for Drive Wheels vs Price
st.subheader("Drive Wheels vs Price")
sns.boxplot(x="drive-wheels", y="price", data=df, palette="coolwarm")
plt.xlabel("Drive Wheels")
plt.ylabel("Price")
plt.title("Price Distribution by Drive Wheels")
st.pyplot()

# Descriptive Statistics for Continuous Variables
st.subheader("Descriptive Statistics for Continuous Variables")
st.write(df.describe())

# Descriptive Statistics for Categorical Variables
st.subheader("Descriptive Statistics for Categorical Variables")
st.write(df.describe(include=['object']))

# Value Counts for "Drive Wheels"
st.subheader("Value Counts for 'Drive Wheels'")
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts.index.name = 'drive-wheels'
st.write(drive_wheels_counts)

# Value Counts for "Engine Location"
st.subheader("Value Counts for 'Engine Location'")
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
st.write(engine_loc_counts)

# Grouping by 'Drive Wheels' and 'Body Style' and calculating average price
st.subheader("Average Price by Drive Wheels and Body Style")
df_group_one = df[['drive-wheels', 'body-style', 'price']]
df_group_one = df_group_one.groupby(['drive-wheels', 'body-style'], as_index=False).mean()
st.write(df_group_one)

# Pivot table for grouping by 'Drive Wheels' and 'Body Style'
st.subheader("Pivot Table of Average Price by Drive Wheels and Body Style")
pivot_table = df_group_one.pivot(index='drive-wheels', columns='body-style', values='price')
st.write(pivot_table)
