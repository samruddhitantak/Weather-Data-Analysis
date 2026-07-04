import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#load dataset
df = pd.read_csv("weather_data.csv")
print(df.head())
print(df.info())
print(df.describe())
#clean data
print(df.isnull().sum())
#fill missing values
df["Temperature_C"] = df["Temperature_C"].fillna(df["Temperature_C"].mean())
df["Humidity_pct"] = df["Humidity_pct"].fillna(df["Humidity_pct"].mean())
#remove duplicate rows
df = df.drop_duplicates()
# Convert Date_Time to datetime
df["Date_Time"] = pd.to_datetime(df["Date_Time"])
# Create Month and Year columns
df["Month"] = df["Date_Time"].dt.month_name()
df["Year"] = df["Date_Time"].dt.year
#EDA
print(df["Temperature_C"].mean())      #average temperature
print(df["Temperature_C"].max())       #highest temprature
print(df["Temperature_C"].min())         #lowest temperature
print(df["Humidity_pct"].mean())           #average humidity
print(df["Precipitation_mm"].sum())            #total precipitation
print(df["Wind_Speed_kmh"].describe())    #wind speed 
#visualization
#temprature trend
sample_df = df.sample(5000, random_state=42).sort_values("Date_Time")
plt.figure(figsize=(12,5))
plt.plot(sample_df["Date_Time"], sample_df["Temperature_C"])
plt.title("Temperature Trend")
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#humidity distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Humidity_pct"],bins=20)
plt.title("Humidity Distribution")
plt.show()
#Precipitation trend
sample_df = df.sample(5000, random_state=42).sort_values("Date_Time")
plt.figure(figsize=(12,5))
plt.plot(sample_df["Date_Time"], sample_df["Precipitation_mm"])
plt.title("Precipitation Trend")
plt.xlabel("Date")
plt.ylabel("Precipitation (mm)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#monthly average temperature
monthly_temp = (df.groupby("Month")["Temperature_C"].mean().reindex(["January","February","March","April","May","June","July","August","September","October","November","December"]))
plt.figure(figsize=(12,5))
monthly_temp.plot(kind="bar")
plt.title("Monthly Average Temperature")
plt.xlabel("Month")
plt.ylabel("Temperature (°C)")
plt.tight_layout()
plt.show()
#monthly average humidity
monthly_humidity = (df.groupby("Month")["Humidity_pct"].mean().reindex(["January","February","March","April","May","June","July","August","September","October","November","December"]))
plt.figure(figsize=(12,5))
monthly_humidity.plot(kind="bar")
plt.title("Monthly Average Humidity")
plt.xlabel("Month")
plt.ylabel("Humidity (%)")
plt.tight_layout()
plt.show()
#coorelationn heatmap
plt.figure(figsize=(8,6))
sns.heatmap(
    df[["Temperature_C","Humidity_pct","Precipitation_mm","Wind_Speed_kmh"]].corr(),annot=True,cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
#temperature boxplot
plt.figure(figsize=(6,6))
sns.boxplot(y=df["Temperature_C"])
plt.title("Temperature Boxplot")
plt.ylabel("Temperature (°C)")
plt.show()
#scatter plot
plt.figure(figsize=(8,6))
sns.scatterplot(x="Humidity_pct",y="Temperature_C",data=df.sample(5000, random_state=42))
plt.title("Temperature vs Humidity")
plt.xlabel("Humidity (%)")
plt.ylabel("Temperature (°C)")
plt.show()
#compare weather across month
monthly = (
    df.groupby("Month")[["Temperature_C","Humidity_pct","Precipitation_mm"]].mean().reindex(["January","February","March","April","May","June","July","August","September","October","November","December"]))
print(monthly)
#insights
print("\n--------- WEATHER INSIGHTS ---------\n")
print("Hottest Month:",monthly["Temperature_C"].idxmax())
print("Coldest Month:",monthly["Temperature_C"].idxmin())
print("Highest Rainfall:",monthly["Precipitation_mm"].idxmax())
print("Lowest Rainfall:",monthly["Precipitation_mm"].idxmin())
print("Highest Humidity:",monthly["Humidity_pct"].idxmax())
print("Lowest Humidity:",monthly["Humidity_pct"].idxmin())

df.to_csv("clean_weather.csv", index=False)
print("Clean dataset saved successfully.")