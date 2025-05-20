#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Solar Data Cleaning and EDA for Togo
Week 0 Challenge - 10 Academy AIM : By Ruth.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set plot style
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# Create directory for plots if it doesn't exist
os.makedirs('plots/togo', exist_ok=True)
# Create directory for data if it doesn't exist
os.makedirs('data', exist_ok=True)

# 1. Load the data
print("Loading data...")
file_path = 'data/togo-dapaong_qc.csv'
df = pd.read_csv(file_path)

# First look at the data
print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset shape:", df.shape)

# 2. Summary Statistics & Missing-Value Report
print("\n--- Summary Statistics & Missing-Value Report ---")
# Convert Timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
print("\nDataset time range:", df['Timestamp'].min(), "to", df['Timestamp'].max())

# Summary statistics
print("\nSummary statistics:")
summary_stats = df.describe()
print(summary_stats)

# Missing values analysis
print("\nMissing values analysis:")
missing_values = df.isna().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percentage
})
print(missing_df[missing_df['Percentage'] > 0])

columns_high_missing = missing_df[missing_df['Percentage'] > 5].index.tolist()
if columns_high_missing:
    print(f"\nColumns with >5% missing values: {columns_high_missing}")
else:
    print("\nNo columns with >5% missing values.")

# 3. Outlier Detection & Basic Cleaning
print("\n--- Outlier Detection & Basic Cleaning ---")

# Function to calculate z-scores and identify outliers
def detect_outliers(df, columns, z_threshold=3):
    outliers_info = {}
    for col in columns:
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        outliers_mask = z_scores > z_threshold
        outliers_count = np.sum(outliers_mask)
        outliers_percentage = (outliers_count / len(z_scores)) * 100
        outliers_info[col] = {
            'count': outliers_count,
            'percentage': outliers_percentage
        }
    return outliers_info

# Numeric columns to check for outliers
numeric_cols = ['GHI', 'DNI', 'DHI', 'ModA', 'ModB', 'WS', 'WSgust']

# Check for outliers
print("\nOutlier detection (|Z-score| > 3):")
outliers_info = detect_outliers(df, numeric_cols)
for col, info in outliers_info.items():
    print(f"{col}: {info['count']} outliers ({info['percentage']:.2f}%)")

# Create a copy for cleaning
df_clean = df.copy()

# Replace outliers with NaN
for col in numeric_cols:
    z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
    outliers_mask = z_scores > 3
    df_clean.loc[df_clean[col].dropna().index[outliers_mask], col] = np.nan

# Impute missing values with median for key columns
for col in numeric_cols:
    median_value = df_clean[col].median()
    df_clean[col].fillna(median_value, inplace=True)

# Check for negative GHI, DNI, DHI values (these should typically be positive during daytime)
print("\nNegative values in solar radiation measurements:")
for col in ['GHI', 'DNI', 'DHI']:
    neg_count = (df_clean[col] < 0).sum()
    neg_percentage = (neg_count / len(df_clean)) * 100
    print(f"{col}: {neg_count} negative values ({neg_percentage:.2f}%)")

# Set negative solar radiation values to 0 during nighttime (this is a common approach)
# Define daytime: 6 AM to 6 PM
df_clean['hour'] = df_clean['Timestamp'].dt.hour
daytime_mask = (df_clean['hour'] >= 6) & (df_clean['hour'] <= 18)
nighttime_mask = ~daytime_mask

# Set negative values during nighttime to 0
for col in ['GHI', 'DNI', 'DHI']:
    # Only during nighttime, negative values are expected
    df_clean.loc[nighttime_mask & (df_clean[col] < 0), col] = 0
    # During daytime, negative values could be sensor errors, replace with 0
    df_clean.loc[daytime_mask & (df_clean[col] < 0), col] = 0

# Save cleaned data
print("\nSaving cleaned data...")
df_clean.to_csv('data/togo_clean.csv', index=False)
print("Cleaned data saved to data/togo_clean.csv")

# 4. Time Series Analysis
print("\n--- Time Series Analysis ---")

# Aggregate data by day to make visualizations more manageable
df_daily = df_clean.groupby(df_clean['Timestamp'].dt.date).agg({
    'GHI': 'mean',
    'DNI': 'mean',
    'DHI': 'mean',
    'Tamb': 'mean'
}).reset_index()
df_daily['Timestamp'] = pd.to_datetime(df_daily['Timestamp'])

# Plot daily averages
plt.figure(figsize=(14, 7))
plt.plot(df_daily['Timestamp'], df_daily['GHI'], label='GHI', color='red')
plt.plot(df_daily['Timestamp'], df_daily['DNI'], label='DNI', color='blue')
plt.plot(df_daily['Timestamp'], df_daily['DHI'], label='DHI', color='green')
plt.title('Daily Average Solar Radiation in Togo (Dapaong)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Radiation (W/m²)', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('plots/togo/daily_solar_radiation.png')

# Plot temperature
plt.figure(figsize=(14, 7))
plt.plot(df_daily['Timestamp'], df_daily['Tamb'], label='Ambient Temperature', color='orange')
plt.title('Daily Average Temperature in Togo (Dapaong)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Temperature (°C)', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('plots/togo/daily_temperature.png')

# Monthly aggregation
df_clean['month'] = df_clean['Timestamp'].dt.month
df_monthly = df_clean.groupby('month').agg({
    'GHI': 'mean',
    'DNI': 'mean',
    'DHI': 'mean',
    'Tamb': 'mean'
}).reset_index()

# Plot monthly averages
plt.figure(figsize=(14, 7))
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
x = np.arange(len(months))
width = 0.2

plt.bar(x - width, df_monthly['GHI'], width, label='GHI', color='red')
plt.bar(x, df_monthly['DNI'], width, label='DNI', color='blue')
plt.bar(x + width, df_monthly['DHI'], width, label='DHI', color='green')

plt.title('Monthly Average Solar Radiation in Togo (Dapaong)', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Radiation (W/m²)', fontsize=14)
plt.xticks(x, months)
plt.legend()
plt.tight_layout()
plt.savefig('plots/togo/monthly_solar_radiation.png')

# 5. Cleaning Impact Analysis
print("\n--- Cleaning Impact Analysis ---")

# Group by Cleaning flag
cleaning_impact = df_clean.groupby('Cleaning').agg({
    'ModA': 'mean',
    'ModB': 'mean'
}).reset_index()

print("\nAverage ModA and ModB values by Cleaning status:")
print(cleaning_impact)

# Plot cleaning impact
plt.figure(figsize=(10, 6))
x = ['No Cleaning (0)', 'Cleaning (1)']
x_pos = np.arange(len(x))
width = 0.35

# Check if we have both cleaning states (0 and 1)
if len(cleaning_impact) >= 2:
    plt.bar(x_pos - width/2, cleaning_impact['ModA'], width, label='ModA')
    plt.bar(x_pos + width/2, cleaning_impact['ModB'], width, label='ModB')
    plt.title('Impact of Cleaning on Module Measurements', fontsize=16)
    plt.xlabel('Cleaning Status', fontsize=14)
    plt.ylabel('Average Measurement (W/m²)', fontsize=14)
    plt.xticks(x_pos, x)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/togo/cleaning_impact.png')
else:
    print("Not enough data to plot cleaning impact (need both cleaning states 0 and 1)")

# 6. Correlation & Relationship Analysis
print("\n--- Correlation & Relationship Analysis ---")

# Calculate correlations
corr_columns = ['GHI', 'DNI', 'DHI', 'TModA', 'TModB', 'Tamb', 'RH', 'WS', 'BP']
corr_matrix = df_clean[corr_columns].corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('Correlation Matrix of Key Variables', fontsize=16)
plt.tight_layout()
plt.savefig('plots/togo/correlation_heatmap.png')

# Scatter plots
# WS vs GHI
plt.figure(figsize=(10, 6))
plt.scatter(df_clean['WS'], df_clean['GHI'], alpha=0.5)
plt.title('Wind Speed vs. Global Horizontal Irradiance', fontsize=16)
plt.xlabel('Wind Speed (m/s)', fontsize=14)
plt.ylabel('GHI (W/m²)', fontsize=14)
plt.tight_layout()
plt.savefig('plots/togo/ws_vs_ghi.png')

# RH vs Tamb
plt.figure(figsize=(10, 6))
plt.scatter(df_clean['RH'], df_clean['Tamb'], alpha=0.5)
plt.title('Relative Humidity vs. Ambient Temperature', fontsize=16)
plt.xlabel('Relative Humidity (%)', fontsize=14)
plt.ylabel('Ambient Temperature (°C)', fontsize=14)
plt.tight_layout()
plt.savefig('plots/togo/rh_vs_tamb.png')

# 7. Wind & Distribution Analysis
print("\n--- Wind & Distribution Analysis ---")

# Histogram for GHI
plt.figure(figsize=(10, 6))
plt.hist(df_clean['GHI'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Distribution of Global Horizontal Irradiance', fontsize=16)
plt.xlabel('GHI (W/m²)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.tight_layout()
plt.savefig('plots/togo/ghi_histogram.png')

# Histogram for WS
plt.figure(figsize=(10, 6))
plt.hist(df_clean['WS'], bins=50, alpha=0.7, color='green', edgecolor='black')
plt.title('Distribution of Wind Speed', fontsize=16)
plt.xlabel('Wind Speed (m/s)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.tight_layout()
plt.savefig('plots/togo/ws_histogram.png')

# Wind Rose Plot (simplified version with histogram by direction)
plt.figure(figsize=(10, 10))
# Bin wind directions into 16 sectors
bins = np.arange(0, 361, 22.5)
df_clean['WD_bin'] = pd.cut(df_clean['WD'], bins=bins, right=False, include_lowest=True)
wind_counts = df_clean.groupby('WD_bin').size()

# Plot
ax = plt.subplot(111, polar=True)
theta = np.linspace(0, 2*np.pi, len(wind_counts), endpoint=False)
radii = wind_counts.values
width = 2*np.pi / len(wind_counts)
bars = ax.bar(theta, radii, width=width, bottom=0.0)

# Set the direction labels
direction_labels = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
                    'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)  # Clockwise
ax.set_thetagrids(np.degrees(theta), direction_labels)

plt.title('Wind Direction Distribution', fontsize=16)
plt.tight_layout()
plt.savefig('plots/togo/wind_rose.png')

# 8. Temperature Analysis
print("\n--- Temperature Analysis ---")

# Create a figure with RH vs Temperature colored by GHI
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df_clean['RH'], df_clean['Tamb'], 
                     c=df_clean['GHI'], cmap='viridis', 
                     alpha=0.5, edgecolors='none')
plt.colorbar(scatter, label='GHI (W/m²)')
plt.title('Relationship between Relative Humidity, Temperature, and Solar Radiation', fontsize=16)
plt.xlabel('Relative Humidity (%)', fontsize=14)
plt.ylabel('Ambient Temperature (°C)', fontsize=14)
plt.tight_layout()
plt.savefig('plots/togo/rh_tamb_ghi.png')

# 9. Bubble Chart
print("\n--- Bubble Chart Analysis ---")

# Sample the data to make the plot more readable (every 100th point)
sample_df = df_clean.iloc[::100].copy()

plt.figure(figsize=(12, 8))
bubble = plt.scatter(sample_df['Tamb'], sample_df['GHI'], 
                    s=sample_df['RH'], alpha=0.5, 
                    c=sample_df['RH'], cmap='YlOrRd')
plt.colorbar(bubble, label='Relative Humidity (%)')
plt.title('GHI vs. Temperature with Bubble Size Representing Humidity', fontsize=16)
plt.xlabel('Ambient Temperature (°C)', fontsize=14)
plt.ylabel('GHI (W/m²)', fontsize=14)
plt.tight_layout()
plt.savefig('plots/togo/tamb_ghi_rh_bubble.png')

print("\nAnalysis complete. All plots saved in the 'plots/togo' directory.")
print("Cleaned data saved to 'data/togo_clean.csv'.")

if __name__ == "__main__":
    print("Script execution completed.")
