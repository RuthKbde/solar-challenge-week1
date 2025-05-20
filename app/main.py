"""
Streamlit dashboard for Solar Data Analysis
Week 0 Challenge - 10 Academy AIM
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (
    load_data, get_combined_data, plot_boxplot, plot_summary_metrics,
    plot_daily_trend, plot_monthly_comparison, create_correlation_heatmap,
    generate_summary_table
)

# Set page configuration
st.set_page_config(
    page_title="Solar Farm Analysis Dashboard",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App title and description
st.title("Solar Farm Analysis Dashboard")
st.markdown("""
    This dashboard provides interactive visualizations of solar radiation data from three African countries:
    Benin, Sierra Leone, and Togo. Explore the data to identify high-potential regions for solar installations.
""")

# Load data
data_dict = load_data()

if data_dict is None:
    st.error("Could not load data. Please ensure the cleaned CSV files are in the data/ directory.")
    st.stop()

# Create a combined dataset for cross-country analysis
combined_data = get_combined_data(data_dict)

# Sidebar
st.sidebar.title("Controls")

# Country selector
selected_country = st.sidebar.selectbox(
    "Select a country for individual analysis:",
    ["Benin", "Togo", "Sierra Leone"]
)

# Metric selector
selected_metric = st.sidebar.selectbox(
    "Select a metric to analyze:",
    ["GHI", "DNI", "DHI"]
)

# Time range
st.sidebar.markdown("### Time Period")
time_period = st.sidebar.radio(
    "Select time aggregation:",
    ["Daily", "Monthly"]
)

# Visualization Options
st.sidebar.markdown("### Visualization Options")
show_boxplots = st.sidebar.checkbox("Show Boxplots", value=True)
show_summary = st.sidebar.checkbox("Show Summary Metrics", value=True)
show_correlations = st.sidebar.checkbox("Show Correlation Analysis", value=False)

# Main content
st.markdown("## Cross-Country Comparison")

# Show summary table
st.markdown("### Summary Statistics by Country")
summary_df = generate_summary_table(combined_data)
st.dataframe(summary_df, use_container_width=True)

# Summary metrics visualization
if show_summary:
    st.markdown("### Average Solar Radiation Metrics")
    summary_fig = plot_summary_metrics(combined_data)
    st.plotly_chart(summary_fig, use_container_width=True)

# Boxplot comparison
if show_boxplots:
    st.markdown(f"### Boxplot Comparison for {selected_metric}")
    boxplot_fig = plot_boxplot(combined_data, selected_metric)
    st.plotly_chart(boxplot_fig, use_container_width=True)

# Country-specific analysis
st.markdown(f"## {selected_country} Analysis")

# Time series visualization
st.markdown(f"### {time_period} Trends")
if time_period == "Daily":
    trend_fig = plot_daily_trend(data_dict, selected_country, selected_metric)
    st.plotly_chart(trend_fig, use_container_width=True)
else:
    st.markdown("### Monthly Comparison Across Countries")
    monthly_fig = plot_monthly_comparison(data_dict, selected_metric)
    st.plotly_chart(monthly_fig, use_container_width=True)

# Correlation analysis
if show_correlations:
    st.markdown(f"### Correlation Analysis for {selected_country}")
    corr_fig = create_correlation_heatmap(data_dict, selected_country)
    st.plotly_chart(corr_fig, use_container_width=True)

# Key findings
st.markdown("## Key Findings")
st.markdown("""
    Based on our analysis, we've identified the following key insights:

    1. **Benin** shows the highest average Global Horizontal Irradiance (GHI), making it potentially the most suitable location for photovoltaic solar installations.

    2. **Sierra Leone** demonstrates the least variability in GHI (lowest standard deviation), suggesting more consistent solar radiation throughout the year, which is advantageous for solar energy production reliability.

    3. **Benin** has the highest Direct Normal Irradiance (DNI), which is particularly important for concentrated solar power technologies.

    Note: Statistical testing (Kruskal-Wallis) indicates significant differences between countries in solar radiation metrics.
""")

# Footer
st.markdown("---")
st.markdown("Solar Farm Analysis Dashboard | Week 0 Challenge - 10 Academy AIM | 2025")
