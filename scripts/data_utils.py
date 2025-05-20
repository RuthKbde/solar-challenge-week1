"""
Utility functions for solar data analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_data(file_path):
    """
    Load solar data from CSV file
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Loaded and preprocessed data
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    return df


def detect_outliers(df, columns, z_threshold=3):
    """
    Detect outliers in specified columns using Z-score method
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    columns : list
        List of column names to check for outliers
    z_threshold : float
        Z-score threshold to consider a value as outlier
        
    Returns:
    --------
    dict
        Dictionary with outlier information for each column
    """
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


def clean_data(df, numeric_cols):
    """
    Clean data by handling outliers and missing values
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    numeric_cols : list
        List of numeric columns to clean
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned dataframe
    """
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
    
    # Add hour column for day/night analysis
    df_clean['hour'] = df_clean['Timestamp'].dt.hour
    
    # Define daytime: 6 AM to 6 PM
    df_clean['is_daytime'] = (df_clean['hour'] >= 6) & (df_clean['hour'] <= 18)
    
    # Handle negative solar radiation values
    for col in ['GHI', 'DNI', 'DHI']:
        # During nighttime, negative values are expected, set to 0
        df_clean.loc[~df_clean['is_daytime'] & (df_clean[col] < 0), col] = 0
        # During daytime, negative values are likely sensor errors, set to 0
        df_clean.loc[df_clean['is_daytime'] & (df_clean[col] < 0), col] = 0
    
    return df_clean


def create_correlation_heatmap(df, columns, title='Correlation Matrix', figsize=(12, 10)):
    """
    Create a correlation heatmap
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    columns : list
        List of columns to include in correlation
    title : str
        Plot title
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with the heatmap
    """
    corr_matrix = df[columns].corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
    plt.title(title, fontsize=16)
    plt.tight_layout()
    
    return plt.gcf()


def aggregate_by_time(df, time_period='day', metrics=['GHI', 'DNI', 'DHI', 'Tamb']):
    """
    Aggregate data by specified time period
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    time_period : str
        Time period for aggregation ('day', 'month')
    metrics : list
        List of metrics to aggregate
        
    Returns:
    --------
    pandas.DataFrame
        Aggregated dataframe
    """
    if time_period == 'day':
        return df.groupby(df['Timestamp'].dt.date).agg({
            metric: 'mean' for metric in metrics
        }).reset_index()
    
    elif time_period == 'month':
        df['month'] = df['Timestamp'].dt.month
        return df.groupby('month').agg({
            metric: 'mean' for metric in metrics
        }).reset_index()
    
    else:
        raise ValueError("time_period must be 'day' or 'month'")
