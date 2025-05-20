import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_data():
    """
    Load cleaned data for all three countries
    """
    # Try different possible paths
    possible_paths = [
        # Direct path
        'data/',
        # From root directory
        '../data/',
        # Absolute path
        'C:/Users/HDesalegn/Ruth/data/'
    ]
    
    for base_path in possible_paths:
        try:
            benin_df = pd.read_csv(f'{base_path}benin_clean.csv')
            togo_df = pd.read_csv(f'{base_path}togo_clean.csv')
            sierraleone_df = pd.read_csv(f'{base_path}sierraleone_clean.csv')
            
            # Add country column
            benin_df['country'] = 'Benin'
            togo_df['country'] = 'Togo'
            sierraleone_df['country'] = 'Sierra Leone'
            
            # Convert timestamp to datetime
            for df in [benin_df, togo_df, sierraleone_df]:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            
            return {
                'Benin': benin_df,
                'Togo': togo_df,
                'Sierra Leone': sierraleone_df,
            }
        except FileNotFoundError:
            # Try the next path
            continue
    
    # If we've tried all paths and none worked
    return None


def get_combined_data(data_dict):
    """
    Combine data from all countries
    """
    # Sample data for better performance (1% of each dataset)
    samples = []
    for country, df in data_dict.items():
        sample = df.sample(frac=0.01, random_state=42)
        samples.append(sample)
    
    return pd.concat(samples)


def plot_boxplot(data, metric):
    """
    Create a boxplot for the specified metric across countries
    """
    fig = px.box(data, x='country', y=metric, 
                 title=f'Comparison of {metric} Across Countries',
                 color='country')
    fig.update_layout(
        xaxis_title='Country',
        yaxis_title=f'{metric} (W/m²)',
        height=500,
    )
    return fig


def plot_summary_metrics(data):
    """
    Plot average GHI, DNI, DHI for each country as a bar chart
    """
    metrics = ['GHI', 'DNI', 'DHI']
    countries = data['country'].unique()
    
    # Calculate mean values
    summary_data = []
    for country in countries:
        country_data = data[data['country'] == country]
        for metric in metrics:
            summary_data.append({
                'country': country,
                'metric': metric,
                'value': country_data[metric].mean()
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    fig = px.bar(summary_df, x='country', y='value', color='metric',
                 barmode='group', title='Average Solar Radiation Metrics by Country',
                 labels={'value': 'Average Value (W/m²)', 'country': 'Country'})
    
    return fig


def plot_daily_trend(data_dict, country, metric):
    """
    Plot daily average trend for a selected country and metric
    """
    df = data_dict[country]
    
    # Group by day
    df_daily = df.groupby(df['Timestamp'].dt.date).agg({
        metric: 'mean'
    }).reset_index()
    df_daily['Timestamp'] = pd.to_datetime(df_daily['Timestamp'])
    
    fig = px.line(df_daily, x='Timestamp', y=metric,
                 title=f'Daily Average {metric} in {country}',
                 labels={metric: f'{metric} (W/m²)', 'Timestamp': 'Date'})
    
    return fig


def plot_monthly_comparison(data_dict, metric):
    """
    Plot monthly average for selected metric across all countries
    """
    monthly_data = []
    
    for country, df in data_dict.items():
        # Add month column
        df['month'] = df['Timestamp'].dt.month
        
        # Group by month
        df_monthly = df.groupby('month').agg({
            metric: 'mean'
        }).reset_index()
        
        df_monthly['country'] = country
        monthly_data.append(df_monthly)
    
    combined_monthly = pd.concat(monthly_data)
    
    # Map month numbers to names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    combined_monthly['month_name'] = combined_monthly['month'].apply(lambda x: month_names[x-1])
    
    fig = px.line(combined_monthly, x='month', y=metric, color='country',
                 title=f'Monthly Average {metric} Across Countries',
                 labels={metric: f'{metric} (W/m²)', 'month': 'Month'},
                 range_x=[1, 12])
    
    # Update x-axis ticks to show month names
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = list(range(1, 13)),
            ticktext = month_names
        )
    )
    
    return fig


def create_correlation_heatmap(data_dict, country):
    """
    Create a correlation heatmap for selected country
    """
    df = data_dict[country]
    
    # Select columns for correlation analysis
    corr_columns = ['GHI', 'DNI', 'DHI', 'TModA', 'TModB', 'Tamb', 'RH', 'WS', 'BP']
    corr_matrix = df[corr_columns].corr().round(2)
    
    fig = px.imshow(corr_matrix, 
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    title=f'Correlation Matrix for {country}')
    
    return fig


def generate_summary_table(data):
    """
    Generate a summary table of metrics by country
    """
    metrics = ['GHI', 'DNI', 'DHI']
    countries = data['country'].unique()
    
    # Calculate statistics
    summary_data = {}
    for country in countries:
        country_df = data[data['country'] == country]
        country_summary = {}
        
        for metric in metrics:
            country_summary[f'{metric}_mean'] = country_df[metric].mean()
            country_summary[f'{metric}_median'] = country_df[metric].median()
            country_summary[f'{metric}_std'] = country_df[metric].std()
        
        summary_data[country] = country_summary
    
    summary_df = pd.DataFrame(summary_data).T
    
    # Format numbers to 2 decimal places
    for col in summary_df.columns:
        summary_df[col] = summary_df[col].round(2)
    
    return summary_df