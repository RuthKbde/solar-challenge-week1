# Notebooks

This directory contains Jupyter notebooks for the Solar Data Analysis project.

## Contents

- **benin_eda.ipynb**: Exploratory data analysis for Benin solar data
- **togo_eda.ipynb**: Exploratory data analysis for Togo solar data
- **sierraleone_eda.ipynb**: Exploratory data analysis for Sierra Leone solar data
- **compare_countries.ipynb**: Cross-country comparison of solar metrics

## Running the Notebooks

These notebooks require the cleaned data files to be present in the `data/` directory. To run the notebooks:

1. Ensure you have installed all requirements from the main project's `requirements.txt`
2. Place the raw CSV data files in the `data/` directory
3. Run the notebooks in the following order:
   - First, run the individual country EDA notebooks to generate the cleaned data files
   - Then, run the cross-country comparison notebook to analyze the differences

## Notebook Details

### Country EDA Notebooks

Each country's EDA notebook follows a similar structure:

1. Data loading and preprocessing
2. Summary statistics and missing value report
3. Outlier detection and cleaning
4. Time series analysis of solar radiation and temperature
5. Analysis of cleaning impact on module measurements
6. Correlation and relationship analysis
7. Wind and distribution analysis
8. Temperature analysis
9. Bubble chart visualization

### Cross-Country Comparison

The comparison notebook examines the solar potential across all three countries:

1. Compares key metrics (GHI, DNI, DHI) using boxplots
2. Provides a summary table with mean, median, and standard deviation
3. Performs statistical testing to determine significance of differences
4. Identifies and ranks countries by solar potential
5. Provides key observations and recommendations
