"""
Test suite for utility functions in scripts package
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_utils import detect_outliers, clean_data


class TestDataUtils(unittest.TestCase):
    """Test class for data_utils.py"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample dataframe
        self.df = pd.DataFrame({
            'Timestamp': pd.date_range(start='2023-01-01', periods=100, freq='H'),
            'GHI': np.random.normal(500, 100, 100),
            'DNI': np.random.normal(700, 150, 100),
            'DHI': np.random.normal(200, 50, 100),
            'Tamb': np.random.normal(25, 5, 100),
            'RH': np.random.normal(60, 10, 100),
        })
        
        # Add some outliers
        self.df.loc[10, 'GHI'] = 2000  # High outlier
        self.df.loc[20, 'DNI'] = -500  # Negative outlier
        self.df.loc[30, 'DHI'] = 1500  # High outlier
        
        # Add some missing values
        self.df.loc[40, 'GHI'] = np.nan
        self.df.loc[50, 'DNI'] = np.nan
        
        self.numeric_cols = ['GHI', 'DNI', 'DHI', 'Tamb', 'RH']
    
    def test_detect_outliers(self):
        """Test outlier detection function"""
        outliers_info = detect_outliers(self.df, self.numeric_cols)
        
        # Check that the function detected at least the outliers we added
        self.assertGreaterEqual(outliers_info['GHI']['count'], 1)
        self.assertGreaterEqual(outliers_info['DNI']['count'], 1)
        self.assertGreaterEqual(outliers_info['DHI']['count'], 1)
    
    def test_clean_data(self):
        """Test data cleaning function"""
        df_clean = clean_data(self.df, self.numeric_cols)
        
        # Check that clean data has the expected columns
        self.assertIn('hour', df_clean.columns)
        self.assertIn('is_daytime', df_clean.columns)
        
        # Check that clean data has no negative values in solar columns during daytime
        daytime_mask = df_clean['is_daytime']
        self.assertEqual(sum(df_clean.loc[daytime_mask, 'GHI'] < 0), 0)
        self.assertEqual(sum(df_clean.loc[daytime_mask, 'DNI'] < 0), 0)
        self.assertEqual(sum(df_clean.loc[daytime_mask, 'DHI'] < 0), 0)
        
        # Check that clean data has no missing values in numeric columns
        for col in self.numeric_cols:
            self.assertEqual(df_clean[col].isna().sum(), 0)


if __name__ == '__main__':
    unittest.main()
