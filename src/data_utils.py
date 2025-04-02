# src/data_utils.py

import pandas as pd

def standardize_dates(df, date_columns):
    """
    Convert string date columns to datetime objects.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing date columns to convert
    date_columns : list
        List of column names containing dates to convert
        
    Returns:
    --------
    pandas.DataFrame
        Copy of input DataFrame with new standardized date columns
    """
    df_copy = df.copy()
    for col in date_columns:
        if col in df.columns:
            try:
                # Try standard conversion first
                df_copy[f'{col}_date'] = pd.to_datetime(df_copy[col])
            except:
                # Fall back to specific format if standard fails
                try:
                    # Format for 'YYMMDD' values
                    df_copy[f'{col}_date'] = pd.to_datetime(df_copy[col], format='%y%m%d')
                except:
                    print(f"Could not convert column {col}")
    return df_copy

def extract_birth_info(birth_number):
    """
    Extract demographic information from Czech birth numbers.
    Handles potentially invalid day values.
    
    Parameters:
    -----------
    birth_number : int or str
        Czech birth number
        
    Returns:
    --------
    dict
        Dictionary containing birth_year, birth_month, birth_day,
        gender, age, and data_quality flag
    """
    birth_str = str(birth_number)
    
    # Extract components
    year_code = int(birth_str[:2])
    month_code = int(birth_str[2:4])
    day_code = int(birth_str[4:6])
    
    # Determine gender and adjust month
    if month_code > 50:
        gender = 'Female'
        birth_month = month_code - 50
    else:
        gender = 'Male'
        birth_month = month_code
    
    # Handle century
    if year_code >= 54:  # Assumption based on dataset being from the 1990s
        birth_year = 1900 + year_code
    else:
        birth_year = 2000 + year_code
    
    # Data quality check for day
    data_quality = "valid"
    normalized_day = day_code
    
    # Get maximum days for the month (simplified)
    days_in_month = 31  # Default for months with 31 days
    if birth_month in [4, 6, 9, 11]:
        days_in_month = 30
    elif birth_month == 2:
        # Simple leap year check (not perfect but good enough for this application)
        is_leap_year = (birth_year % 4 == 0 and birth_year % 100 != 0) or (birth_year % 400 == 0)
        days_in_month = 29 if is_leap_year else 28
    
    # Check if day is valid
    if day_code > days_in_month:
        data_quality = "questionable_day"
        normalized_day = min(day_code, days_in_month)  # Cap at maximum days for visual display
        
    return {
        'birth_year': birth_year,
        'birth_month': birth_month,
        'birth_day': day_code,           # Original day code
        'normalized_day': normalized_day, # Day capped at month maximum
        'gender': gender,
        'age': 2025 - birth_year,        # Current year
        'data_quality': data_quality      # Flag for data quality issues
    }