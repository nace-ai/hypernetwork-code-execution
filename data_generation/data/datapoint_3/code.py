import pandas as pd
from typing import Optional

def calculate_current_ratio(
    csv_path: str,
    assets_column: str = "total_current_assets",
    liabilities_column: str = "total_current_liabilities",
    filter_column: Optional[str] = None,
    filter_value: Optional[str] = None,
    min_threshold: Optional[float] = None
) -> float:
    """
    Calculate the current ratio from a CSV file containing financial data.
    
    Current Ratio = Total Current Assets / Total Current Liabilities
    
    Args:
        csv_path: Path to the CSV file containing financial data.
        assets_column: Column name for total current assets.
        liabilities_column: Column name for total current liabilities.
        filter_column: Optional column to filter data (e.g., by division).
        filter_value: Value to filter on if filter_column is specified.
        min_threshold: Optional minimum threshold for the current ratio.
        
    Returns:
        The calculated current ratio as a float, or a default value on error.
    """
    try:
        df = pd.read_csv(csv_path)

        # Validate that essential columns exist
        if assets_column not in df.columns or liabilities_column not in df.columns:
            print(f"Error: Specified columns '{assets_column}' or '{liabilities_column}' not found in CSV.")
            return -1.0

        # Apply filter if specified
        if filter_column and filter_value:
            if filter_column not in df.columns:
                print(f"Error: Filter column '{filter_column}' not found in CSV.")
                return -1.0
            df = df[df[filter_column] == filter_value]

        # Calculate the Current Ratio
        total_assets = df[assets_column].sum()
        total_liabilities = df[liabilities_column].sum()

        if total_liabilities == 0:
            print("Error: Total current liabilities are zero, undefined ratio.")
            return -1.0

        current_ratio = total_assets / total_liabilities

        # Check against minimum threshold if specified
        if min_threshold is not None and current_ratio < min_threshold:
            print(f"Current ratio {current_ratio} is below the minimum threshold {min_threshold}.")
            return min_threshold

        return float(current_ratio)
        
    except Exception as e:
        print(f"Exception occurred: {e}")
        return 0.0