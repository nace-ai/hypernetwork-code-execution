import pandas as pd
from typing import Optional

def calculate_return_on_assets(
    csv_path: str,
    asset_value_column: str = "current_price",
    shares_column: str = "shares",
    cost_column: str = "purchase_price",
    filter_column: Optional[str] = None,
    filter_value: Optional[str] = None,
    min_threshold: float = 0.0
) -> float:
    """
    Calculate the return on assets (ROA) for an investment portfolio from CSV data.
    
    ROA is calculated as the total current value of the assets minus the total cost of the assets,
    divided by the total cost of the assets. This gives a measure of profitability relative to the 
    investment made.
    
    Args:
        csv_path: Path to the CSV file.
        asset_value_column: Column name for the current value of each asset.
        shares_column: Column name for the number of shares.
        cost_column: Column name for the original purchase price per share.
        filter_column: Optional column to filter the data by a specific category (e.g., sector).
        filter_value: Value to filter by within the filter_column.
        min_threshold: Minimum threshold for total cost to proceed with calculation, avoids division by zero.
        
    Returns:
        Return on assets as a float, or -1 in case of any error.
    """
    try:
        df = pd.read_csv(csv_path)
        
        if filter_column and filter_value:
            df = df[df[filter_column] == filter_value]
        
        df['total_asset_value'] = df[asset_value_column] * df[shares_column]
        df['total_cost'] = df[cost_column] * df[shares_column]
        
        total_asset_value = df['total_asset_value'].sum()
        total_cost = df['total_cost'].sum()

        if total_cost <= min_threshold:
            return -1.0  # Avoid division by zero or insignificant cost

        roa = (total_asset_value - total_cost) / total_cost
        return float(roa)
    except Exception:
        return -1.0