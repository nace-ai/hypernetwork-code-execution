import pandas as pd
from typing import Optional

def calculate_gross_profit_margin(
    csv_path: str,
    revenue_column: str,
    cost_column: str,
    filter_column: Optional[str] = None,
    filter_value: Optional[str] = None,
    min_threshold: Optional[float] = None,
    discount_rate: float = 0.0
) -> float:
    """
    Calculate the gross profit margin from an expense report CSV file.
    
    Args:
        csv_path: Path to the CSV file.
        revenue_column: Column name for total revenue values.
        cost_column: Column name for total cost or expense values.
        filter_column: Optional column name to filter the data by a specific value.
        filter_value: The value to filter the data on for the filter_column.
        min_threshold: Optional minimum threshold for the revenue to be considered.
        discount_rate: Discount rate to apply to costs for adjusted profit calculation.
        
    Returns:
        The gross profit margin as a float. Returns -1 in case of an error.
    """
    try:
        df = pd.read_csv(csv_path)

        if filter_column and filter_value:
            df = df[df[filter_column] == filter_value]

        if min_threshold is not None:
            df = df[df[revenue_column] >= min_threshold]

        total_revenue = df[revenue_column].sum()
        total_costs = df[cost_column].sum() * (1 - discount_rate)

        if total_revenue == 0:
            return 0.0

        gross_profit = total_revenue - total_costs
        gross_profit_margin = (gross_profit / total_revenue) * 100

        return float(gross_profit_margin)
    except Exception:
        return -1.0