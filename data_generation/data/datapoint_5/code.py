import pandas as pd
from typing import Optional

def calculate_total_liabilities(
    csv_path: str,
    quantity_column: str = "quantity_on_hand",
    cost_column: str = "unit_cost",
    filter_column: Optional[str] = None,
    filter_value: Optional[str] = None,
    min_threshold: Optional[int] = None,
    warehouse_column: Optional[str] = "warehouse",
    group_by_column: Optional[str] = None,
    discount_rate: float = 0.0
) -> float:
    """
    Calculate the total liabilities from inventory stock data in a CSV file.
    
    The total liabilities are computed as the sum of the product of the quantity and cost,
    potentially filtered and grouped by specified columns, and adjusted by a discount rate.

    Args:
        csv_path: Path to the CSV file.
        quantity_column: Name of the column specifying the quantities.
        cost_column: Name of the column specifying the unit costs.
        filter_column: Optional column to filter by.
        filter_value: Value to filter on if filter_column is provided.
        min_threshold: Minimum threshold for quantity to consider in calculations.
        warehouse_column: Name of the column specifying the warehouse (for context).
        group_by_column: Optional column to group by before calculations.
        discount_rate: Discount rate to apply to the total liabilities.

    Returns:
        Total liabilities as a float, or -1.0 if an error occurs.
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Validate columns
        if quantity_column not in df.columns or cost_column not in df.columns:
            return -1.0
        
        # Apply filter if specified
        if filter_column and filter_value:
            df = df[df[filter_column] == filter_value]

        # Apply minimum threshold filter
        if min_threshold is not None:
            df = df[df[quantity_column] >= min_threshold]

        # Calculate liabilities
        df['liabilities'] = df[quantity_column] * df[cost_column]

        if group_by_column and group_by_column in df.columns:
            df = df.groupby(group_by_column)['liabilities'].sum().reset_index()

        total_liabilities = df['liabilities'].sum()

        # Apply discount rate
        total_liabilities *= (1 - discount_rate)

        return float(total_liabilities)
    except Exception:
        return -1.0