import pandas as pd
from typing import Optional

def calculate_net_profit(
    csv_path: str,
    revenue_column: str = "quantity_on_hand",
    cost_column: str = "unit_cost",
    filter_column: Optional[str] = None,
    filter_value: Optional[str] = None,
    discount_rate: float = 0.0
) -> float:
    """
    Calculate the net profit from inventory stock data in a CSV file.
    
    Args:
        csv_path: Path to the CSV file.
        revenue_column: Column representing the revenue-generating quantity.
        cost_column: Column representing the unit cost of items.
        filter_column: Optional column for filtering the data.
        filter_value: Value to filter on (if filter_column is provided).
        discount_rate: Discount rate to apply to the revenue (as a fraction, e.g., 0.1 for 10%).
        
    Returns:
        Net profit calculated as a float.
    """
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_path)

        # Apply filtering if specified
        if filter_column and filter_value:
            df = df[df[filter_column] == filter_value]

        # Calculate the total revenue with discount applied
        total_revenue = df[revenue_column].sum() * (1 - discount_rate)

        # Calculate the total cost
        total_cost = df[cost_column].sum()

        # Calculate net profit
        net_profit = total_revenue - total_cost

        return float(net_profit)
    except Exception as e:
        # Log the error if needed: print(f"An error occurred: {e}")
        return 0.0