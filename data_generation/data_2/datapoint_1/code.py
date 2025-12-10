import pandas as pd
from typing import Optional

def calculate_total_revenue(
    csv_path: str,
    value_column: str = "order_total",
    shipping_column: str = "shipping_cost",
    filter_column: Optional[str] = None,
    filter_value: Optional[str] = None,
    status_column: Optional[str] = "status",
    accepted_status: Optional[str] = "completed",
    discount_rate: float = 0.0,
    include_shipping: bool = True
) -> float:
    """
    Calculate the total revenue from a CSV file of customer orders.
    
    Args:
        csv_path: Path to the CSV file containing order data.
        value_column: Column name for order total values.
        shipping_column: Column name for shipping cost values.
        filter_column: Optional column to filter data by a specific value.
        filter_value: Value to filter on if filter_column is provided.
        status_column: Column to filter orders by status.
        accepted_status: Specific status to include in the calculation.
        discount_rate: Discount rate to apply to the order totals.
        include_shipping: Whether to include shipping cost in the total revenue.
        
    Returns:
        Total calculated revenue as a float. Returns -1 on error.
    """
    try:
        df = pd.read_csv(csv_path)
      
        if filter_column and filter_value:
            df = df[df[filter_column] == filter_value]

        if status_column and accepted_status:
            df = df[df[status_column] == accepted_status]

        df[value_column] = df[value_column] * (1 - discount_rate)

        if include_shipping:
            total_revenue = (df[value_column] + df[shipping_column]).sum()
        else:
            total_revenue = df[value_column].sum()

        return float(total_revenue)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print("Error: The file was not found or is empty.")
        return -1.0
    except KeyError as e:
        print(f"Error: Missing column - {e}")
        return -1.0
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return -1.0