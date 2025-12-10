import pandas as pd
from typing import Optional

def calculate_expense_ratio(
    csv_path: str,
    total_expense_column: str = "order_total",
    expense_filter_column: Optional[str] = "status",
    expense_filter_value: Optional[str] = "completed",
    shipping_cost_column: str = "shipping_cost",
    aggregation_method: str = "sum",
    min_threshold: float = 0.0
) -> float:
    """
    Calculate the expense ratio from CSV data.
    
    Args:
        csv_path: Path to the CSV file
        total_expense_column: Column containing total expenses to aggregate
        expense_filter_column: Optional column to filter expenses
        expense_filter_value: Value to filter on (if expense_filter_column provided)
        shipping_cost_column: Column containing shipping cost to aggregate
        aggregation_method: Method to aggregate expenses ('sum', 'mean', etc.)
        min_threshold: Minimum threshold for shipping costs to consider
        
    Returns:
        Calculated expense ratio as a float
    """
    try:
        df = pd.read_csv(csv_path)

        if expense_filter_column and expense_filter_value:
            df = df[df[expense_filter_column] == expense_filter_value]

        if aggregation_method == "sum":
            total_expense = df[total_expense_column].sum()
            total_shipping = df[shipping_cost_column].sum()
        elif aggregation_method == "mean":
            total_expense = df[total_expense_column].mean()
            total_shipping = df[shipping_cost_column].mean()
        else:
            raise ValueError("Unsupported aggregation method")

        total_shipping = total_shipping if total_shipping >= min_threshold else 0.0
        
        if total_expense == 0:
            return 0.0

        expense_ratio = total_shipping / total_expense
        return float(expense_ratio)
    except (KeyError, ValueError, pd.errors.EmptyDataError):
        return -1.0
    except Exception:
        return 0.0