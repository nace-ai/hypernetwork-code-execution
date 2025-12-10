import pandas as pd
from typing import Optional

def calculate_net_profit(
    csv_path: str,
    revenue_column: str = "Revenue",
    expense_column: str = "Amount",
    revenue_filter_column: Optional[str] = None,
    revenue_filter_value: Optional[str] = None,
    expense_filter_column: Optional[str] = None,
    expense_filter_value: Optional[str] = None,
    discount_rate: float = 0.0
) -> float:
    """
    Calculate the net profit from CSV data containing revenue and expenses.
    
    Args:
        csv_path: Path to the CSV file
        revenue_column: Column containing revenue values
        expense_column: Column containing expense values
        revenue_filter_column: Optional column to filter revenue by
        revenue_filter_value: Value to filter revenue on (if revenue_filter_column provided)
        expense_filter_column: Optional column to filter expenses by
        expense_filter_value: Value to filter expenses on (if expense_filter_column provided)
        discount_rate: Rate to apply as a discount on expenses
        
    Returns:
        Calculated net profit as a float
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Calculate total revenue
        if revenue_filter_column and revenue_filter_value:
            revenue_df = df[df[revenue_filter_column] == revenue_filter_value]
        else:
            revenue_df = df
        total_revenue = revenue_df[revenue_column].sum()
        
        # Calculate total expenses
        if expense_filter_column and expense_filter_value:
            expense_df = df[df[expense_filter_column] == expense_filter_value]
        else:
            expense_df = df
        total_expenses = expense_df[expense_column].sum() * (1 - discount_rate)
        
        # Calculate net profit
        net_profit = total_revenue - total_expenses
        
        return float(net_profit)
    except Exception:
        return 0.0