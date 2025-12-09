import pandas as pd
from typing import Optional

def calculate_book_value_per_share(
    csv_path: str,
    equity_column: str = "equity",
    liabilities_column: str = "liabilities",
    shares_outstanding_column: str = "shares_outstanding",
    filter_column: Optional[str] = None,
    filter_value: Optional[str] = None,
    minimum_equity: float = 0.0
) -> float:
    """
    Calculate the book value per share from CSV data.
    
    Args:
        csv_path: Path to the CSV file
        equity_column: Column containing the equity values
        liabilities_column: Column containing the liabilities values
        shares_outstanding_column: Column with the number of shares outstanding
        filter_column: Optional column to filter by
        filter_value: Value to filter on (if filter_column is provided)
        minimum_equity: Minimum equity threshold for consideration
        
    Returns:
        Calculated book value per share as a float
    """
    try:
        df = pd.read_csv(csv_path)
        
        if filter_column and filter_value:
            df = df[df[filter_column] == filter_value]
        
        df["net_assets"] = df[equity_column] - df[liabilities_column]
        df = df[df["net_assets"] >= minimum_equity]
        
        total_equity = df["net_assets"].sum()
        total_shares = df[shares_outstanding_column].sum()
        
        if total_shares == 0:
            return 0.0
        
        book_value_per_share = total_equity / total_shares
        return float(book_value_per_share)
    except Exception:
        return 0.0