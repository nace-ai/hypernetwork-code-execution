import pandas as pd
from typing import Optional

def calculate_quick_ratio(
    csv_path: str,
    current_assets_column: str = "order_total",
    inventory_column: str = "inventory_value",
    current_liabilities_column: str = "liabilities",
    status_column: Optional[str] = "status",
    status_filter: Optional[str] = "completed",
    min_threshold: float = 0.1
) -> float:
    """
    Calculate the quick ratio from CSV data, which measures the ability of a business
    to meet its short-term obligations with its most liquid assets.

    Args:
        csv_path: Path to the CSV file
        current_assets_column: Column representing current assets (typically order totals)
        inventory_column: Column representing inventory value
        current_liabilities_column: Column representing current liabilities
        status_column: Optional column to filter rows based on status
        status_filter: Status value to filter on (if status_column is provided)
        min_threshold: Minimum threshold for the quick ratio to be considered valid

    Returns:
        The calculated quick ratio as a float, or -1 if an error occurs or the data is invalid
    """
    try:
        # Load the CSV data
        df = pd.read_csv(csv_path)

        # Filter based on status if specified
        if status_column and status_filter:
            df = df[df[status_column] == status_filter]

        # Validate necessary columns exist
        if not all(col in df.columns for col in [current_assets_column, inventory_column, current_liabilities_column]):
            return -1.0

        # Calculate quick assets (current assets minus inventory)
        df['quick_assets'] = df[current_assets_column] - df[inventory_column]

        # Calculate the sum of quick assets and current liabilities
        total_quick_assets = df['quick_assets'].sum()
        total_current_liabilities = df[current_liabilities_column].sum()

        # Validate if there are any current liabilities to avoid division by zero
        if total_current_liabilities == 0:
            return -1.0

        # Calculate the quick ratio
        quick_ratio = total_quick_assets / total_current_liabilities

        # Ensure the quick ratio meets the minimum threshold
        return float(quick_ratio) if quick_ratio >= min_threshold else -1.0

    except Exception:
        return -1.0