import pandas as pd
from typing import Optional

def calculate_total_assets(
    csv_path: str,
    price_column: str = "listing_price",
    filter_column: Optional[str] = None,
    filter_value: Optional[str] = None,
    discount_rate: float = 0.0,
    status_column: Optional[str] = "status",
    active_status: str = "active"
) -> float:
    """
    Calculate the total assets from real estate listing data.

    Args:
        csv_path: Path to the CSV file.
        price_column: Column containing property prices.
        filter_column: Optional column to filter properties.
        filter_value: Value to filter on (if filter_column provided).
        discount_rate: Discount applied to the total amount.
        status_column: Column indicating the status of the property.
        active_status: Property status to include in the calculation.

    Returns:
        Total assets as a float, after applying discount.
    """
    try:
        df = pd.read_csv(csv_path)
        
        if filter_column and filter_value:
            df = df[df[filter_column] == filter_value]
        
        if status_column:
            df = df[df[status_column] == active_status]

        total_assets = df[price_column].sum()
        total_assets *= (1 - discount_rate)
        
        return float(total_assets)
    except Exception:
        return 0.0