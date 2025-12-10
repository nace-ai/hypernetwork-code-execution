import pandas as pd
from typing import Optional

def calculate_working_capital(
    csv_path: str,
    listing_price_column: str = "listing_price",
    property_type_column: Optional[str] = None,
    filter_property_type: Optional[str] = None,
    discount_rate: float = 0.0
) -> float:
    """
    Calculate the working capital from real estate listings data.
    
    Args:
        csv_path: Path to the CSV file
        listing_price_column: Column containing the listing prices
        property_type_column: Optional column to filter property types
        filter_property_type: Property type to filter on (if property_type_column provided)
        discount_rate: Discount rate to apply to the total listing price
        
    Returns:
        Calculated working capital as a float
    """
    try:
        df = pd.read_csv(csv_path)
        
        if property_type_column and filter_property_type:
            df = df[df[property_type_column] == filter_property_type]
        
        total_value = df[listing_price_column].sum()
        working_capital = total_value * (1 - discount_rate)
        
        return float(working_capital)
    except Exception:
        return 0.0