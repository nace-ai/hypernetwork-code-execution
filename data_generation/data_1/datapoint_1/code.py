import pandas as pd
from typing import Optional

def calculate_book_value_per_share(
    csv_path: str,
    total_assets_column: str = "total_assets",
    total_liabilities_column: str = "total_liabilities",
    shares_outstanding_column: str = "shares_outstanding",
    filter_column: Optional[str] = None,
    filter_value: Optional[str] = None,
    group_by_column: Optional[str] = None,
    min_threshold: Optional[float] = 0.0,
    discount_rate: float = 1.0
) -> float:
    """
    Calculate the book value per share from CSV data.

    Args:
        csv_path: Path to the CSV file.
        total_assets_column: Column name for total assets.
        total_liabilities_column: Column name for total liabilities.
        shares_outstanding_column: Column name for shares outstanding.
        filter_column: Optional column to filter by.
        filter_value: Value to filter on (if filter_column provided).
        group_by_column: Optional column to group by for aggregation.
        min_threshold: Minimum threshold for filtering net assets.
        discount_rate: Factor to apply as a discount to the result.

    Returns:
        Calculated book value per share as a float, or -1 on error.
    """
    try:
        # Load the CSV data into a DataFrame
        df = pd.read_csv(csv_path)

        # Apply filtering if filter_column and filter_value are provided
        if filter_column and filter_value:
            df = df[df[filter_column] == filter_value]

        # Calculate net assets
        df["net_assets"] = df[total_assets_column] - df[total_liabilities_column]

        # Apply minimum threshold filtering on net assets
        df = df[df["net_assets"] > min_threshold]

        if group_by_column:
            # Group by the specified column and calculate the mean book value per share
            grouped = df.groupby(group_by_column).apply(
                lambda x: (x["net_assets"].sum() / x[shares_outstanding_column].sum())
                if x[shares_outstanding_column].sum() != 0 else 0
            )
            # Apply the discount rate
            result = grouped.mean() * discount_rate
        else:
            # Calculate the book value per share without grouping
            total_net_assets = df["net_assets"].sum()
            total_shares = df[shares_outstanding_column].sum()
            if total_shares == 0:
                return -1.0
            result = (total_net_assets / total_shares) * discount_rate

        return float(result)
    except Exception as e:
        print(f"Error calculating book value per share: {e}")
        return -1.0