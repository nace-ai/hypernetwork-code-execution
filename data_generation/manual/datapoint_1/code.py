import pandas as pd
import re
from typing import Optional, Dict, Union


def read_markdown_tables(file_path: str) -> Dict[str, pd.DataFrame]:
    """
    Read all tables from a markdown file and return them as a dictionary.
    
    Args:
        file_path: Path to the markdown file
        
    Returns:
        Dictionary mapping table names to DataFrames
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    tables = {}
    
    # Pattern to find table headers (lines before tables starting with #)
    # and markdown tables (lines with | separators)
    lines = content.split('\n')
    
    current_table_name = None
    table_lines = []
    
    for i, line in enumerate(lines):
        # Check for header (potential table name)
        if line.startswith('#'):
            # If we were collecting a table, save it
            if table_lines and current_table_name:
                df = parse_markdown_table(table_lines)
                if df is not None:
                    tables[current_table_name] = df
            # Reset for new section
            current_table_name = line.lstrip('#').strip()
            table_lines = []
        # Check if line is part of a markdown table
        elif '|' in line:
            table_lines.append(line)
        else:
            # Non-table line encountered, save current table if exists
            if table_lines and current_table_name:
                df = parse_markdown_table(table_lines)
                if df is not None:
                    tables[current_table_name] = df
                table_lines = []
    
    # Don't forget the last table
    if table_lines and current_table_name:
        df = parse_markdown_table(table_lines)
        if df is not None:
            tables[current_table_name] = df
    
    return tables


def parse_markdown_table(lines: list) -> Optional[pd.DataFrame]:
    """
    Parse markdown table lines into a DataFrame.
    
    Args:
        lines: List of markdown table lines
        
    Returns:
        DataFrame or None if parsing fails
    """
    if len(lines) < 2:
        return None
    
    # Filter out separator lines (lines with only |, -, :, and spaces)
    data_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not re.match(r'^[\|\-\:\s]+$', stripped):
            data_lines.append(stripped)
    
    if len(data_lines) < 1:
        return None
    
    # Parse header
    header = [col.strip() for col in data_lines[0].split('|') if col.strip()]
    
    # Parse data rows
    rows = []
    for line in data_lines[1:]:
        row = [cell.strip() for cell in line.split('|') if cell.strip()]
        if len(row) == len(header):
            rows.append(row)
    
    if not rows:
        return None
    
    df = pd.DataFrame(rows, columns=header)
    
    # Try to convert numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col].str.replace(',', '').str.replace('$', ''), errors='ignore')
    
    return df


def calculate_total_assets(
    data: Dict[str, pd.DataFrame],
    table_name: Optional[str] = None,
    total_asset_key: Optional[str] = None,
    liability_key: Optional[str] = None,
    equity_key: Optional[str] = None,
    period_key: Optional[str] = None,
    period_value: Optional[Union[str, int]] = None
) -> Union[pd.Series, float, str]:
    """
    Calculate total assets from the data.
    
    Logic:
    1. If no table_name provided, return 'NA'
    2. If period_key and period_value are provided, filter to that specific row
    3. If total_asset_key is provided and exists, return that column/value
    4. If both liability_key and equity_key are provided, return their sum
    5. Otherwise return 'NA'
    
    Args:
        data: Dictionary of DataFrames keyed by table name
        table_name: Name of the table to use
        total_asset_key: Column name for total assets (direct lookup)
        liability_key: Column name for liabilities
        equity_key: Column name for equity
        period_key: Column name for the period/year (e.g., "Year", "Quarter", "Period")
        period_value: The specific period value to filter by (e.g., 2023, "Q1 2024")
        
    Returns:
        Total assets as float (if period specified) or Series, or 'NA' if cannot be calculated
    """
    if not table_name:
        return 'NA'
    
    if table_name not in data:
        return 'NA'
    
    pd_table = data[table_name]
    
    # Filter by period if specified
    if period_key and period_value is not None:
        if period_key not in pd_table.columns:
            return 'NA'
        # Handle both string and numeric period values
        mask = pd_table[period_key] == period_value
        if not mask.any():
            # Try string comparison
            mask = pd_table[period_key].astype(str) == str(period_value)
        if not mask.any():
            return 'NA'
        pd_table = pd_table[mask]
    
    if total_asset_key and total_asset_key in pd_table.columns:
        result = pd_table[total_asset_key]
        # Return single value if filtered to one row
        if len(result) == 1:
            return result.iloc[0]
        return result
    
    if liability_key and equity_key:
        if liability_key in pd_table.columns and equity_key in pd_table.columns:
            result = pd_table[liability_key] + pd_table[equity_key]
            # Return single value if filtered to one row
            if len(result) == 1:
                return result.iloc[0]
            return result
    
    return 'NA'


def calculate_revenue(
    data: Dict[str, pd.DataFrame],
    table_name: Optional[str] = None,
    revenue_key: Optional[str] = None,
    sales_key: Optional[str] = None,
    other_income_key: Optional[str] = None,
    period_key: Optional[str] = None,
    period_value: Optional[Union[str, int]] = None
) -> Union[pd.Series, float, str]:
    """
    Calculate revenue from the data.
    
    Logic:
    1. If no table_name provided, return 'NA'
    2. If period_key and period_value are provided, filter to that specific row
    3. If revenue_key is provided and exists, return that column/value
    4. If sales_key and other_income_key are provided, return their sum
    5. If only sales_key is provided, return that
    6. Otherwise return 'NA'
    
    Args:
        data: Dictionary of DataFrames keyed by table name
        table_name: Name of the table to use
        revenue_key: Column name for revenue (direct lookup)
        sales_key: Column name for sales
        other_income_key: Column name for other income
        period_key: Column name for the period/year (e.g., "Year", "Quarter", "Period")
        period_value: The specific period value to filter by (e.g., 2023, "Q1 2024")
        
    Returns:
        Revenue as float (if period specified) or Series, or 'NA' if cannot be calculated
    """
    if not table_name:
        return 'NA'
    
    if table_name not in data:
        return 'NA'
    
    pd_table = data[table_name]
    
    # Filter by period if specified
    if period_key and period_value is not None:
        if period_key not in pd_table.columns:
            return 'NA'
        # Handle both string and numeric period values
        mask = pd_table[period_key] == period_value
        if not mask.any():
            # Try string comparison
            mask = pd_table[period_key].astype(str) == str(period_value)
        if not mask.any():
            return 'NA'
        pd_table = pd_table[mask]
    
    if revenue_key and revenue_key in pd_table.columns:
        result = pd_table[revenue_key]
        if len(result) == 1:
            return result.iloc[0]
        return result
    
    if sales_key and sales_key in pd_table.columns:
        result = pd_table[sales_key]
        if other_income_key and other_income_key in pd_table.columns:
            result = result + pd_table[other_income_key]
        if len(result) == 1:
            return result.iloc[0]
        return result
    
    return 'NA'


# Example usage
if __name__ == "__main__":
    # Example: Create a sample markdown file for testing
    sample_markdown = """
# Balance Sheet

| Year | Total Assets | Liabilities | Equity |
|------|-------------|-------------|--------|
| 2022 | 1000000 | 600000 | 400000 |
| 2023 | 1200000 | 700000 | 500000 |

# Income Statement

| Year | Revenue | Sales | Other Income | Expenses |
|------|---------|-------|--------------|----------|
| 2022 | 500000 | 450000 | 50000 | 300000 |
| 2023 | 650000 | 600000 | 50000 | 400000 |
"""
    
    # Write sample file
    with open('sample_data.md', 'w') as f:
        f.write(sample_markdown)
    
    # Read tables from markdown
    tables = read_markdown_tables('sample_data.md')
    
    print("Available tables:", list(tables.keys()))
    print()
    
    # Calculate total assets for a specific year (returns single value)
    total_assets_2023 = calculate_total_assets(
        data=tables,
        table_name="Balance Sheet",
        total_asset_key="Total Assets",
        period_key="Year",
        period_value=2023
    )
    print("Total Assets for 2023 (single value):", total_assets_2023)
    print()
    
    # Calculate total assets for all years (returns series)
    total_assets_all = calculate_total_assets(
        data=tables,
        table_name="Balance Sheet",
        total_asset_key="Total Assets"
    )
    print("Total Assets (all years):")
    print(total_assets_all)
    print()
    
    # Calculate total assets using liability + equity for specific year
    total_assets_calc_2022 = calculate_total_assets(
        data=tables,
        table_name="Balance Sheet",
        liability_key="Liabilities",
        equity_key="Equity",
        period_key="Year",
        period_value=2022
    )
    print("Total Assets 2022 (calculated from Liabilities + Equity):", total_assets_calc_2022)
    print()
    
    # Calculate revenue for specific year
    revenue_2023 = calculate_revenue(
        data=tables,
        table_name="Income Statement",
        revenue_key="Revenue",
        period_key="Year",
        period_value=2023
    )
    print("Revenue for 2023:", revenue_2023)
    print()
    
    # Test with missing table
    result = calculate_total_assets(
        data=tables,
        table_name="NonExistent Table",
        total_asset_key="Total Assets"
    )
    print("Result with missing table:", result)
    
    # Test with non-existent year
    result_bad_year = calculate_total_assets(
        data=tables,
        table_name="Balance Sheet",
        total_asset_key="Total Assets",
        period_key="Year",
        period_value=2020
    )
    print("Result with non-existent year:", result_bad_year)

