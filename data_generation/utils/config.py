"""
Configuration classes and constants for code generation.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GenerationConfig:
    """Configuration for code and data generation."""
    
    # OpenAI settings
    openai_model: str = "gpt-4o"
    
    # Code length settings (in lines, for uniform distribution)
    min_code_lines: int = 20
    max_code_lines: int = 100
    code_length_buckets: int = 5  # Number of uniform buckets for code length
    
    # Data generation settings
    num_datapoints: int = 10  # Number of code files to generate
    csv_samples_per_code: int = 100  # Number of CSV files per code
    
    # Output paths
    output_dir: str = "data"
    
    # Seed for reproducibility
    seed: Optional[int] = None
    
    # Execution timeout in seconds
    execution_timeout: int = 30


# Types of calculations that can be generated
CALCULATION_TYPES = [
    "total_assets",
    "total_revenue",
    "net_profit",
    "profit_margin",
    "debt_to_equity_ratio",
    "return_on_assets",
    "return_on_equity",
    "current_ratio",
    "quick_ratio",
    "gross_profit_margin",
    "operating_income",
    "working_capital",
    "asset_turnover",
    "inventory_turnover",
    "revenue_growth_rate",
    "expense_ratio",
    "cash_flow_from_operations",
    "total_liabilities",
    "shareholders_equity",
    "book_value_per_share",
]

# Diverse table contexts with specific domain characteristics
TABLE_CONTEXTS = [
    {
        "name": "bank_holdings",
        "typical_columns": ["account_id", "holder_name", "balance", "account_type", "interest_rate", "open_date"],
        "numeric_columns": ["balance", "interest_rate"],
        "categorical_columns": ["account_type"],
    },
    {
        "name": "retail_sales",
        "typical_columns": ["transaction_id", "product_name", "quantity", "unit_price", "discount", "sale_date", "region"],
        "numeric_columns": ["quantity", "unit_price", "discount"],
        "categorical_columns": ["region", "product_name"],
    },
    {
        "name": "employee_payroll",
        "typical_columns": ["employee_id", "department", "base_salary", "bonus", "tax_deduction", "hire_date", "position"],
        "numeric_columns": ["base_salary", "bonus", "tax_deduction"],
        "categorical_columns": ["department", "position"],
    },
    {
        "name": "inventory_stock",
        "typical_columns": ["sku", "product_name", "quantity_on_hand", "reorder_level", "unit_cost", "warehouse"],
        "numeric_columns": ["quantity_on_hand", "reorder_level", "unit_cost"],
        "categorical_columns": ["warehouse", "product_name"],
    },
    {
        "name": "customer_orders",
        "typical_columns": ["order_id", "customer_id", "order_total", "shipping_cost", "order_date", "status", "payment_method"],
        "numeric_columns": ["order_total", "shipping_cost"],
        "categorical_columns": ["status", "payment_method"],
    },
    {
        "name": "investment_portfolio",
        "typical_columns": ["ticker", "shares", "purchase_price", "current_price", "purchase_date", "sector"],
        "numeric_columns": ["shares", "purchase_price", "current_price"],
        "categorical_columns": ["sector", "ticker"],
    },
    {
        "name": "expense_report",
        "typical_columns": ["expense_id", "category", "amount", "date", "department", "approved_by", "vendor"],
        "numeric_columns": ["amount"],
        "categorical_columns": ["category", "department", "vendor"],
    },
    {
        "name": "loan_portfolio",
        "typical_columns": ["loan_id", "principal", "interest_rate", "term_months", "monthly_payment", "status", "loan_type"],
        "numeric_columns": ["principal", "interest_rate", "term_months", "monthly_payment"],
        "categorical_columns": ["status", "loan_type"],
    },
    {
        "name": "quarterly_financials",
        "typical_columns": ["fiscal_year", "quarter", "revenue", "cost_of_goods", "operating_expenses", "net_income", "division"],
        "numeric_columns": ["revenue", "cost_of_goods", "operating_expenses", "net_income"],
        "categorical_columns": ["division"],
    },
    {
        "name": "real_estate_listings",
        "typical_columns": ["property_id", "address", "listing_price", "square_feet", "bedrooms", "property_type", "status"],
        "numeric_columns": ["listing_price", "square_feet", "bedrooms"],
        "categorical_columns": ["property_type", "status"],
    },
]

