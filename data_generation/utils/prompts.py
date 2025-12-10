"""
Prompt templates for code and data generation.
"""

from typing import Dict, Any


def get_code_generation_prompt(
    calculation_type: str,
    table_context: Dict[str, Any],
    target_lines: int,
    complexity_level: str
) -> str:
    """Generate the prompt for OpenAI to create Python code."""
    
    context_name = table_context["name"]
    typical_cols = table_context["typical_columns"]
    numeric_cols = table_context["numeric_columns"]
    categorical_cols = table_context["categorical_columns"]
    
    return f"""Generate a Python function that performs calculations on CSV data.

REQUIREMENTS:
1. Create exactly ONE main function that:
   - Loads a CSV file using pandas (single csv_path parameter, not a list)
   - Performs a calculation related to: {calculation_type}
   - Returns a single numeric value (float or int)
   - Has meaningful parameters that affect the calculation

2. IMPORTANT - FLEXIBLE COLUMN HANDLING:
   The function should accept column names as parameters (not hardcoded).
   This allows the same function to work with different CSV schemas.
   
   For example, instead of:
   ```python
   df["Revenue"].sum()  # Bad: hardcoded column name
   ```
   Use:
   ```python
   def calculate_total(csv_path: str, value_column: str = "Revenue") -> float:
       df[value_column].sum()  # Good: column name is a parameter
   ```

3. The function should be self-contained and include:
   - Proper type hints
   - A docstring explaining what it does
   - Input validation
   - Error handling that returns a default numeric value (like 0.0 or -1) on errors

4. Code specifications:
   - Target approximately {target_lines} lines of code (including imports, docstring, comments)
   - Complexity level: {complexity_level}
   - Table context: {context_name}

5. Context-specific guidance:
   - Typical columns for this domain: {typical_cols}
   - Numeric columns: {numeric_cols}
   - Categorical columns (for filtering/grouping): {categorical_cols}
   
   Use parameter names that make sense for this domain (e.g., value_column, filter_column, group_by_column).

6. The function signature should look like:
   def calculate_<metric>(csv_path: str, <column_params>, <filter_params>, <other_params>) -> float:
   
   Parameter categories to consider:
   - Column name parameters (e.g., value_column, filter_column)
   - Filter parameters (e.g., filter_value, min_threshold)
   - Aggregation parameters (e.g., aggregation_method)
   - Numeric modifiers (e.g., multiplier, discount_rate)

7. CSV Loading:
   - Use pandas to load CSV files
   - The CSV will have headers in the first row
   - Accept a single csv_path (string), not a list

8. IMPORTANT: 
   - Only output the Python code, nothing else
   - Include all necessary imports at the top
   - The code must be executable
   - Do NOT include example usage or main block
   - Do NOT use markdown code blocks

EXAMPLE STRUCTURE (for reference only, be creative with yours):

import pandas as pd
from typing import Optional

def calculate_metric(
    csv_path: str,
    value_column: str = "Amount",
    filter_column: Optional[str] = None,
    filter_value: Optional[str] = None,
    multiplier: float = 1.0
) -> float:
    \"\"\"
    Calculate a metric from CSV data.
    
    Args:
        csv_path: Path to the CSV file
        value_column: Column containing numeric values to aggregate
        filter_column: Optional column to filter by
        filter_value: Value to filter on (if filter_column provided)
        multiplier: Factor to apply to the result
        
    Returns:
        Calculated metric as a float
    \"\"\"
    try:
        df = pd.read_csv(csv_path)
        
        if filter_column and filter_value:
            df = df[df[filter_column] == filter_value]
        
        result = df[value_column].sum() * multiplier
        return float(result)
    except Exception:
        return 0.0

Now generate the code for {calculation_type} with {context_name} context:"""


def get_csv_generation_prompt(code: str, num_samples: int, sample_batch_idx: int) -> str:
    """Generate the prompt for creating CSV data and parameters with diverse schemas."""
    
    return f"""Based on the following Python code, generate test data with DIVERSE schemas.

PYTHON CODE:
```python
{code}
```

TASK:
Generate {num_samples} different test cases. For each test case:
1. Generate CSV data with a DIFFERENT schema variation
2. Provide appropriate parameter values that match the CSV schema
3. Generate a NATURAL LANGUAGE QUERY that asks for the calculation result
4. DO NOT calculate expected_result - it will be computed by running the code

CRITICAL - SCHEMA DIVERSITY:
Each test case should have a DIFFERENT CSV schema. Vary:
- Column NAMES (use synonyms, abbreviations, different naming conventions)
- Column ORDER
- Number of ROWS (3-20 rows)
- Additional COLUMNS (extra columns that won't be used)
- Data VALUE ranges (small numbers, large numbers, decimals, negatives where appropriate)

Examples of column name variations:
- "Revenue" → "revenue", "sales", "income", "total_sales", "sales_amount", "gross_revenue"
- "Amount" → "amount", "value", "sum", "total", "amt"
- "Date" → "date", "transaction_date", "dt", "period", "timestamp"
- "Category" → "category", "type", "cat", "classification", "group"

QUERY GENERATION:
Generate a natural language query that asks for the specific calculation result.

CRITICAL - QUERY ACCURACY:
The query MUST accurately describe what the code ACTUALLY computes. Carefully trace through the code logic:
- The function returns a SINGLE numeric value (float)
- If the code uses group_by but then aggregates all groups into one final number, the query should reflect this
- Do NOT say "for each category" if the code returns one aggregated number
- Do NOT imply multiple results when the code returns a single value

For example, if the code:
1. Groups by category
2. Sums values per category
3. Then sums/averages ALL those group results into ONE final number
The query should say "What is the overall/total X across all categories?" NOT "What is X for each category?"

CRITICAL - INCLUDE ALL CONDITIONS:
The query MUST mention ALL parameters that affect the final result, including:
- Filter conditions (e.g., "where region is 'North'")
- Threshold filters (e.g., "excluding values below 1000", "where cash flow is at least 1000")
- Grouping operations that affect the calculation
- Any multipliers, rates, or modifiers applied

For example, if parameters include:
- filter_column="region", filter_value="North"
- min_threshold=1000
- group_by_column="category"

BAD query: "What is the total cash flow for the North region?"
GOOD query: "What is the total cash flow for the North region, grouped by category, excluding categories with cash flow below 1000?"

The query should:
- Be a clear, specific question about the data
- ACCURATELY describe the exact computation the code performs
- MENTION ALL filtering conditions and thresholds from parameters
- Reference the actual column names used in THIS test case's CSV
- Sound like a real business question someone might ask

Query examples (adapt to your actual column names and parameters):
- "What is the total sales amount for the Electronics category where quantity exceeds 50?"
- "Calculate the sum of all transaction values in the 'North' region with amounts above 100."
- "What is the average unit_price across all products in warehouse A, excluding items below the 500 threshold?"
- "What is the overall asset turnover ratio when grouping sales and assets by category?"
- "What is the gross profit margin when applying a 1.15 multiplier to the cost column?"

This is batch {sample_batch_idx} of samples. Make sure schemas are different from typical patterns.

OUTPUT FORMAT (JSON):
{{
    "test_cases": [
        {{
            "csv_schema": {{
                "columns": ["column1", "column2", ...],
                "description": "Brief description of this schema variant"
            }},
            "csv_data": [
                {{"column1": value1, "column2": value2, ...}},
                ...
            ],
            "parameters": {{
                "csv_path": "data.csv",
                "param1": value1,
                "param2": value2,
                ...
            }},
            "query": "Natural language question asking for the calculation result"
        }},
        ...
    ]
}}

REQUIREMENTS:
1. Generate exactly {num_samples} test cases with DIVERSE schemas
2. Column names in parameters MUST match the actual CSV column names
3. csv_path should always be "data.csv"
4. DO NOT include expected_result - it will be computed automatically
5. All numeric values should be reasonable (no extreme values)
6. Ensure the data makes logical sense for each schema
7. Each query must be specific to that test case's data and parameters

IMPORTANT:
- Output ONLY valid JSON, no markdown or explanation
- Focus on schema diversity - each test case should feel like it came from a different data source
- Parameter column names must exactly match CSV column names
- Queries should be diverse in phrasing and structure"""

