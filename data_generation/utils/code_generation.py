"""
Code Generation Script using OpenAI API

This script generates Python code that:
1. Contains one function that loads tables from CSV
2. Calculates a numeric value (revenue, total assets, ratios, etc.)
3. Returns a number
4. Has configurable parameters

For each generated code, it also generates a batch of CSV files with 
corresponding parameter values. Ground truth is computed by actually
executing the generated Python code.
"""

import json
import random
import tempfile
import subprocess
import sys
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()


# ============================================================================
# Configuration
# ============================================================================

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


# ============================================================================
# Code Templates and Prompts
# ============================================================================

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
The query should:
- Be a clear, specific question about the data
- Reference the actual column names used in THIS test case's CSV
- Include any filter conditions or parameters that affect the calculation
- Sound like a real business question someone might ask

Query examples (adapt to your actual column names and parameters):
- "What is the total sales amount for the Electronics category?"
- "Calculate the sum of all transaction values where the region is 'North'."
- "What is the average unit_price across all products in warehouse A?"
- "Find the total revenue for Q2 2023."
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


# ============================================================================
# Python Execution Engine
# ============================================================================

class PythonExecutor:
    """Executes Python code safely to compute ground truth results."""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
    
    def execute_function(
        self,
        code: str,
        csv_data: List[Dict[str, Any]],
        parameters: Dict[str, Any]
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Execute the generated Python code with given CSV data and parameters.
        
        Args:
            code: The Python code containing the function
            csv_data: List of dictionaries representing CSV rows
            parameters: Dictionary of function parameters
            
        Returns:
            Tuple of (result, error_message). result is None if execution failed.
        """
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save CSV data
            csv_path = temp_path / "data.csv"
            self._save_csv(csv_data, csv_path)
            
            # Update csv_path in parameters to point to temp file
            exec_params = parameters.copy()
            if "csv_path" in exec_params:
                exec_params["csv_path"] = str(csv_path)
            if "csv_paths" in exec_params:
                # Handle list of paths (replace all with the single temp file)
                exec_params["csv_paths"] = [str(csv_path)]
            
            # Extract function name from code
            func_name = self._extract_function_name(code)
            if func_name is None:
                return None, "Could not extract function name from code"
            
            # Create execution script
            exec_script = self._create_execution_script(code, func_name, exec_params)
            
            script_path = temp_path / "exec_script.py"
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(exec_script)
            
            # Execute the script
            try:
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=str(temp_path)
                )
                
                if result.returncode != 0:
                    return None, f"Execution error: {result.stderr}"
                
                # Parse the output
                output = result.stdout.strip()
                try:
                    value = float(output)
                    return value, None
                except ValueError:
                    return None, f"Could not parse output as float: {output}"
                    
            except subprocess.TimeoutExpired:
                return None, f"Execution timed out after {self.timeout} seconds"
            except Exception as e:
                return None, f"Execution failed: {str(e)}"
    
    def _save_csv(self, rows: List[Dict], path: Path) -> None:
        """Save rows as CSV file."""
        
        if not rows:
            return
        
        import csv
        
        headers = list(rows[0].keys())
        
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
    
    def _extract_function_name(self, code: str) -> Optional[str]:
        """Extract the main function name from code."""
        
        import re
        
        # Find all function definitions
        pattern = r"^def\s+(\w+)\s*\("
        matches = re.findall(pattern, code, re.MULTILINE)
        
        if not matches:
            return None
        
        # Return the first function (main function)
        # Skip helper functions that start with underscore
        for match in matches:
            if not match.startswith("_"):
                return match
        
        return matches[0]
    
    def _create_execution_script(
        self,
        code: str,
        func_name: str,
        parameters: Dict[str, Any]
    ) -> str:
        """Create a script that executes the function and prints the result."""
        
        # Serialize parameters
        params_json = json.dumps(parameters)
        
        error_print = 'print(f"ERROR: {e}", file=sys.stderr)'
        script = f'''
import json
import sys

# The generated code
{code}

# Parameters
params = json.loads('{params_json}')

# Execute and print result
try:
    result = {func_name}(**params)
    print(float(result))
except Exception as e:
    {error_print}
    sys.exit(1)
'''
        return script


# ============================================================================
# OpenAI API Interaction
# ============================================================================

class CodeGenerator:
    """Handles code generation using OpenAI API."""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.client = OpenAI()  # Uses OPENAI_API_KEY env var
        
        if config.seed is not None:
            random.seed(config.seed)
    
    def generate_code(
        self,
        calculation_type: str,
        table_context: Dict[str, Any],
        target_lines: int
    ) -> Optional[str]:
        """Generate Python code using OpenAI."""
        
        # Determine complexity based on target lines
        if target_lines < 40:
            complexity = "simple (basic calculation, few parameters)"
        elif target_lines < 70:
            complexity = "medium (multiple steps, some validation)"
        else:
            complexity = "complex (multiple calculations, extensive validation, helper logic)"
        
        prompt = get_code_generation_prompt(
            calculation_type=calculation_type,
            table_context=table_context,
            target_lines=target_lines,
            complexity_level=complexity
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert Python developer specializing in data analysis. Generate clean, well-documented, executable Python code. The code should be flexible enough to work with different CSV schemas by using column name parameters."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=4000
            )
            
            code = response.choices[0].message.content
            
            # Clean up the code (remove markdown if present)
            code = self._clean_code(code)
            
            return code
            
        except Exception as e:
            print(f"Error generating code: {e}")
            return None
    
    def generate_csv_data(
        self,
        code: str,
        num_samples: int,
        batch_idx: int = 0
    ) -> Optional[Dict[str, Any]]:
        """Generate CSV data and parameters for the given code with diverse schemas."""
        
        prompt = get_csv_generation_prompt(code, num_samples, batch_idx)
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data generation expert. Generate realistic test data with DIVERSE schemas in JSON format. Each test case should have different column names and structures. Output ONLY valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9,  # Higher temperature for more diversity
                max_tokens=16000,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            
            return data
            
        except Exception as e:
            print(f"Error generating CSV data: {e}")
            return None
    
    def _clean_code(self, code: str) -> str:
        """Remove markdown formatting from code if present."""
        
        # Remove markdown code blocks
        if code.startswith("```"):
            lines = code.split("\n")
            # Remove first line (```python or similar)
            lines = lines[1:]
            # Remove last line if it's just ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines)
        
        return code.strip()


# ============================================================================
# Data Generation Pipeline
# ============================================================================

class DataGenerationPipeline:
    """Main pipeline for generating code and data with ground truth execution."""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.generator = CodeGenerator(config)
        self.executor = PythonExecutor(timeout=config.execution_timeout)
        
        # Resolve output path relative to script location if not absolute
        output_path = Path(config.output_dir)
        if not output_path.is_absolute():
            # Make it relative to the script's parent directory (data_generation/)
            script_dir = Path(__file__).resolve().parent.parent
            output_path = script_dir / config.output_dir
        self.output_path = output_path.resolve()
        
    def _get_target_line_counts(self) -> List[int]:
        """Generate uniformly distributed target line counts."""
        
        bucket_size = (self.config.max_code_lines - self.config.min_code_lines) / self.config.code_length_buckets
        
        # Calculate how many datapoints per bucket
        per_bucket = self.config.num_datapoints // self.config.code_length_buckets
        remainder = self.config.num_datapoints % self.config.code_length_buckets
        
        line_counts = []
        
        for bucket_idx in range(self.config.code_length_buckets):
            bucket_start = self.config.min_code_lines + bucket_idx * bucket_size
            bucket_end = bucket_start + bucket_size
            
            # Number of samples for this bucket
            n_samples = per_bucket + (1 if bucket_idx < remainder else 0)
            
            for _ in range(n_samples):
                # Random line count within this bucket
                target = int(random.uniform(bucket_start, bucket_end))
                line_counts.append(target)
        
        random.shuffle(line_counts)
        return line_counts
    
    def _compute_ground_truth(
        self,
        code: str,
        test_cases: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Compute ground truth results by executing code on each test case."""
        
        results = []
        
        for i, test_case in enumerate(test_cases):
            csv_data = test_case.get("csv_data", [])
            parameters = test_case.get("parameters", {})
            
            # Execute the code
            result, error = self.executor.execute_function(code, csv_data, parameters)
            
            if result is not None:
                test_case["expected_result"] = result
                test_case["execution_status"] = "success"
                results.append(test_case)
            else:
                # Log the error but still include the test case with error info
                test_case["expected_result"] = None
                test_case["execution_status"] = "failed"
                test_case["execution_error"] = error
                print(f"    Warning: Test case {i} execution failed: {error}")
                # Skip failed cases
        
        return results
    
    def _save_datapoint(
        self,
        datapoint_id: int,
        code: str,
        test_cases: List[Dict[str, Any]]
    ) -> int:
        """Save a generated datapoint to disk. Returns number of saved samples."""
        
        datapoint_dir = self.output_path / f"datapoint_{datapoint_id}"
        csv_dir = datapoint_dir / "csv"
        
        # Create directories
        datapoint_dir.mkdir(parents=True, exist_ok=True)
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        # Save code
        code_path = datapoint_dir / "code.py"
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)
        
        # Save each test case (only successful ones)
        saved_count = 0
        
        for test_case in test_cases:
            # Skip failed executions
            if test_case.get("execution_status") != "success":
                continue
            
            case_dir = csv_dir / f"sample_{saved_count}"
            case_dir.mkdir(parents=True, exist_ok=True)
            
            # Save schema info for this specific sample
            schema_info = test_case.get("csv_schema", {})
            schema_path = case_dir / "schema.json"
            with open(schema_path, "w", encoding="utf-8") as f:
                json.dump(schema_info, f, indent=2)
            
            # Save CSV data
            csv_rows = test_case.get("csv_data", [])
            if csv_rows:
                csv_path = case_dir / "data.csv"
                self._save_csv(csv_rows, csv_path)
            
            # Save parameters
            params = test_case.get("parameters", {})
            params_path = case_dir / "parameters.json"
            with open(params_path, "w", encoding="utf-8") as f:
                json.dump(params, f, indent=2)
            
            # Save expected result (ground truth computed by execution)
            expected = test_case.get("expected_result")
            result_path = case_dir / "expected_result.json"
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump({"expected_result": expected}, f, indent=2)
            
            # Save natural language query
            query = test_case.get("query", "")
            if query:
                query_path = case_dir / "query.txt"
                with open(query_path, "w", encoding="utf-8") as f:
                    f.write(query)
            
            saved_count += 1
        
        return saved_count
    
    def _save_csv(self, rows: List[Dict], path: Path) -> None:
        """Save rows as CSV file."""
        
        if not rows:
            return
        
        import csv
        
        headers = list(rows[0].keys())
        
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
    
    def run(self) -> None:
        """Run the full data generation pipeline."""
        
        print("Starting data generation pipeline...")
        print(f"Config: {self.config}")
        print()
        
        # Get target line counts for uniform distribution
        target_lines = self._get_target_line_counts()
        
        successful = 0
        failed = 0
        
        for i, target_line_count in enumerate(target_lines):
            datapoint_id = i + 1
            
            print(f"[{datapoint_id}/{len(target_lines)}] Generating datapoint...")
            print(f"  Target lines: {target_line_count}")
            
            # Randomly select calculation type and table context
            calc_type = random.choice(CALCULATION_TYPES)
            table_ctx = random.choice(TABLE_CONTEXTS)
            
            print(f"  Calculation type: {calc_type}")
            print(f"  Table context: {table_ctx['name']}")
            
            # Generate code
            code = self.generator.generate_code(
                calculation_type=calc_type,
                table_context=table_ctx,
                target_lines=target_line_count
            )
            
            if code is None:
                print("  ERROR: Failed to generate code")
                failed += 1
                continue
            
            actual_lines = len(code.split("\n"))
            print(f"  Generated code: {actual_lines} lines")
            
            # Generate CSV data and parameters in batches for more diversity
            print(f"  Generating {self.config.csv_samples_per_code} CSV samples with diverse schemas...")
            
            all_test_cases = []
            samples_per_batch = min(10, self.config.csv_samples_per_code)
            num_batches = (self.config.csv_samples_per_code + samples_per_batch - 1) // samples_per_batch
            
            for batch_idx in range(num_batches):
                remaining = self.config.csv_samples_per_code - len(all_test_cases)
                batch_size = min(samples_per_batch, remaining)
                
                if batch_size <= 0:
                    break
                
                csv_data = self.generator.generate_csv_data(
                    code=code,
                    num_samples=batch_size,
                    batch_idx=batch_idx
                )
                
                if csv_data is not None:
                    batch_cases = csv_data.get("test_cases", [])
                    all_test_cases.extend(batch_cases)
                    print(f"    Batch {batch_idx + 1}: Generated {len(batch_cases)} test cases")
            
            if not all_test_cases:
                print("  ERROR: Failed to generate any CSV data")
                failed += 1
                continue
            
            print("  Computing ground truth by executing code...")
            
            # Compute ground truth by actually executing the code
            validated_cases = self._compute_ground_truth(code, all_test_cases)
            
            success_count = sum(1 for tc in validated_cases if tc.get("execution_status") == "success")
            print(f"  Execution results: {success_count}/{len(all_test_cases)} successful")
            
            if success_count == 0:
                print("  ERROR: No test cases executed successfully")
                failed += 1
                continue
            
            # Save datapoint
            saved_count = self._save_datapoint(datapoint_id, code, validated_cases)
            print(f"  Saved {saved_count} samples to: {self.output_path / f'datapoint_{datapoint_id}'}")
            
            successful += 1
            print()
        
        print("Generation complete!")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Output directory: {self.output_path}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for code generation."""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate Python code and CSV data for calculations with ground truth execution"
    )
    
    parser.add_argument(
        "--num-datapoints",
        type=int,
        default=10,
        help="Number of code files to generate (default: 10)"
    )
    
    parser.add_argument(
        "--csv-samples",
        type=int,
        default=100,
        help="Number of CSV samples per code file (default: 100)"
    )
    
    parser.add_argument(
        "--min-lines",
        type=int,
        default=20,
        help="Minimum code lines (default: 20)"
    )
    
    parser.add_argument(
        "--max-lines",
        type=int,
        default=100,
        help="Maximum code lines (default: 100)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory (default: data)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--buckets",
        type=int,
        default=5,
        help="Number of code length buckets for uniform distribution (default: 5)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Execution timeout in seconds (default: 30)"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = GenerationConfig(
        openai_model=args.model,
        min_code_lines=args.min_lines,
        max_code_lines=args.max_lines,
        code_length_buckets=args.buckets,
        num_datapoints=args.num_datapoints,
        csv_samples_per_code=args.csv_samples,
        output_dir=args.output_dir,
        seed=args.seed,
        execution_timeout=args.timeout
    )
    
    # Run pipeline
    pipeline = DataGenerationPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
