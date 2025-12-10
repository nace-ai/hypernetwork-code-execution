"""
Python execution engine for computing ground truth results.
"""

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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

