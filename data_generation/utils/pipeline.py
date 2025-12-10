"""
Data generation pipeline for code and CSV data with ground truth execution.
"""

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List

try:
    from .config import GenerationConfig, CALCULATION_TYPES, TABLE_CONTEXTS
    from .executor import PythonExecutor
    from .generator import CodeGenerator
except ImportError:
    from config import GenerationConfig, CALCULATION_TYPES, TABLE_CONTEXTS
    from executor import PythonExecutor
    from generator import CodeGenerator


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

