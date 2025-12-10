"""
Main entry point for code generation.

This script generates Python code that:
1. Contains one function that loads tables from CSV
2. Calculates a numeric value (revenue, total assets, ratios, etc.)
3. Returns a number
4. Has configurable parameters

For each generated code, it also generates a batch of CSV files with 
corresponding parameter values. Ground truth is computed by actually
executing the generated Python code.
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# Handle imports for both direct execution and module execution
if __name__ == "__main__":
    # Add parent directory to path for direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import GenerationConfig
    from pipeline import DataGenerationPipeline
else:
    from .config import GenerationConfig
    from .pipeline import DataGenerationPipeline

load_dotenv()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(
        description="Generate Python code and CSV data for calculations with ground truth execution"
    )
    
    parser.add_argument(
        "--num-datapoints",
        type=int,
        default=5,
        help="Number of code files to generate (default: 10)"
    )
    
    parser.add_argument(
        "--csv-samples",
        type=int,
        default=10,
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
        default="./data_2",
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
    
    return parser.parse_args()


def main():
    """Main entry point for code generation."""
    
    args = parse_args()
    
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

