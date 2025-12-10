"""
Code and data generation using OpenAI API.
"""

import json
import random
from typing import Any, Dict, Optional

from openai import OpenAI

try:
    from .config import GenerationConfig
    from .prompts import get_code_generation_prompt, get_csv_generation_prompt
except ImportError:
    from config import GenerationConfig
    from prompts import get_code_generation_prompt, get_csv_generation_prompt


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

