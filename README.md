# hypernetwork-code-execution

### Run the code generation
```bash
python data_generation/utils/code_generation.py \
    --num-datapoints 5 \
    --csv-samples 10 \
    --min-lines 30 \
    --max-lines 150 \
    --output-dir ./data \
    --seed 42
```