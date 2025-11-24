# Parallelism Benchmark

Chinese poetry parallelism detection benchmark across multiple granularity levels.

## Setup

```bash
git clone https://github.com/mcjkurz/parallelism-benchmark
cd parallelism-benchmark
pip install -r requirements.txt
python setup.py
```

## Usage

Run the scripts in order:

### 1. Train Models
```bash
python train_models.py
```

Loads poems, prepares training data, and trains 4 models:
- Char-level model
- Couplet-level model
- Poem 4-label model
- Poem 1-label model

Saves models and test data to `saved_artifacts/`.

### 2. Evaluate Models
```bash
python evaluate.py
```

Evaluates all models on test sets and runs cross-level evaluations.

### 3. Analyze Scenarios
```bash
python analyze_scenarios.py
```

Analyzes specific failure scenarios and saves results to text files:
- `scenario_A.txt`: Char model fails, couplet model succeeds
- `scenario_B.txt`: Poem4 model fails, couplet model succeeds
- `scenario_C.txt`: Poem1 global hallucination

### 4. Test Single Examples
```bash
python test_single.py
```

Tests all models on single example inputs.

## Project Structure

- `data_loader.py`: Loads and preprocesses poems
- `datasets.py`: PyTorch dataset classes
- `models.py`: Custom model definitions
- `utils.py`: Helper functions
- `train_models.py`: Training pipeline
- `evaluate.py`: Evaluation pipeline
- `analyze_scenarios.py`: Scenario analysis
- `test_single.py`: Single example testing

