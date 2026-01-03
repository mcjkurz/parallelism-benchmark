# Parallelism Benchmark

Chinese poetry parallelism detection benchmark across multiple granularity levels.

## Supported Devices

- **CUDA** (NVIDIA GPUs)
- **MPS** (Apple Silicon - M1/M2/M3)
- **CPU** (fallback, slower)

## Setup

```bash
git clone https://github.com/mcjkurz/parallelism-benchmark
cd parallelism-benchmark
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python setup.py
```

## Quick Start: Full Pipeline

Run the complete pipeline with 100 trials for statistical evaluation:

```bash
./pipeline.sh
```

Options:
```bash
./pipeline.sh 50              # Run with 50 trials instead of 100
./pipeline.sh 100 --skip-train  # Skip training, only run multi-trial evaluation
```

This will:
1. Train all 4 models (1 epoch each)
2. Export the Silver Standard dataset to `data/silver_standard.json`
3. Export training/test splits to `saved_artifacts/*.json`
4. Run 100 training+evaluation trials with different seeds
5. Report mean ± std statistics for all metrics

## Usage (Individual Scripts)

### 1. Train Models
```bash
python train_models.py
```

Loads poems, prepares training data, and trains 4 models (1 epoch each):
- Char-level model
- Couplet-level model
- Poem 4-label model
- Poem 1-label model

Saves models to `saved_artifacts/` and exports:
- `data/silver_standard.json`: Complete labeled dataset
- `saved_artifacts/*_train.json`: Training splits
- `saved_artifacts/*_test.json`: Test splits

### 2. Evaluate Models
```bash
# Single evaluation on saved models
python evaluate.py

# Multi-trial statistical evaluation (trains + evaluates N times)
python evaluate.py --trials 100 --output results.json
```

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

```
parallelism-benchmark/
├── pipeline.sh           # Full pipeline script
├── train_utils.py        # Shared training config & functions
├── train_models.py       # Training pipeline
├── evaluate.py           # Evaluation pipeline (single + multi-trial)
├── data_loader.py        # Loads and preprocesses poems
├── datasets.py           # PyTorch dataset classes
├── models.py             # Custom model definitions
├── utils.py              # Data splitting helpers
├── analyze_scenarios.py  # Scenario analysis
├── test_single.py        # Single example testing
├── data/
│   ├── poems/            # Raw poem CSV files by dynasty
│   ├── char_communities.json  # Character semantic groupings
│   └── silver_standard.json   # Exported labeled dataset (generated)
└── saved_artifacts/      # Trained models and data splits (generated)
```

### Configuration

Edit `train_utils.py` to change training parameters:

```python
EPOCHS_CHAR = 1      # Character-level model
EPOCHS_COUPLET = 1   # Couplet-level model  
EPOCHS_POEM4 = 1     # Poem 4-label model
EPOCHS_POEM1 = 1     # Poem 1-label model
```

## Output Files

After running the pipeline:

| File | Description |
|------|-------------|
| `data/silver_standard.json` | Silver Standard dataset with parallelism labels |
| `saved_artifacts/*_train.json` | Training splits for each model type |
| `saved_artifacts/*_test.json` | Test splits for each model type |
| `saved_artifacts/*_model/` | Trained HuggingFace models |
| `evaluation_results.json` | Statistical results (mean ± std from N trials) |

