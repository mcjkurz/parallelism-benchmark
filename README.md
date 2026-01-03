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
./pipeline.sh 100
```

Options:
```bash
./pipeline.sh                    # Default: 1 trial
./pipeline.sh 50                 # Run with 50 trials
./pipeline.sh --skip-prep        # Skip data preparation, only run trials
./pipeline.sh --max-poems 10000  # Only classify 10k poems (faster)
```

This will:
1. Prepare the Silver Standard dataset using SikuBERT (one-time, slow)
2. Export labeled poems to `data/silver_standard.json`
3. Run N training+evaluation trials with different seeds
4. Report mean ± std statistics for all metrics

## Usage (Individual Scripts)

### 1. Prepare Data

```bash
python prepare_data.py
python prepare_data.py --max-poems 10000  # Faster: only classify 10k poems
```

This runs the expensive one-time SikuBERT classification on all couplets.
Saves results to `data/silver_standard.json`.

### 2. Run Trials

```bash
python run_trials.py                      # Single trial (seed=42)
python run_trials.py --trials 100         # 100 trials with different seeds
python run_trials.py --training-samples 5000  # Use 5000 training samples per task
python run_trials.py --output results.json  # Custom output file
```

Each trial:
- Samples training data from the silver standard
- Trains 4 models (char, couplet, poem-4label, poem-1label)
- Evaluates on held-out test set

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
├── prepare_data.py       # Data preparation (SikuBERT classification)
├── run_trials.py         # Training and evaluation trials
├── train_utils.py        # Shared training config & functions
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

Edit `run_trials.py --training-samples` to change target training samples per task (default: 10000).

## Output Files

After running the pipeline:

| File | Description |
|------|-------------|
| `data/silver_standard.json` | Silver Standard dataset with parallelism labels |
| `evaluation_results.json` | Statistical results (mean ± std from N trials) |

