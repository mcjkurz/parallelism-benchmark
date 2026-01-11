# Parallelism Benchmark

Chinese poetry parallelism detection benchmark across multiple granularity levels.

## Setup

```bash
git clone https://github.com/mcjkurz/parallelism-benchmark
cd parallelism-benchmark
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start: Full Pipeline

Run the complete pipeline with 100 trials for statistical evaluation:

```bash
./scripts/pipeline.sh 100
```

Options:
```bash
./scripts/pipeline.sh                     # Default: 1 trial
./scripts/pipeline.sh 50                  # Run with 50 trials
./scripts/pipeline.sh --skip-prep         # Skip data preparation, only run trials
./scripts/pipeline.sh --train-poems 10000 # Only classify 10k poems (faster)
```

This will:
1. Prepare the Silver Standard dataset using SikuBERT (one-time, slow)
2. Export labeled poems to `data/silver_standard_train.json` and `data/silver_standard_test.json`
3. Run N training+evaluation trials with different seeds
4. Report mean ± std statistics for all metrics
5. Save results to `results/evaluation_results.json`

## Usage (Individual Scripts)

### 1. Prepare Data

```bash
python scripts/prepare_data.py
python scripts/prepare_data.py --train-poems 80000  # 80k training poems
python scripts/prepare_data.py --test-poems 1000    # 1k test poems (default)
python scripts/prepare_data.py --train-only         # Skip test data generation
```

This runs the expensive one-time SikuBERT classification on all couplets.
Saves results to `data/silver_standard_train.json` and `data/silver_standard_test.json`.

### 2. Run Trials

```bash
python scripts/run_trials.py                       # 100 trials (default)
python scripts/run_trials.py --trials 50           # 50 trials with different seeds
python scripts/run_trials.py --train-samples 5000  # Use 5000 training samples per task
python scripts/run_trials.py --test-samples 500    # Use 500 test samples per task
python scripts/run_trials.py --output results/custom  # Custom output directory
```

Each trial trains and evaluates 4 models (char, couplet, poem4, poem1).
The best performing models are saved to `results/models/`.

### 3. Analyze Models

```bash
python scripts/analyze_scenarios.py
```

Runs pairwise comparisons between all 4 models and outputs:
- `results/model_comparison_summary.json`: Accuracy stats and disagreement counts
- `results/model_comparison_full.json`: Full results with all examples

### 4. Generate Figures

```bash
python figures/generate_figures.py
```

This generates publication-quality figures (300 dpi) including:
- Box plots showing distribution
- Summary tables (including LaTeX format)

### 5. Test Single Examples

```bash
python scripts/test_single.py
```

Tests all models on single example inputs.

## Project Structure

```
parallelism-benchmark/
├── scripts/
│   ├── pipeline.sh           # Full pipeline script
│   ├── prepare_data.py       # Data preparation (SikuBERT classification)
│   ├── run_trials.py         # Training and evaluation trials
│   ├── analyze_scenarios.py  # Pairwise model comparison
│   ├── test_single.py        # Single example testing
│   └── test_cuda.py          # CUDA availability check
├── train_utils.py            # Shared training config & functions
├── datasets.py               # PyTorch dataset classes
├── models.py                 # Custom model definitions
├── inference.py              # Shared inference functions
├── figures/
│   ├── generate_figures.py       # Publication figure generation
│   └── model_architectures.py    # Model architecture diagrams
├── data/
│   ├── poems/                    # Raw poem CSV files by dynasty
│   ├── char_communities.json     # Character semantic groupings
│   ├── silver_standard_train.json  # Training dataset (generated)
│   └── silver_standard_test.json   # Test dataset (generated)
└── results/                  # Evaluation results (generated)
    ├── evaluation_results.json
    ├── model_comparison_summary.json
    ├── model_comparison_full.json
    └── models/               # Trained models (generated)
```

### Default Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--trials` | 100 | Number of training/evaluation trials to run |
| `--train-samples` | 9,000 | Training samples per task (char, couplet, poem4, poem1) |
| `--test-samples` | 1,000 | Test samples per task (char, couplet, poem4, poem1) |
| `--train-poems` | all | Maximum poems for training set during data prep |
| `--test-poems` | 1,000 | Number of poems for test set during data prep |
| Epochs | 1 | Training epochs for each model (configurable in `train_utils.py`) |
| Batch size | 8 | Training batch size |
| Learning rate | 2e-5 | AdamW optimizer learning rate |
| Warmup | 5% | Linear warmup during first 5% of training steps |
| Test split | 10% | Portion of data held out for evaluation (train_ratio=0.9) |
| `--model-seed` | 1 | Model initialization seed |
| `--data-seed` | 100 | Starting data seed (increments by 1 for each trial) |

### Token Lengths

Max lengths are calculated precisely for 五言律诗 (pentasyllabic regulated verse):

| Model | Formula | Tokens | Max Length |
|-------|---------|--------|------------|
| Character | `[CLS] c1 [SEP] c2 [SEP]` | 5 | 8 |
| Couplet | `[CLS] + 5 + "，" + 5 + [SEP]` | 13 | 16 |
| Poem4 | `[CLS] + 4×([CPn] + 5 + "，" + 5 + "。") + [SEP]` | 54 | 56 |
| Poem1 | `[CLS] + 4×(5 + "，" + 5 + "。") + [SEP]` | 50 | 52 |

### Configuration

Edit `train_utils.py` to change training parameters:

```python
EPOCHS_CHAR = 1      # Character-level model
EPOCHS_COUPLET = 1   # Couplet-level model  
EPOCHS_POEM4 = 1     # Poem 4-label model
EPOCHS_POEM1 = 1     # Poem 1-label model
```

Use `--train-samples` and `--test-samples` arguments in `scripts/run_trials.py` to change sample counts per task.

## Output Files

| File | Description |
|------|-------------|
| `data/silver_standard_train.json` | Training dataset with parallelism labels |
| `data/silver_standard_test.json` | Test dataset (non-overlapping with training) |
| `results/evaluation_results.json` | Trial statistics (mean ± std) |
| `results/model_comparison_summary.json` | Pairwise model comparison stats |
| `results/model_comparison_full.json` | Full comparison with all examples |
| `results/models/` | Best performing models (4 models + tokenizer) |
| `figures/*.png` | Publication-quality figures (300 dpi) |
