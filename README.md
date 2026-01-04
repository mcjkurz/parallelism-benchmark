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
```

## Quick Start: Full Pipeline

Run the complete pipeline with 100 trials for statistical evaluation:

```bash
./scripts/pipeline.sh 100
```

Options:
```bash
./scripts/pipeline.sh                    # Default: 1 trial
./scripts/pipeline.sh 50                 # Run with 50 trials
./scripts/pipeline.sh --skip-prep        # Skip data preparation, only run trials
./scripts/pipeline.sh --max-poems 10000  # Only classify 10k poems (faster)
```

This will:
1. Prepare the Silver Standard dataset using SikuBERT (one-time, slow)
2. Export labeled poems to `data/silver_standard.json`
3. Run N training+evaluation trials with different seeds
4. Report mean ± std statistics for all metrics
5. Save results to `results/evaluation_results.json`

## Usage (Individual Scripts)

### 1. Prepare Data

```bash
python scripts/prepare_data.py
python scripts/prepare_data.py --max-poems 10000  # Faster: only classify 10k poems
```

This runs the expensive one-time SikuBERT classification on all couplets.
Saves results to `data/silver_standard.json`.

### 2. Run Trials

```bash
python scripts/run_trials.py                      # Single trial (seed=42)
python scripts/run_trials.py --trials 100         # 100 trials with different seeds
python scripts/run_trials.py --training-samples 5000  # Use 5000 training samples per task
python scripts/run_trials.py --output results/custom.json  # Custom output file
```

Each trial trains and evaluates 4 models (char, couplet, poem4, poem1).
The best performing models are saved to `saved_artifacts/`.

### 3. Analyze Models

```bash
python scripts/analyze_scenarios.py
```

Runs pairwise comparisons between all 4 models and outputs:
- `results/model_comparison_summary.json`: Accuracy stats and disagreement counts
- `results/model_comparison_full.json`: Full results with all examples

### 4. Generate Figures

Open and run the Jupyter notebook:
```bash
jupyter notebook figures/accuracy_figures.ipynb
```

This generates publication-quality figures (300 dpi) including:
- Bar charts with error bars
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
├── utils.py                  # Data splitting helpers
├── data/
│   ├── poems/                # Raw poem CSV files by dynasty
│   ├── char_communities.json # Character semantic groupings
│   └── silver_standard.json  # Exported labeled dataset (generated)
├── results/                  # Evaluation results (generated)
│   ├── evaluation_results.json
│   ├── model_comparison_summary.json
│   └── model_comparison_full.json
├── figures/                  # Publication figures
│   └── accuracy_figures.ipynb  # Jupyter notebook for generating figures
└── saved_artifacts/          # Trained models and data splits (generated)
```

### Default Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--trials` | 1 | Number of training/evaluation trials to run |
| `--training-samples` | 10,000 | Target training samples per task (char, couplet, poem4, poem1) |
| `--max-poems` | all | Maximum poems to classify with SikuBERT during data prep |
| Epochs | 1 | Training epochs for each model (configurable in `train_utils.py`) |
| Batch size | 8 | Training batch size |
| Learning rate | 2e-5 | AdamW optimizer learning rate |
| Warmup | 10% | Linear warmup during first 10% of training steps |
| Test split | 10% | Portion of data held out for evaluation (train_ratio=0.9) |
| Random seed | 42 | Base seed (increments by 1 for each trial) |

### Configuration

Edit `train_utils.py` to change training parameters:

```python
EPOCHS_CHAR = 1      # Character-level model
EPOCHS_COUPLET = 1   # Couplet-level model  
EPOCHS_POEM4 = 1     # Poem 4-label model
EPOCHS_POEM1 = 1     # Poem 1-label model
```

Edit `scripts/run_trials.py --training-samples` to change target training samples per task (default: 10,000).

## Output Files

| File | Description |
|------|-------------|
| `data/silver_standard.json` | Silver Standard dataset with parallelism labels |
| `results/evaluation_results.json` | Trial statistics (mean ± std) |
| `results/model_comparison_summary.json` | Pairwise model comparison stats |
| `results/model_comparison_full.json` | Full comparison with all examples |
| `saved_artifacts/` | Best performing models (4 models + tokenizer + test data) |
| `figures/*.png` | Publication-quality figures (300 dpi) |
