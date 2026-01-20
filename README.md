# Parallelism Benchmark

Benchmark for detecting parallelism (對仗) in Classical Chinese poetry at multiple granularity levels. Uses [SikuBERT](https://huggingface.co/SIKU-BERT/sikubert) as the base model.

**Models:** Character → Couplet → Poem (4-label) → Poem (binary)

## Setup

```bash
git clone https://github.com/mcjkurz/parallelism-benchmark
cd parallelism-benchmark
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

Run the full pipeline with 100 trials:

```bash
./scripts/pipeline.sh 100
```

Options:
```bash
./scripts/pipeline.sh                     # Default: 1 trial
./scripts/pipeline.sh 50                  # 50 trials
./scripts/pipeline.sh --skip-prep         # Skip data prep (reuse existing data)
./scripts/pipeline.sh --train-poems 10000 # Limit to 10k poems (faster)
```

Pipeline steps:
1. Prepare Silver Standard dataset via SikuBERT classification (one-time)
2. Run N training+evaluation trials with different data seeds
3. Report mean ± std for all metrics
4. Save results to `results/evaluation_results.json`

## Individual Scripts

**Prepare Data** — One-time SikuBERT classification:
```bash
python scripts/prepare_data.py
python scripts/prepare_data.py --train-poems 80000  # Limit training poems
python scripts/prepare_data.py --test-poems 1000    # 1k test poems (default)
```

**Run Trials** — Train and evaluate all 4 models:
```bash
python scripts/run_trials.py --trials 100           # 100 trials (default)
python scripts/run_trials.py --train-samples 5000   # Samples per task
python scripts/run_trials.py --output results/custom
```

**Analyze Models** — Pairwise model comparisons:
```bash
python scripts/analyze_scenarios.py
```

**Generate Figures** — Publication-quality plots (300 dpi):
```bash
python figures/generate_figures.py
```

**Test Single Examples:**
```bash
python scripts/test_single.py
```

## Project Structure

```
parallelism-benchmark/
├── scripts/
│   ├── pipeline.sh           # Full pipeline
│   ├── prepare_data.py       # Data preparation
│   ├── run_trials.py         # Training trials
│   ├── analyze_scenarios.py  # Model comparison
│   └── test_single.py        # Single example testing
├── train_utils.py            # Training config & functions
├── datasets.py               # PyTorch datasets
├── models.py                 # Model definitions
├── inference.py              # Inference functions
├── figures/                  # Figure generation
├── data/
│   ├── poems/                # Raw poem CSVs by dynasty
│   └── *.json                # Generated datasets
└── results/                  # Generated outputs
    ├── evaluation_results.json
    └── models/               # Trained models
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--trials` | 100 | Number of trials |
| `--train-samples` | 9,000 | Training samples per task |
| `--test-samples` | 1,000 | Test samples per task |
| Epochs | 1 | Per model (edit `train_utils.py`) |
| Batch size | 8 | Training batch size |
| Learning rate | 2e-5 | AdamW optimizer |
| Warmup | 5% | Warmup steps |
| Weight decay | 0.001 | L2 regularization |

## Token Lengths

For 五言律詩 (pentasyllabic regulated verse):

| Model | Input Format | Max Length |
|-------|--------------|------------|
| Character | `[CLS] c1 [SEP] c2 [SEP]` | 8 |
| Couplet | `[CLS] 5chars，5chars [SEP]` | 16 |
| Poem4 | `[CLS] [CP1] line1。... [CP4] line4。[SEP]` | 56 |
| Poem1 | `[CLS] line1。line2。line3。line4。[SEP]` | 52 |

## Output Files

| File | Description |
|------|-------------|
| `data/silver_standard_*.json` | Train/test datasets |
| `results/evaluation_results.json` | Trial statistics (mean ± std) |
| `results/models/` | Best models (4 models + tokenizer) |
| `figures/*.png` | Publication figures (300 dpi) |
