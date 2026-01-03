"""
Run training and evaluation trials for parallelism detection models.

This script loads the pre-classified data from data/silver_standard.json
(created by prepare_data.py) and runs one or more trials with different
random seeds.

Usage:
    python run_trials.py                    # Single trial (seed=42)
    python run_trials.py --trials 100       # 100 trials with different seeds
    python run_trials.py --training-samples 5000  # Use 5000 training examples per task
    python run_trials.py --output results.json  # Custom output file
"""

import argparse
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import CharPairDataset, CoupletDataset, PoemDataset4Labels, PoemDataset1Label
from models import PoemParallelismClassifier
from utils import create_training_datasets, split_raw_data
from train_utils import (
    get_device, create_tokenizer, train_all_models, free_memory
)


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_standard(model, dataset, device, batch_size=32):
    """Evaluate model accuracy on a dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            outputs = model(**batch)
            logits = outputs["logits"]

            if logits.dim() == 3:
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.numel()
            else:
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

    return correct / total if total > 0 else 0.0


def evaluate_char_induced_couplet_accuracy(char_model, raw_couplet_data, tokenizer, device):
    """Evaluate couplet accuracy induced by character-level predictions."""
    char_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for item in raw_couplet_data:
            l1, l2 = item["couplet"]
            true_label = item["label"]

            pairs = list(zip(l1, l2))
            if not pairs:
                continue

            encoded = tokenizer(
                [p[0] for p in pairs],
                [p[1] for p in pairs],
                truncation=True, padding=True, max_length=16, return_tensors="pt"
            ).to(device)

            logits = char_model(**encoded).logits
            char_preds = logits.argmax(dim=-1)

            predicted_label = 1 if char_preds.sum().item() >= 3 else 0

            if predicted_label == true_label:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


def evaluate_couplet_induced_poem_accuracy(couplet_model, raw_poem_data, tokenizer, device):
    """Evaluate poem accuracy induced by couplet-level predictions."""
    couplet_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for item in raw_poem_data:
            couplets = item["couplets"]
            true_label = item["label"]

            inner_couplets = [couplets[1], couplets[2]]
            couplet_strs = [c[0] + "，" + c[1] for c in inner_couplets]

            encoded = tokenizer(
                couplet_strs,
                truncation=True, padding=True, max_length=64, return_tensors="pt"
            ).to(device)

            logits = couplet_model(**encoded).logits
            preds = logits.argmax(dim=-1)

            predicted_label = 1 if (preds == 1).all().item() else 0

            if predicted_label == true_label:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


def evaluate_poem4_inner_accuracy(model, dataset, device):
    """Evaluate Poem4 model accuracy on inner couplets only."""
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            logits = model(**batch)["logits"]
            preds = logits.argmax(dim=-1)

            inner_preds = preds[:, 1:3]
            inner_labels = labels[:, 1:3]

            correct += (inner_preds == inner_labels).sum().item()
            total += inner_labels.numel()

    return correct / total if total > 0 else 0.0


def evaluate_poem1_inner_accuracy(poem1_model, raw_poem_data, tokenizer, device):
    """Evaluate Poem1 model accuracy on inner couplet prediction."""
    poem1_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for item in raw_poem_data:
            couplets = item["couplets"]

            if "line_match" in item:
                labels = item["line_match"]
                cp2_label = labels[1]
                cp3_label = labels[2]
                true_inner_parallel = 1 if (cp2_label == 1 and cp3_label == 1) else 0
            else:
                true_inner_parallel = item["label"]

            text = ""
            for l1, l2 in couplets:
                text += l1 + "，" + l2 + "。"

            encoded = tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)

            logits = poem1_model(**encoded).logits
            model_pred = logits.argmax(dim=-1).item()

            pred_inner_parallel = 1 if model_pred == 1 else 0

            if pred_inner_parallel == true_inner_parallel:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


# =============================================================================
# Trial Execution
# =============================================================================

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_silver_standard(path="data/silver_standard.json"):
    """Load the pre-classified poems from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        poems = json.load(f)
    print(f"Loaded {len(poems)} pre-classified poems from {path}")
    return poems


def run_single_trial(poems, seed, tokenizer, device, training_samples=10000, verbose=True):
    """Run a single training and evaluation trial with the given seed."""
    set_seed(seed)
    
    # Create and split datasets with this seed
    training_data_characters, training_data_couplets, training_data_poems_4labels, training_data_poems_1label = \
        create_training_datasets(poems, max_samples=training_samples, seed=seed)
    
    char_train_raw, char_test_raw = split_raw_data(training_data_characters, seed=seed)
    coup_train_raw, coup_test_raw = split_raw_data(training_data_couplets, seed=seed)
    poem4_train_raw, poem4_test_raw = split_raw_data(training_data_poems_4labels, seed=seed)
    poem1_train_raw, poem1_test_raw = split_raw_data(training_data_poems_1label, seed=seed)
    
    # Create PyTorch datasets
    char_train_ds = CharPairDataset(char_train_raw, tokenizer)
    coup_train_ds = CoupletDataset(coup_train_raw, tokenizer)
    poem4_train_ds = PoemDataset4Labels(poem4_train_raw, tokenizer)
    poem1_train_ds = PoemDataset1Label(poem1_train_raw, tokenizer)
    
    char_test_ds = CharPairDataset(char_test_raw, tokenizer)
    coup_test_ds = CoupletDataset(coup_test_raw, tokenizer)
    poem4_test_ds = PoemDataset4Labels(poem4_test_raw, tokenizer)
    poem1_test_ds = PoemDataset1Label(poem1_test_raw, tokenizer)
    
    # Train all models
    char_model, coup_model, poem4_model, poem1_model = train_all_models(
        char_train_ds, coup_train_ds, poem4_train_ds, poem1_train_ds,
        tokenizer, device=device, verbose=verbose
    )
    
    # Evaluate all models
    results = {
        "seed": seed,
        "char_acc": evaluate_standard(char_model, char_test_ds, device),
        "coup_acc": evaluate_standard(coup_model, coup_test_ds, device),
        "poem4_overall_acc": evaluate_standard(poem4_model, poem4_test_ds, device),
        "poem4_inner_acc": evaluate_poem4_inner_accuracy(poem4_model, poem4_test_ds, device),
        "poem1_acc": evaluate_standard(poem1_model, poem1_test_ds, device),
        "poem1_inner_acc": evaluate_poem1_inner_accuracy(poem1_model, poem1_test_raw, tokenizer, device),
        "char_induced_coup_acc": evaluate_char_induced_couplet_accuracy(char_model, coup_test_raw, tokenizer, device),
        "coup_induced_poem_acc": evaluate_couplet_induced_poem_accuracy(coup_model, poem1_test_raw, tokenizer, device),
    }
    
    # Clean up models (don't save them)
    del char_model, coup_model, poem4_model, poem1_model
    free_memory(device)
    
    return results


def compute_statistics(all_results):
    """Compute mean, std, min, max for each metric across trials."""
    # Get all metric keys (excluding 'seed')
    metric_keys = [k for k in all_results[0].keys() if k != "seed"]
    
    statistics = {}
    for key in metric_keys:
        values = [r[key] for r in all_results]
        values_np = np.array(values)
        statistics[key] = {
            "mean": float(np.mean(values_np)),
            "std": float(np.std(values_np)),
            "min": float(np.min(values_np)),
            "max": float(np.max(values_np)),
        }
    
    return statistics


def run_trials(num_trials, output_file, silver_path="data/silver_standard.json", training_samples=10000):
    """Run training and evaluation trials."""
    device = get_device()
    print(f"Using device: {device}")
    
    print()
    print("=" * 60)
    print(f"Running {num_trials} trial(s)")
    print(f"Target training samples per task: {training_samples}")
    print("=" * 60)
    print()
    
    # Load pre-classified poems
    poems = load_silver_standard(silver_path)
    
    # Initialize tokenizer
    tokenizer = create_tokenizer()
    
    # Run trials
    all_results = []
    for trial in range(num_trials):
        seed = 42 + trial
        print(f"\n--- Trial {trial + 1}/{num_trials} (seed={seed}) ---")
        
        verbose = (num_trials == 1)  # Only show progress bars for single trial
        trial_results = run_single_trial(poems, seed, tokenizer, device, training_samples=training_samples, verbose=verbose)
        all_results.append(trial_results)
        
        # Print summary for this trial
        print(f"  Char: {trial_results['char_acc']:.4f}  "
              f"Couplet: {trial_results['coup_acc']:.4f}  "
              f"Poem4: {trial_results['poem4_overall_acc']:.4f}  "
              f"Poem1: {trial_results['poem1_acc']:.4f}")
    
    # Compute and display statistics
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    
    if num_trials > 1:
        statistics = compute_statistics(all_results)
        
        for key, stats in statistics.items():
            print(f"{key}:")
            print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print()
        
        # Save results
        output_data = {
            "num_trials": num_trials,
            "statistics": statistics,
            "trials": all_results,
        }
    else:
        # Single trial - just print and save the results
        for key, value in all_results[0].items():
            if key != "seed":
                print(f"{key}: {value:.4f}")
        print()
        
        output_data = {
            "num_trials": 1,
            "trials": all_results,
        }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run parallelism model training and evaluation trials"
    )
    parser.add_argument(
        "--trials", type=int, default=1,
        help="Number of trials to run (default: 1)"
    )
    parser.add_argument(
        "--output", type=str, default="evaluation_results.json",
        help="Output file for results (default: evaluation_results.json)"
    )
    parser.add_argument(
        "--data", type=str, default="data/silver_standard.json",
        help="Path to silver standard data (default: data/silver_standard.json)"
    )
    parser.add_argument(
        "--training-samples", type=int, default=10000,
        help="Target number of training samples per task (default: 10000)"
    )
    args = parser.parse_args()
    
    run_trials(
        num_trials=args.trials,
        output_file=args.output,
        silver_path=args.data,
        training_samples=args.training_samples,
    )


if __name__ == "__main__":
    main()

