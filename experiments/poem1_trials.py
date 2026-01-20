"""
Standalone experiment: Train poem1 model for multiple trials.

Each trial trains a poem1 model (regulated vs non-regulated poem classification)
and evaluates its performance. Results are aggregated with statistics.

Usage:
    python experiments/poem1_trials.py --trials 10 --epochs 3
    python experiments/poem1_trials.py --trials 50 --epochs 5
"""

import argparse
import json
import logging
import os
import random
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Suppress transformers warnings before importing
logging.getLogger('transformers').setLevel(logging.ERROR)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from datasets import PoemDataset1Label
from train_utils import (
    get_device, create_tokenizer, train_model, free_memory,
    PRETRAINED_MODEL_NAME
)
from transformers import set_seed, BertForSequenceClassification

# Configuration
MIN_ACCURACY_THRESHOLD = 0.6
TRAIN_RATIO = 0.9
TRAIN_SAMPLES = 9000
TEST_SAMPLES = 1000
MAX_RETRIES_PER_TRIAL = 10


def load_silver_standard(path=None):
    """Load poems from JSON."""
    if path is None:
        path = os.path.join(PROJECT_ROOT, "data", "silver_standard_train.json")
    if not os.path.exists(path):
        print(f"Error: Data not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_poems(poems, seed, train_ratio=0.9):
    """Split poems into train/test sets."""
    poems_copy = list(poems)
    random.seed(seed)
    random.shuffle(poems_copy)
    split_idx = int(len(poems_copy) * train_ratio)
    return poems_copy[:split_idx], poems_copy[split_idx:]


def create_poem1_data(poems):
    """Create poem 1-label data (regulated if inner couplets are parallel)."""
    data = []
    for poem in poems:
        if len(poem["couplets"]) == 4 and len(poem["line_match"]) == 4:
            # Regulated if inner couplets (indices 1,2) are both parallel
            label = 1 if (poem["line_match"][1] == 1 and poem["line_match"][2] == 1) else 0
            data.append({
                "couplets": poem["couplets"],
                "label": label
            })
    return data


def balance_data(data, key="label", max_samples=None):
    """Balance binary data and optionally limit to max_samples."""
    c0 = [x for x in data if x[key] == 0]
    c1 = [x for x in data if x[key] == 1]
    if len(c0) == 0 or len(c1) == 0:
        return data
    
    n = min(len(c0), len(c1))
    
    # If max_samples specified, limit each class to max_samples/2
    if max_samples is not None:
        n = min(n, max_samples // 2)
    
    random.shuffle(c0)
    random.shuffle(c1)
    balanced = c0[:n] + c1[:n]
    random.shuffle(balanced)
    return balanced


def compute_metrics(preds, labels):
    """Compute accuracy, precision, recall, F1."""
    preds = np.array(preds)
    labels = np.array(labels)
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    
    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def create_bert_classifier(model_seed, num_labels=2):
    """Create BERT classifier."""
    set_seed(model_seed)
    torch.manual_seed(model_seed)
    model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=num_labels)
    model.classifier.bias.data.zero_()
    return model


def evaluate_model(model, dataset, device, batch_size=32):
    """Evaluate model on dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            outputs = model(**batch)
            logits = outputs["logits"]
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    return compute_metrics(all_preds, all_labels)


def train_poem1_with_retry(tokenizer, train_data, test_data, epochs, device, data_seed, seed_counter):
    """Train poem1 model with retries on failure.
    
    Returns (metrics, model, updated_seed_counter) or (None, None, updated_seed_counter).
    """
    train_ds = PoemDataset1Label(train_data, tokenizer)
    test_ds = PoemDataset1Label(test_data, tokenizer)
    
    for attempt in range(MAX_RETRIES_PER_TRIAL):
        model_seed = seed_counter
        seed_counter += 1
        
        if attempt > 0:
            print(f"    Retry {attempt}/{MAX_RETRIES_PER_TRIAL-1} (model_seed={model_seed})")
        
        model = create_bert_classifier(model_seed)
        
        set_seed(data_seed)
        model = train_model(model, train_ds, epochs=epochs, device=device,
                           verbose=False, use_balanced_batches=True)
        metrics = evaluate_model(model, test_ds, device)
        
        print(f"    acc={metrics['accuracy']:.4f} prec={metrics['precision']:.4f} "
              f"rec={metrics['recall']:.4f} f1={metrics['f1']:.4f}")
        
        if metrics["accuracy"] >= MIN_ACCURACY_THRESHOLD:
            # Return model for potential best model tracking
            return metrics, model, seed_counter
        
        print(f"    ✗ accuracy < {MIN_ACCURACY_THRESHOLD}")
        del model
        free_memory(device)
    
    print(f"    ✗ Failed after {MAX_RETRIES_PER_TRIAL} attempts")
    return None, None, seed_counter


def run_single_trial(poems, tokenizer, device, seed_counter, data_seed, epochs):
    """Run a single trial: train poem1 model and evaluate.
    
    Returns (result, model, updated_seed_counter) or (None, None, updated_seed_counter).
    """
    print(f"\n  [Trial] data_seed={data_seed}, model_seed_start={seed_counter}")
    random.seed(data_seed)
    
    # Split poems into train/test
    train_poems, test_poems = split_poems(poems, data_seed, TRAIN_RATIO)
    
    # Create poem1 data
    poem1_train_all = create_poem1_data(train_poems)
    poem1_test_all = create_poem1_data(test_poems)
    
    # Balance data
    poem1_train = balance_data(poem1_train_all, key="label", max_samples=TRAIN_SAMPLES)
    poem1_test = balance_data(poem1_test_all, key="label", max_samples=TEST_SAMPLES)
    
    print(f"    Data sizes: train={len(poem1_train)}, test={len(poem1_test)}")
    
    # Train and evaluate
    metrics, model, seed_counter = train_poem1_with_retry(
        tokenizer, poem1_train, poem1_test, epochs, device, data_seed, seed_counter
    )
    
    if metrics is None:
        return None, None, seed_counter
    
    print(f"    ✓ Success")
    
    result = {
        "data_seed": data_seed,
        "metrics": metrics,
    }
    
    return result, model, seed_counter


def run_trials(poems, tokenizer, device, target_trials, epochs, model_seed_start, data_seed_start, max_attempts=500):
    """Run trials until we get target_trials successful ones.
    
    Returns (results, failed_seeds, best_model).
    """
    print(f"\n{'='*60}")
    print(f"Poem1 Experiment")
    print(f"{'='*60}")
    print(f"Target: {target_trials} successful trials")
    print(f"Epochs per trial: {epochs}")
    print(f"Model seed start: {model_seed_start}, Data seed start: {data_seed_start}")
    print(f"Min accuracy threshold: {MIN_ACCURACY_THRESHOLD}")
    print(f"{'='*60}")
    
    successful_results = []
    failed_seeds = []
    current_data_seed = data_seed_start
    seed_counter = model_seed_start
    best_model = None
    best_f1 = -1
    
    while len(successful_results) < target_trials and (current_data_seed - data_seed_start) < max_attempts:
        result, model, seed_counter = run_single_trial(
            poems, tokenizer, device, seed_counter, current_data_seed, epochs
        )
        
        if result is not None:
            successful_results.append(result)
            print(f"\n  Progress: {len(successful_results)}/{target_trials} successful trials")
            
            # Track best model by F1 score
            f1 = result["metrics"]["f1"]
            if f1 > best_f1:
                if best_model is not None:
                    del best_model
                best_f1 = f1
                best_model = model
                print(f"    ★ New best model (F1={f1:.4f})")
            else:
                del model
                free_memory(device)
        else:
            failed_seeds.append(current_data_seed)
        
        current_data_seed += 1
    
    return successful_results, failed_seeds, best_model


def aggregate_results(results):
    """Aggregate results across trials."""
    if not results:
        return {}
    
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    
    aggregated = {
        "num_trials": len(results),
        "trials": results,
        "statistics": {}
    }
    
    for metric in metric_keys:
        values = [r["metrics"][metric] for r in results]
        aggregated["statistics"][metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    
    return aggregated


def save_results(aggregated, output_path):
    """Save results to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)
    print(f"Results saved to {output_path}")


def print_summary(aggregated):
    """Print summary of results."""
    print("\n" + "=" * 70)
    print("POEM1 EXPERIMENT SUMMARY")
    print("=" * 70)
    
    stats = aggregated.get("statistics", {})
    num_trials = aggregated.get("num_trials", 0)
    
    print(f"\nSuccessful Trials: {num_trials}")
    print()
    print(f"{'Metric':<15} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("-" * 70)
    
    for metric in ["accuracy", "precision", "recall", "f1"]:
        if metric in stats:
            s = stats[metric]
            print(f"{metric.capitalize():<15} {s['mean']:>12.4f} {s['std']:>12.4f} {s['min']:>12.4f} {s['max']:>12.4f}")
    
    print("-" * 70)
    
    # Print failed seeds if any
    failed = aggregated.get("failed_seeds", [])
    if failed:
        print(f"\nFailed data seeds ({len(failed)}): {failed[:10]}{'...' if len(failed) > 10 else ''}")
    
    print("=" * 70)


def save_best_model(model, tokenizer, artifacts_dir):
    """Save the best model and tokenizer."""
    model_dir = os.path.join(artifacts_dir, "model")
    tokenizer_dir = os.path.join(artifacts_dir, "tokenizer")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    model.save_pretrained(model_dir)
    print(f"Saved best model to {model_dir}")
    
    tokenizer.save_pretrained(tokenizer_dir)
    print(f"Saved tokenizer to {tokenizer_dir}")


def main():
    default_data = os.path.join(PROJECT_ROOT, "data", "silver_standard_train.json")
    default_output = os.path.join(SCRIPT_DIR, "poem1_results.json")
    artifacts_dir = os.path.join(SCRIPT_DIR, "artifacts_poem1")
    
    parser = argparse.ArgumentParser(description="Poem1 model training experiment")
    parser.add_argument("--trials", type=int, required=True, help="Number of successful trials to run")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs per trial")
    parser.add_argument("--model-seed", type=int, default=1, help="Model initialization seed start (default: 1)")
    parser.add_argument("--data-seed", type=int, default=100, help="Starting data seed (default: 100)")
    parser.add_argument("--data", type=str, default=default_data, help="Path to training data")
    parser.add_argument("--output", type=str, default=default_output, help="Output path for results JSON")
    args = parser.parse_args()
    
    device = get_device()
    print(f"Device: {device}")
    
    # Load data
    poems = load_silver_standard(args.data)
    print(f"Loaded {len(poems)} poems")
    
    # Create tokenizer
    tokenizer = create_tokenizer()
    
    # Run trials
    results, failed, best_model = run_trials(
        poems, tokenizer, device,
        target_trials=args.trials,
        epochs=args.epochs,
        model_seed_start=args.model_seed,
        data_seed_start=args.data_seed
    )
    
    # Save best model
    if best_model is not None:
        save_best_model(best_model, tokenizer, artifacts_dir)
    
    # Aggregate results
    aggregated = aggregate_results(results)
    aggregated["failed_seeds"] = failed
    aggregated["config"] = {
        "trials": args.trials,
        "epochs": args.epochs,
        "model_seed_start": args.model_seed,
        "data_seed_start": args.data_seed,
        "train_samples": TRAIN_SAMPLES,
        "test_samples": TEST_SAMPLES,
        "min_accuracy_threshold": MIN_ACCURACY_THRESHOLD,
    }
    
    # Save results
    save_results(aggregated, args.output)
    
    # Print summary
    print_summary(aggregated)


if __name__ == "__main__":
    main()
