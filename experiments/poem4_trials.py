"""
Standalone experiment: Train poem4 model for multiple trials.

Each trial trains a poem4 model (4-label parallelism classification per couplet)
and evaluates its performance. Results are aggregated with statistics.

Computes two types of metrics:
1. All-couplet metrics: Treats all 4 couplet predictions as individual binary classifications
2. Poem4 → Poem1 induced metrics: If inner couplets (1,2) both parallel → regulated

Usage:
    python experiments/poem4_trials.py --trials 10 --epochs 3
    python experiments/poem4_trials.py --trials 50 --epochs 5
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

from datasets import PoemDataset4Labels
from models import PoemParallelismClassifier
from train_utils import (
    get_device, create_tokenizer, train_model, free_memory,
    PRETRAINED_MODEL_NAME, COUPLET_TOKENS
)
from transformers import set_seed

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


def create_poem4_data(poems):
    """Create poem 4-label data (parallelism label for each couplet).
    
    Also stores the poem1-style label for induced metrics.
    """
    data = []
    for poem in poems:
        if len(poem["couplets"]) == 4 and len(poem["line_match"]) == 4:
            labels = poem["line_match"]  # 4 binary labels
            # Poem1-style label: regulated if inner couplets (1,2) are both parallel
            poem1_label = 1 if (labels[1] == 1 and labels[2] == 1) else 0
            data.append({
                "couplets": poem["couplets"],
                "labels": labels,
                "poem1_label": poem1_label  # For induced metrics
            })
    return data


def balance_poem4_data_per_position(data, target_samples=None, force_target=True, label=None):
    """Balance poem4 data so each couplet position has ~50/50 parallel/non-parallel.
    
    This prevents the model from learning positional priors instead of content.
    
    Strategy: Sample poems such that for EACH position independently, we have
    roughly equal numbers of parallel (1) and non-parallel (0) labels.
    
    Args:
        data: List of poem4 items with "labels" key (list of 4 binary labels)
        target_samples: Target number of samples to return
        force_target: If True (default), sample additional unbalanced data to reach
            target_samples when balanced data is insufficient. If False, return only
            what can be balanced (may be less than target_samples).
        label: Optional label for console output (e.g., "poem4_train")
    
    Returns:
        List of poem4 items
    """
    if len(data) == 0:
        return data
    
    # Group poems by label at each position
    pos_groups = {pos: {0: [], 1: []} for pos in range(4)}
    for idx, item in enumerate(data):
        labels = item["labels"]
        for pos in range(4):
            pos_groups[pos][labels[pos]].append(idx)
    
    # Find minority count at each position
    minority_counts = []
    for pos in range(4):
        n0 = len(pos_groups[pos][0])
        n1 = len(pos_groups[pos][1])
        minority_counts.append(min(n0, n1))
    
    # The bottleneck is the smallest minority count
    bottleneck = min(minority_counts)
    bottleneck_pos = minority_counts.index(bottleneck)
    
    if target_samples is not None:
        samples_per_class = min(bottleneck, target_samples // 2)
    else:
        samples_per_class = bottleneck
    
    # Greedy selection with priority hierarchy:
    # 1. Position 2 (3rd couplet): HIGHEST - maximize non-parallel (label=0)
    # 2. Position 1 (2nd couplet): HIGH - maximize non-parallel (label=0)
    # 3. Position 3 (4th couplet): MEDIUM - maximize parallel (label=1)
    # 4. Position 0 (1st couplet): LOWEST - already balanced, flexible
    
    # For each position, identify which label is the minority
    minority_label = {}
    for pos in range(4):
        n0 = len(pos_groups[pos][0])
        n1 = len(pos_groups[pos][1])
        minority_label[pos] = 0 if n0 <= n1 else 1
    
    # Priority positions in order (highest to lowest priority)
    # Each tuple: (position, target_label_to_maximize)
    priority_order = [
        (2, 0),  # 3rd couplet: maximize non-parallel
        (1, 0),  # 2nd couplet: maximize non-parallel  
        (3, 1),  # 4th couplet: maximize parallel
    ]
    # Position 0 is lowest priority - we don't actively balance it
    
    # Track counts as we select
    pos_counts = {pos: {0: 0, 1: 0} for pos in range(4)}
    selected_indices = set()
    
    def add_poem(idx, selected_indices, pos_counts):
        """Add poem and update counts"""
        selected_indices.add(idx)
        labels = data[idx]["labels"]
        for pos in range(4):
            pos_counts[pos][labels[pos]] += 1
    
    def priority_contribution(idx):
        """Score by how many priority positions this poem helps"""
        labels = data[idx]["labels"]
        score = 0
        for i, (pos, target_label) in enumerate(priority_order):
            if labels[pos] == target_label:
                # Higher weight for higher priority positions
                score += (len(priority_order) - i) * 10
        return score
    
    def can_add_for_priority(idx, pos_counts, samples_per_class, priority_positions):
        """Check limits only for specified priority positions"""
        labels = data[idx]["labels"]
        for pos in priority_positions:
            if pos_counts[pos][labels[pos]] >= samples_per_class:
                return False
        return True
    
    def can_add_strict(idx, pos_counts, samples_per_class):
        """Check if adding this poem would exceed any position's limit"""
        labels = data[idx]["labels"]
        for pos in range(4):
            if pos_counts[pos][labels[pos]] >= samples_per_class:
                return False
        return True
    
    # Phase 1: Poems that help ALL priority positions (2, 1, and 3)
    all_priority_poems = [i for i in range(len(data)) 
                          if all(data[i]["labels"][pos] == target_label 
                                 for pos, target_label in priority_order)]
    random.shuffle(all_priority_poems)
    
    for idx in all_priority_poems:
        priority_positions = [pos for pos, _ in priority_order]
        if can_add_for_priority(idx, pos_counts, samples_per_class, priority_positions):
            add_poem(idx, selected_indices, pos_counts)
        # Check if all priority targets met
        if all(pos_counts[pos][target_label] >= samples_per_class 
               for pos, target_label in priority_order):
            break
    
    # Phase 2: Work through priority positions one by one
    for priority_idx, (target_pos, target_label) in enumerate(priority_order):
        if pos_counts[target_pos][target_label] >= samples_per_class:
            continue  # Already met
        
        # Only check THIS position's limits - allow higher priority positions to overflow
        # This ensures we can still add poems for lower priority positions
        check_positions = [target_pos]
        
        # Find poems that have the target label at this position
        candidate_poems = [i for i in range(len(data)) 
                           if i not in selected_indices and
                           data[i]["labels"][target_pos] == target_label]
        random.shuffle(candidate_poems)
        # Sort by how much they help other priority positions
        candidate_poems.sort(key=priority_contribution, reverse=True)
        
        for idx in candidate_poems:
            if can_add_for_priority(idx, pos_counts, samples_per_class, check_positions):
                add_poem(idx, selected_indices, pos_counts)
            if pos_counts[target_pos][target_label] >= samples_per_class:
                break
    
    # Phase 3: Fill remaining slots with other poems (now apply strict constraints)
    remaining_poems = [i for i in range(len(data)) if i not in selected_indices]
    random.shuffle(remaining_poems)
    remaining_poems.sort(key=priority_contribution, reverse=True)
    
    for idx in remaining_poems:
        if can_add_strict(idx, pos_counts, samples_per_class):
            add_poem(idx, selected_indices, pos_counts)
        
        # Check if we've reached target for ALL positions (both classes)
        all_full = all(
            pos_counts[pos][0] >= samples_per_class and pos_counts[pos][1] >= samples_per_class
            for pos in range(4)
        )
        if all_full:
            break
    
    balanced_count = len(selected_indices)
    
    # If we haven't reached target and force_target is enabled, sample additional poems
    if target_samples is not None and len(selected_indices) < target_samples:
        if force_target:
            shortfall = target_samples - len(selected_indices)
            remaining_indices = [i for i in range(len(data)) if i not in selected_indices]
            random.shuffle(remaining_indices)
            additional = remaining_indices[:shortfall]
            selected_indices.update(additional)
    
    # Calculate final distribution
    final_pos_counts = {pos: {0: 0, 1: 0} for pos in range(4)}
    for idx in selected_indices:
        labels = data[idx]["labels"]
        for pos in range(4):
            final_pos_counts[pos][labels[pos]] += 1
    
    # Calculate max imbalance across positions
    max_imbalance = 0
    for pos in range(4):
        n0, n1 = final_pos_counts[pos][0], final_pos_counts[pos][1]
        total = n0 + n1
        if total > 0:
            imbalance = abs(n0 - n1) / total
            max_imbalance = max(max_imbalance, imbalance)
    
    # Print consolidated summary
    prefix = f"    {label}: " if label else "    "
    if len(selected_indices) > balanced_count:
        extra_info = f", +{len(selected_indices) - balanced_count} unbalanced"
    else:
        extra_info = ""
    print(f"{prefix}{len(selected_indices)} poems (bottleneck: pos{bottleneck_pos}={bottleneck}{extra_info}, imbalance: {max_imbalance*100:.0f}%)")
    
    if force_target and target_samples is not None and len(selected_indices) < target_samples:
        print(f"{prefix}✗ ERROR: Only {len(selected_indices)} poems available, target was {target_samples}")
    
    balanced = [data[i] for i in selected_indices]
    random.shuffle(balanced)
    return balanced


def compute_metrics(preds, labels):
    """Compute accuracy, precision, recall, F1 for binary classification."""
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


def compute_all_couplet_metrics(all_preds, all_labels):
    """Compute metrics treating all couplet predictions as individual classifications.
    
    Args:
        all_preds: List of lists, each inner list has 4 predictions
        all_labels: List of lists, each inner list has 4 labels
    
    Returns:
        Metrics dict
    """
    # Flatten all predictions and labels
    flat_preds = [p for preds in all_preds for p in preds]
    flat_labels = [l for labels in all_labels for l in labels]
    return compute_metrics(flat_preds, flat_labels)


def compute_induced_poem1_metrics(all_preds, poem1_labels):
    """Compute poem4→poem1 induced metrics.
    
    For each poem: if inner couplets (indices 1,2) are both parallel → regulated.
    
    Args:
        all_preds: List of lists, each inner list has 4 predictions
        poem1_labels: List of poem1-style labels (1 if regulated, 0 if not)
    
    Returns:
        Metrics dict
    """
    induced_preds = []
    for preds in all_preds:
        # Induce: if both inner couplets (indices 1, 2) are parallel → regulated
        induced = 1 if (preds[1] == 1 and preds[2] == 1) else 0
        induced_preds.append(induced)
    
    return compute_metrics(induced_preds, poem1_labels)


def create_poem4_classifier(model_seed, tokenizer, num_labels=2):
    """Create PoemParallelismClassifier."""
    set_seed(model_seed)
    torch.manual_seed(model_seed)
    model = PoemParallelismClassifier.create_initial(
        PRETRAINED_MODEL_NAME, tokenizer, COUPLET_TOKENS,
        num_couplets=4, num_labels=num_labels
    )
    model.classifier.bias.data.zero_()
    return model


def evaluate_model(model, dataset, device, batch_size=32):
    """Evaluate poem4 model on dataset.
    
    Returns:
        all_preds: List of 4-element lists (predictions for each couplet)
        all_labels: List of 4-element lists (labels for each couplet)
        poem1_labels: List of poem1-style labels
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch_device = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]  # Shape: (batch, 4)
            outputs = model(**batch_device)
            logits = outputs["logits"]  # Shape: (batch, 4, 2)
            preds = logits.argmax(dim=-1)  # Shape: (batch, 4)
            
            # Convert to lists
            for i in range(preds.size(0)):
                all_preds.append(preds[i].cpu().tolist())
                all_labels.append(labels[i].tolist())
    
    return all_preds, all_labels


def train_poem4_with_retry(tokenizer, train_data, test_data, epochs, device, data_seed, seed_counter):
    """Train poem4 model with retries on failure.
    
    Returns (metrics_dict, model, updated_seed_counter) or (None, None, updated_seed_counter).
    metrics_dict contains both 'all_couplet' and 'poem1_induced' metrics.
    """
    train_ds = PoemDataset4Labels(train_data, tokenizer)
    test_ds = PoemDataset4Labels(test_data, tokenizer)
    
    # Extract poem1 labels for induced metrics
    poem1_labels = [item["poem1_label"] for item in test_data]
    
    for attempt in range(MAX_RETRIES_PER_TRIAL):
        model_seed = seed_counter
        seed_counter += 1
        
        if attempt > 0:
            print(f"    Retry {attempt}/{MAX_RETRIES_PER_TRIAL-1} (model_seed={model_seed})")
        
        model = create_poem4_classifier(model_seed, tokenizer)
        
        set_seed(data_seed)
        # Note: use_balanced_batches=False because poem4 has 4 labels per sample
        # Data is already balanced per-position in balance_poem4_data_per_position()
        model = train_model(model, train_ds, epochs=epochs, device=device,
                           verbose=False, use_balanced_batches=False)
        
        # Evaluate
        all_preds, all_labels = evaluate_model(model, test_ds, device)
        
        # Compute both metrics
        all_couplet_metrics = compute_all_couplet_metrics(all_preds, all_labels)
        poem1_induced_metrics = compute_induced_poem1_metrics(all_preds, poem1_labels)
        
        print(f"    All-couplet: acc={all_couplet_metrics['accuracy']:.4f} "
              f"prec={all_couplet_metrics['precision']:.4f} "
              f"rec={all_couplet_metrics['recall']:.4f} f1={all_couplet_metrics['f1']:.4f}")
        print(f"    Poem4→Poem1: acc={poem1_induced_metrics['accuracy']:.4f} "
              f"prec={poem1_induced_metrics['precision']:.4f} "
              f"rec={poem1_induced_metrics['recall']:.4f} f1={poem1_induced_metrics['f1']:.4f}")
        
        if all_couplet_metrics["accuracy"] >= MIN_ACCURACY_THRESHOLD:
            metrics = {
                "all_couplet": all_couplet_metrics,
                "poem1_induced": poem1_induced_metrics
            }
            return metrics, model, seed_counter
        
        print(f"    ✗ accuracy < {MIN_ACCURACY_THRESHOLD}")
        del model
        free_memory(device)
    
    print(f"    ✗ Failed after {MAX_RETRIES_PER_TRIAL} attempts")
    return None, None, seed_counter


def run_single_trial(poems, tokenizer, device, seed_counter, data_seed, epochs):
    """Run a single trial: train poem4 model and evaluate.
    
    Returns (result, model, updated_seed_counter) or (None, None, updated_seed_counter).
    """
    print(f"\n  [Trial] data_seed={data_seed}, model_seed_start={seed_counter}")
    random.seed(data_seed)
    
    # Split poems into train/test
    train_poems, test_poems = split_poems(poems, data_seed, TRAIN_RATIO)
    
    # Create poem4 data
    poem4_train_all = create_poem4_data(train_poems)
    poem4_test_all = create_poem4_data(test_poems)
    
    # Balance data per-position (each couplet position has ~50/50 parallel/non-parallel)
    poem4_train = balance_poem4_data_per_position(
        poem4_train_all, target_samples=TRAIN_SAMPLES, force_target=True, label="train"
    )
    poem4_test = balance_poem4_data_per_position(
        poem4_test_all, target_samples=TEST_SAMPLES, force_target=False, label="test"
    )
    
    # Train and evaluate
    metrics, model, seed_counter = train_poem4_with_retry(
        tokenizer, poem4_train, poem4_test, epochs, device, data_seed, seed_counter
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
    print(f"Poem4 Experiment")
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
            
            # Track best model by all-couplet F1 score
            f1 = result["metrics"]["all_couplet"]["f1"]
            if f1 > best_f1:
                if best_model is not None:
                    del best_model
                best_f1 = f1
                best_model = model
                print(f"    ★ New best model (all-couplet F1={f1:.4f})")
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
        "statistics": {
            "all_couplet": {},
            "poem1_induced": {}
        }
    }
    
    # Aggregate all-couplet metrics
    for metric in metric_keys:
        values = [r["metrics"]["all_couplet"][metric] for r in results]
        aggregated["statistics"]["all_couplet"][metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    
    # Aggregate poem1-induced metrics
    for metric in metric_keys:
        values = [r["metrics"]["poem1_induced"][metric] for r in results]
        aggregated["statistics"]["poem1_induced"][metric] = {
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
    print("POEM4 EXPERIMENT SUMMARY")
    print("=" * 70)
    
    num_trials = aggregated.get("num_trials", 0)
    print(f"\nSuccessful Trials: {num_trials}")
    
    # All-couplet metrics
    print("\n--- All-Couplet Metrics (4 predictions per poem) ---")
    stats = aggregated.get("statistics", {}).get("all_couplet", {})
    print(f"{'Metric':<15} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("-" * 70)
    for metric in ["accuracy", "precision", "recall", "f1"]:
        if metric in stats:
            s = stats[metric]
            print(f"{metric.capitalize():<15} {s['mean']:>12.4f} {s['std']:>12.4f} {s['min']:>12.4f} {s['max']:>12.4f}")
    
    # Poem1-induced metrics
    print("\n--- Poem4 → Poem1 Induced Metrics (inner couplets → regulated) ---")
    stats = aggregated.get("statistics", {}).get("poem1_induced", {})
    print(f"{'Metric':<15} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("-" * 70)
    for metric in ["accuracy", "precision", "recall", "f1"]:
        if metric in stats:
            s = stats[metric]
            print(f"{metric.capitalize():<15} {s['mean']:>12.4f} {s['std']:>12.4f} {s['min']:>12.4f} {s['max']:>12.4f}")
    
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
    default_output = os.path.join(SCRIPT_DIR, "poem4_results.json")
    artifacts_dir = os.path.join(SCRIPT_DIR, "artifacts_poem4")
    
    parser = argparse.ArgumentParser(description="Poem4 model training experiment")
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
