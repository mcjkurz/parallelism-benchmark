"""
Run training and evaluation trials for parallelism detection models.

This script loads the pre-classified data from data/silver_standard.json
(created by prepare_data.py) and runs one or more trials with different
random seeds.

Usage:
    python scripts/run_trials.py                    # Single trial (seed=42)
    python scripts/run_trials.py --trials 100       # 100 trials with different seeds
    python scripts/run_trials.py --training-samples 5000  # Use 5000 training examples
    python scripts/run_trials.py --output results/custom.json  # Custom output file
"""

import argparse
import json
import os
import pickle
import random
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Add parent directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from datasets import CharPairDataset, CoupletDataset, PoemDataset4Labels, PoemDataset1Label
from models import PoemParallelismClassifier
from utils import create_training_datasets, split_raw_data
from train_utils import (
    get_device, create_tokenizer, train_all_models, free_memory, TrainingFailedError
)
from inference import predict_char_level, predict_couplet_level, predict_poem4_level, predict_poem1_level


# =============================================================================
# Evaluation Functions with Precision/Recall
# =============================================================================

def evaluate_with_metrics(model, dataset, device, batch_size=32):
    """Evaluate model and return accuracy, precision, recall, and F1."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            outputs = model(**batch)
            logits = outputs["logits"]

            if logits.dim() == 3:
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.view(-1).cpu().tolist())
                all_labels.extend(labels.view(-1).cpu().tolist())
            else:
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

    return compute_metrics(all_preds, all_labels)


def compute_metrics(preds, labels):
    """Compute accuracy, precision, recall, and F1 from predictions and labels."""
    preds = np.array(preds)
    labels = np.array(labels)
    
    # Basic counts
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
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def evaluate_char_induced_couplet(char_model, raw_couplet_data, tokenizer, device):
    """Evaluate couplet accuracy induced by character-level predictions."""
    char_model.eval()
    all_preds = []
    all_labels = []

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
            
            all_preds.append(predicted_label)
            all_labels.append(true_label)

    return compute_metrics(all_preds, all_labels)


def evaluate_couplet_induced_poem(couplet_model, raw_poem_data, tokenizer, device):
    """Evaluate poem accuracy induced by couplet-level predictions."""
    couplet_model.eval()
    all_preds = []
    all_labels = []

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
            
            all_preds.append(predicted_label)
            all_labels.append(true_label)

    return compute_metrics(all_preds, all_labels)


def evaluate_poem4_inner(model, dataset, device):
    """Evaluate Poem4 model on inner couplets only."""
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            logits = model(**batch)["logits"]
            preds = logits.argmax(dim=-1)

            inner_preds = preds[:, 1:3]
            inner_labels = labels[:, 1:3]

            all_preds.extend(inner_preds.view(-1).cpu().tolist())
            all_labels.extend(inner_labels.view(-1).cpu().tolist())

    return compute_metrics(all_preds, all_labels)


# =============================================================================
# Model Comparison (per-seed analysis)
# =============================================================================

def row_to_example(row):
    """Convert a result dict to a simple example dict."""
    lines = row['full_text']
    return {
        "poem": [f"{lines[k]}，{lines[k+1]}" for k in range(0, 8, 2)],
        "couplet_idx": int(row['couplet_idx']),
        "target": f"{row['l1']}，{row['l2']}",
        "truth": int(row['truth']),
        "predictions": {
            "char": int(row['pred_char']),
            "couplet": int(row['pred_coup']),
            "poem4": int(row['pred_poem4']),
            "poem1": int(row['pred_poem1_implicit']) if row['couplet_idx'] in [1, 2] else None
        }
    }


def pairwise_comparison(results, model_a, model_b, col_a, col_b, inner_only=False):
    """Compare two models and return counts + examples."""
    if inner_only:
        data = [r for r in results if r['couplet_idx'] in [1, 2]]
    else:
        data = results
    
    total = len(data)
    
    both_correct = []
    both_wrong = []
    a_only = []
    b_only = []
    
    for r in data:
        a_correct = r[col_a] == r['truth']
        b_correct = r[col_b] == r['truth']
        
        if a_correct and b_correct:
            both_correct.append(r)
        elif not a_correct and not b_correct:
            both_wrong.append(r)
        elif a_correct and not b_correct:
            a_only.append(r)
        else:
            b_only.append(r)
    
    return {
        "total": total,
        "both_correct": len(both_correct),
        "both_wrong": len(both_wrong),
        f"{model_a}_only_correct": len(a_only),
        f"{model_b}_only_correct": len(b_only),
        "examples": {
            f"{model_a}_only_correct": [row_to_example(r) for r in a_only],
            f"{model_b}_only_correct": [row_to_example(r) for r in b_only]
        }
    }


def generate_model_comparison(poem4_test_raw, models, tokenizer, device):
    """Generate full model comparison data for a single trial."""
    char_model = models["char_model"]
    coup_model = models["coup_model"]
    poem4_model = models["poem4_model"]
    poem1_model = models["poem1_model"]
    
    char_model.eval()
    coup_model.eval()
    poem4_model.eval()
    poem1_model.eval()
    
    results = []
    
    for idx, item in enumerate(poem4_test_raw):
        couplets = item["couplets"]
        labels = item["labels"]
        dynasty = item.get("dynasty", "unknown")

        poem4_preds = predict_poem4_level(couplets, poem4_model, tokenizer, device)
        poem1_pred = predict_poem1_level(couplets, poem1_model, tokenizer, device)

        full_text_lines = []
        for c in couplets:
            full_text_lines.extend([c[0], c[1]])

        for i in range(4):
            l1, l2 = couplets[i]
            truth = labels[i]

            coup_pred = predict_couplet_level(l1, l2, coup_model, tokenizer, device)
            char_cons, char_dets = predict_char_level(l1, l2, char_model, tokenizer, device)

            poem1_implicit = -1
            if i in [1, 2]:
                poem1_implicit = 1 if poem1_pred == 1 else 0

            results.append({
                "poem_id": idx,
                "dynasty": dynasty,
                "full_text": full_text_lines,
                "couplet_idx": i,
                "l1": l1,
                "l2": l2,
                "truth": truth,
                "truth_full": labels,
                "pred_char": char_cons,
                "pred_char_details": char_dets,
                "pred_coup": coup_pred,
                "pred_poem4": poem4_preds[i],
                "pred_poem4_full": poem4_preds,
                "pred_poem1_global": poem1_pred,
                "pred_poem1_implicit": poem1_implicit
            })

    # Compute summary and pairwise comparisons
    inner_results = [r for r in results if r['couplet_idx'] in [1, 2]]
    
    accuracy = {
        "char": float(np.mean([r['pred_char'] == r['truth'] for r in results])),
        "couplet": float(np.mean([r['pred_coup'] == r['truth'] for r in results])),
        "poem4": float(np.mean([r['pred_poem4'] == r['truth'] for r in results])),
        "poem1": float(np.mean([r['pred_poem1_implicit'] == r['truth'] for r in inner_results]))
    }
    
    pairwise = {
        "char_vs_couplet": pairwise_comparison(results, "char", "couplet", "pred_char", "pred_coup"),
        "char_vs_poem4": pairwise_comparison(results, "char", "poem4", "pred_char", "pred_poem4"),
        "char_vs_poem1": pairwise_comparison(results, "char", "poem1", "pred_char", "pred_poem1_implicit", inner_only=True),
        "couplet_vs_poem4": pairwise_comparison(results, "couplet", "poem4", "pred_coup", "pred_poem4"),
        "couplet_vs_poem1": pairwise_comparison(results, "couplet", "poem1", "pred_coup", "pred_poem1_implicit", inner_only=True),
        "poem4_vs_poem1": pairwise_comparison(results, "poem4", "poem1", "pred_poem4", "pred_poem1_implicit", inner_only=True),
    }
    
    return {
        "summary": {
            "total_couplets": len(results),
            "inner_couplets": len(inner_results),
            "accuracy": accuracy
        },
        "pairwise": pairwise
    }


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


def load_silver_standard(path=None):
    """Load the pre-classified poems from JSON."""
    if path is None:
        path = os.path.join(PROJECT_ROOT, "data", "silver_standard.json")
    
    if not os.path.exists(path):
        print(f"Error: Silver standard data not found: {path}")
        print()
        print("Please run data preparation first:")
        print("  python scripts/prepare_data.py")
        print()
        print("Or run the full pipeline:")
        print("  ./scripts/pipeline.sh")
        sys.exit(1)
    
    with open(path, "r", encoding="utf-8") as f:
        poems = json.load(f)
    print(f"Loaded {len(poems)} pre-classified poems from {path}")
    return poems


def run_single_trial(poems, seed, tokenizer, device, training_samples=10000, verbose=True):
    """Run a single training and evaluation trial with the given seed.
    
    Args:
        poems: List of poem data
        seed: Random seed for reproducibility
        tokenizer: The tokenizer to use
        device: Device to train on
        training_samples: Target number of training samples
        verbose: Whether to show progress bars
    
    Returns:
        Tuple of (results_dict, models_dict, test_data_dict)
        
    Raises:
        TrainingFailedError: If any model accuracy is below MIN_ACCURACY_THRESHOLD
    """
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
    
    # Evaluate all models with full metrics
    char_metrics = evaluate_with_metrics(char_model, char_test_ds, device)
    coup_metrics = evaluate_with_metrics(coup_model, coup_test_ds, device)
    poem4_metrics = evaluate_with_metrics(poem4_model, poem4_test_ds, device)
    poem4_inner_metrics = evaluate_poem4_inner(poem4_model, poem4_test_ds, device)
    poem1_metrics = evaluate_with_metrics(poem1_model, poem1_test_ds, device)
    
    # Check if any accuracy is below threshold - if so, training failed
    from train_utils import MIN_ACCURACY_THRESHOLD
    accuracies = {
        "char": char_metrics["accuracy"],
        "couplet": coup_metrics["accuracy"],
        "poem4": poem4_metrics["accuracy"],
        "poem1": poem1_metrics["accuracy"],
    }
    for name, acc in accuracies.items():
        if acc < MIN_ACCURACY_THRESHOLD:
            raise TrainingFailedError(
                f"{name} accuracy {acc:.3f} < {MIN_ACCURACY_THRESHOLD}"
            )
    
    char_induced_coup_metrics = evaluate_char_induced_couplet(char_model, coup_test_raw, tokenizer, device)
    coup_induced_poem_metrics = evaluate_couplet_induced_poem(coup_model, poem1_test_raw, tokenizer, device)
    
    results = {
        "seed": seed,
        "char": char_metrics,
        "couplet": coup_metrics,
        "poem4_overall": poem4_metrics,
        "poem4_inner": poem4_inner_metrics,
        "poem1": poem1_metrics,
        "char_induced_couplet": char_induced_coup_metrics,
        "couplet_induced_poem": coup_induced_poem_metrics,
    }
    
    models = {
        "char_model": char_model,
        "coup_model": coup_model,
        "poem4_model": poem4_model,
        "poem1_model": poem1_model,
    }
    test_data = {
        "poem4_test_raw": poem4_test_raw,
        "coup_test_raw": coup_test_raw,
        "poem1_test_raw": poem1_test_raw,
    }
    
    return results, models, test_data


def compute_statistics(all_results):
    """Compute mean, std, min, max for each metric across trials."""
    # Flatten nested metrics
    statistics = {}
    
    for key in all_results[0].keys():
        if key == "seed":
            continue
        
        if isinstance(all_results[0][key], dict):
            # Nested metrics (like char, couplet, etc.)
            for metric_name in all_results[0][key].keys():
                if metric_name in ["tp", "fp", "fn", "tn"]:
                    continue  # Skip count metrics for statistics
                values = [r[key][metric_name] for r in all_results]
                values_np = np.array(values)
                stat_key = f"{key}_{metric_name}"
                statistics[stat_key] = {
                    "mean": float(np.mean(values_np)),
                    "std": float(np.std(values_np)),
                    "min": float(np.min(values_np)),
                    "max": float(np.max(values_np)),
                }
        else:
            # Simple metric
            values = [r[key] for r in all_results]
            values_np = np.array(values)
            statistics[key] = {
                "mean": float(np.mean(values_np)),
                "std": float(np.std(values_np)),
                "min": float(np.min(values_np)),
                "max": float(np.max(values_np)),
            }
    
    return statistics


def compute_aggregate_score(results):
    """Compute an aggregate score for a trial to determine the best one.
    
    Uses the average of the four primary model accuracies.
    """
    return (
        results["char"]["accuracy"] +
        results["couplet"]["accuracy"] +
        results["poem4_overall"]["accuracy"] +
        results["poem1"]["accuracy"]
    ) / 4


def save_best_models(models, test_data, tokenizer, output_dir=None):
    """Save the best performing models and associated data.
    
    Args:
        models: Dict with char_model, coup_model, poem4_model, poem1_model
        test_data: Dict with poem4_test_raw
        tokenizer: The tokenizer to save
        output_dir: Directory to save artifacts to
    """
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "saved_artifacts")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save tokenizer
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    
    # Save models
    models["char_model"].save_pretrained(os.path.join(output_dir, "char_model"))
    models["coup_model"].save_pretrained(os.path.join(output_dir, "coup_model"))
    models["poem4_model"].save_pretrained(os.path.join(output_dir, "poem4_model"))
    models["poem1_model"].save_pretrained(os.path.join(output_dir, "poem1_model"))
    
    # Save test data for analysis
    with open(os.path.join(output_dir, "poem4_test_raw.pkl"), "wb") as f:
        pickle.dump(test_data["poem4_test_raw"], f)


def save_model_comparison(comparison_data, seed, output_dir=None):
    """Save model comparison data for a specific seed."""
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "results", "comparisons")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"comparison_seed_{seed}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, ensure_ascii=False, indent=2)
    
    return output_path


def run_trials(num_trials, output_file, silver_path=None, training_samples=10000):
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
    
    # Run trials and track best
    all_results = []
    best_score = -1
    best_trial_idx = 0
    best_seed = 42
    
    successful_trials = 0
    current_seed = 42
    failed_seeds = []
    
    while successful_trials < num_trials:
        print(f"\n--- Trial {successful_trials + 1}/{num_trials} (seed={current_seed}) ---")
        
        verbose = (num_trials == 1)  # Only show progress bars for single trial
        
        try:
            trial_results, models, test_data = run_single_trial(
                poems, current_seed, tokenizer, device, 
                training_samples=training_samples, verbose=verbose
            )
            
            all_results.append(trial_results)
            
            # Generate and save model comparison for this seed
            comparison_data = generate_model_comparison(
                test_data["poem4_test_raw"], models, tokenizer, device
            )
            comparison_data["seed"] = current_seed
            comparison_path = save_model_comparison(comparison_data, current_seed)
            
            # Check if this is the best trial so far
            score = compute_aggregate_score(trial_results)
            if score > best_score:
                best_score = score
                best_trial_idx = successful_trials
                best_seed = current_seed
                # Save models (overwrites previous best)
                save_best_models(models, test_data, tokenizer)
            
            # Print summary for this trial
            print(f"  Char: {trial_results['char']['accuracy']:.4f}  "
                  f"Couplet: {trial_results['couplet']['accuracy']:.4f}  "
                  f"Poem4: {trial_results['poem4_overall']['accuracy']:.4f}  "
                  f"Poem1: {trial_results['poem1']['accuracy']:.4f}"
                  + (f"  [NEW BEST: {score:.4f}]" if score == best_score else ""))
            print(f"  Comparison saved to: {comparison_path}")
            
            successful_trials += 1
            
        except TrainingFailedError as e:
            print(f"  FAILED: {e}")
            print(f"  Skipping seed {current_seed}, trying next seed...")
            failed_seeds.append(current_seed)
        
        # Clean up models after each trial
        try:
            del models
        except NameError:
            pass
        free_memory(device)
        
        # Move to next seed
        current_seed += 1
    
    # Compute and display statistics
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    
    if failed_seeds:
        print(f"Failed seeds (skipped): {failed_seeds}")
        print()
    
    if num_trials > 1:
        statistics = compute_statistics(all_results)
        
        # Group statistics by model
        models_order = ["char", "couplet", "poem4_overall", "poem4_inner", "poem1", 
                       "char_induced_couplet", "couplet_induced_poem"]
        metrics_order = ["accuracy", "precision", "recall", "f1"]
        
        for model in models_order:
            print(f"{model}:")
            for metric in metrics_order:
                key = f"{model}_{metric}"
                if key in statistics:
                    stats = statistics[key]
                    print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f} [{stats['min']:.4f}, {stats['max']:.4f}]")
            print()
        
        # Save results
        output_data = {
            "num_trials": num_trials,
            "best_trial": best_trial_idx + 1,
            "best_seed": best_seed,
            "best_score": best_score,
            "failed_seeds": failed_seeds,
            "statistics": statistics,
            "trials": all_results,
        }
    else:
        # Single trial - just print and save the results
        for model_key, metrics in all_results[0].items():
            if model_key == "seed":
                continue
            print(f"{model_key}:")
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    if metric_name not in ["tp", "fp", "fn", "tn"]:
                        print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metrics:.4f}")
        print()
        
        output_data = {
            "num_trials": 1,
            "failed_seeds": failed_seeds,
            "trials": all_results,
        }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to {output_file}")
    
    print()
    print(f"Best models (Trial {best_trial_idx + 1}, seed={best_seed}, score={best_score:.4f}) saved to saved_artifacts/")
    print(f"Model comparisons saved to results/comparisons/")


def main():
    default_output = os.path.join(PROJECT_ROOT, "results", "evaluation_results.json")
    default_data = os.path.join(PROJECT_ROOT, "data", "silver_standard.json")
    
    parser = argparse.ArgumentParser(
        description="Run parallelism model training and evaluation trials"
    )
    parser.add_argument(
        "--trials", type=int, default=1,
        help="Number of trials to run (default: 1)"
    )
    parser.add_argument(
        "--output", type=str, default=default_output,
        help=f"Output file for results (default: results/evaluation_results.json)"
    )
    parser.add_argument(
        "--data", type=str, default=default_data,
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
