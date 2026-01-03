"""
Evaluate parallelism detection models.

Usage:
    python evaluate.py              # Single evaluation on saved models
    python evaluate.py --trials 100 # Multi-trial evaluation with statistics
"""

import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
from tqdm.auto import tqdm
import pickle
import json

from datasets import CharPairDataset, CoupletDataset, PoemDataset4Labels, PoemDataset1Label
from models import PoemParallelismClassifier
from data_loader import prepare_data
from utils import create_training_datasets, split_raw_data
from train_utils import (
    get_device, create_tokenizer, train_all_models, free_memory,
    PRETRAINED_MODEL_NAME
)

device = get_device()
print(f"Using device: {device}")


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_standard(model, dataset, batch_size=32):
    """Evaluate model accuracy on a dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
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


def evaluate_char_induced_couplet_accuracy(char_model, raw_couplet_data, tokenizer):
    """Evaluate couplet accuracy induced by character-level predictions."""
    char_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for item in tqdm(raw_couplet_data, desc="Char->Couplet Eval", leave=False):
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


def evaluate_couplet_induced_poem_accuracy(couplet_model, raw_poem_data, tokenizer):
    """Evaluate poem accuracy induced by couplet-level predictions."""
    couplet_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for item in tqdm(raw_poem_data, desc="Couplet->Poem Eval", leave=False):
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


def evaluate_poem4_inner_accuracy(model, dataset):
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


def evaluate_poem1_inner_accuracy(poem1_model, raw_poem_data, tokenizer):
    """Evaluate Poem1 model accuracy on inner couplet prediction."""
    poem1_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for item in tqdm(raw_poem_data, desc="Poem1 Inner-Couplet Eval", leave=False):
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
# Multi-Trial Evaluation
# =============================================================================

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single_trial(poems, seed, tokenizer):
    """Run a single training and evaluation trial with the given seed."""
    set_seed(seed)
    
    # Create and split datasets
    training_data_characters, training_data_couplets, training_data_poems_4labels, training_data_poems_1label = \
        create_training_datasets(poems)
    
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
        tokenizer, device=device, verbose=False
    )
    
    # Evaluate all models
    results = {
        "char_acc": evaluate_standard(char_model, char_test_ds),
        "coup_acc": evaluate_standard(coup_model, coup_test_ds),
        "poem4_overall_acc": evaluate_standard(poem4_model, poem4_test_ds),
        "poem4_inner_acc": evaluate_poem4_inner_accuracy(poem4_model, poem4_test_ds),
        "poem1_acc": evaluate_standard(poem1_model, poem1_test_ds),
        "poem1_inner_acc": evaluate_poem1_inner_accuracy(poem1_model, poem1_test_raw, tokenizer),
        "char_induced_coup_acc": evaluate_char_induced_couplet_accuracy(char_model, coup_test_raw, tokenizer),
        "coup_induced_poem_acc": evaluate_couplet_induced_poem_accuracy(coup_model, poem1_test_raw, tokenizer),
    }
    
    # Clean up
    del char_model, coup_model, poem4_model, poem1_model
    free_memory(device)
    
    return results


def run_multi_trial_evaluation(num_trials=100, output_file="evaluation_results.json"):
    """Run multiple training/evaluation trials and compute statistics."""
    print(f"\n{'='*60}")
    print(f"Running {num_trials} trials for statistical evaluation")
    print(f"{'='*60}\n")
    
    # Prepare data once
    print("Preparing data...")
    poems = prepare_data(export_silver=False)
    
    # Initialize tokenizer once
    tokenizer = create_tokenizer()
    
    # Collect results
    all_results = {
        "char_acc": [],
        "coup_acc": [],
        "poem4_overall_acc": [],
        "poem4_inner_acc": [],
        "poem1_acc": [],
        "poem1_inner_acc": [],
        "char_induced_coup_acc": [],
        "coup_induced_poem_acc": [],
    }
    
    for trial in range(num_trials):
        seed = 42 + trial
        print(f"\n--- Trial {trial + 1}/{num_trials} (seed={seed}) ---")
        
        trial_results = run_single_trial(poems, seed, tokenizer)
        
        for key, value in trial_results.items():
            all_results[key].append(value)
        
        print(f"  Char: {trial_results['char_acc']:.4f}  "
              f"Couplet: {trial_results['coup_acc']:.4f}  "
              f"Poem4: {trial_results['poem4_overall_acc']:.4f}  "
              f"Poem1: {trial_results['poem1_acc']:.4f}")
    
    # Compute and display statistics
    print(f"\n{'='*60}")
    print("FINAL STATISTICS")
    print(f"{'='*60}\n")
    
    statistics = {}
    for key, values in all_results.items():
        values_np = np.array(values)
        stats = {
            "mean": float(np.mean(values_np)),
            "std": float(np.std(values_np)),
            "min": float(np.min(values_np)),
            "max": float(np.max(values_np)),
            "all_values": values,
        }
        statistics[key] = stats
        print(f"{key}:")
        print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print()
    
    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"num_trials": num_trials, "statistics": statistics}, f, indent=2)
    print(f"Results saved to {output_file}")


# =============================================================================
# Single Evaluation (on saved models)
# =============================================================================

def main_single():
    """Evaluate pre-trained models from saved_artifacts/."""
    print("Loading models and data...")
    tokenizer = BertTokenizerFast.from_pretrained("saved_artifacts/tokenizer")
    
    char_model = BertForSequenceClassification.from_pretrained("saved_artifacts/char_model").to(device)
    coup_model = BertForSequenceClassification.from_pretrained("saved_artifacts/coup_model").to(device)
    poem4_model = PoemParallelismClassifier.from_pretrained("saved_artifacts/poem4_model").to(device)
    poem1_model = BertForSequenceClassification.from_pretrained("saved_artifacts/poem1_model").to(device)

    with open("saved_artifacts/char_test_raw.pkl", "rb") as f:
        char_test_raw = pickle.load(f)
    with open("saved_artifacts/coup_test_raw.pkl", "rb") as f:
        coup_test_raw = pickle.load(f)
    with open("saved_artifacts/poem4_test_raw.pkl", "rb") as f:
        poem4_test_raw = pickle.load(f)
    with open("saved_artifacts/poem1_test_raw.pkl", "rb") as f:
        poem1_test_raw = pickle.load(f)

    char_test_ds = CharPairDataset(char_test_raw, tokenizer)
    coup_test_ds = CoupletDataset(coup_test_raw, tokenizer)
    poem4_test_ds = PoemDataset4Labels(poem4_test_raw, tokenizer)
    poem1_test_ds = PoemDataset1Label(poem1_test_raw, tokenizer)

    print("\nEvaluating models...")
    
    print(f"Char Model Test Acc: {evaluate_standard(char_model, char_test_ds):.4f}")
    print(f"Couplet Model Test Acc: {evaluate_standard(coup_model, coup_test_ds):.4f}")
    
    print(f"Poem4 Model Overall Acc: {evaluate_standard(poem4_model, poem4_test_ds):.4f}")
    print(f"Poem4 Model Inner-Couplet Acc: {evaluate_poem4_inner_accuracy(poem4_model, poem4_test_ds):.4f}")
    
    print(f"Poem1 Model Test Acc: {evaluate_standard(poem1_model, poem1_test_ds):.4f}")
    print(f"Poem1 Model Inner-Couplet Acc: {evaluate_poem1_inner_accuracy(poem1_model, poem1_test_raw, tokenizer):.4f}")

    print("\nCross-Level Evaluations...")
    print(f"Couplet Acc (Induced by Char Model): {evaluate_char_induced_couplet_accuracy(char_model, coup_test_raw, tokenizer):.4f}")
    print(f"Poem Acc (Induced by Couplet Model): {evaluate_couplet_induced_poem_accuracy(coup_model, poem1_test_raw, tokenizer):.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate parallelism models")
    parser.add_argument("--trials", type=int, default=0,
                        help="Number of trials for statistical evaluation (0 = single evaluation)")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                        help="Output file for multi-trial results")
    args = parser.parse_args()
    
    if args.trials > 0:
        run_multi_trial_evaluation(num_trials=args.trials, output_file=args.output)
    else:
        main_single()


if __name__ == "__main__":
    main()
