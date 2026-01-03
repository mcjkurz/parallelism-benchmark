import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizerFast, BertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import pickle
import json

from datasets import CharPairDataset, CoupletDataset, PoemDataset4Labels, PoemDataset1Label
from models import PoemParallelismClassifier
from data_loader import prepare_data
from utils import create_training_datasets, split_raw_data

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

PRETRAINED_MODEL_NAME = "SIKU-BERT/sikubert"

def evaluate_standard(model, dataset, batch_size=32):
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


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model_single(model, dataset, epochs=1, batch_size=8, lr=2e-5):
    """Train a single model (used in multi-trial evaluation)."""
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps
    )

    for epoch in range(epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs["loss"]

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            loop.set_postfix(loss=loss.item())

    return model


def run_single_trial(poems, seed, tokenizer, couplet_tokens):
    """Run a single training and evaluation trial with the given seed."""
    set_seed(seed)
    
    # Create training datasets with this seed
    training_data_characters, training_data_couplets, training_data_poems_4labels, training_data_poems_1label = \
        create_training_datasets(poems)
    
    # Split data
    char_train_raw, char_test_raw = split_raw_data(training_data_characters, seed=seed)
    coup_train_raw, coup_test_raw = split_raw_data(training_data_couplets, seed=seed)
    poem4_train_raw, poem4_test_raw = split_raw_data(training_data_poems_4labels, seed=seed)
    poem1_train_raw, poem1_test_raw = split_raw_data(training_data_poems_1label, seed=seed)
    
    # Create datasets
    char_train_ds = CharPairDataset(char_train_raw, tokenizer)
    coup_train_ds = CoupletDataset(coup_train_raw, tokenizer)
    poem4_train_ds = PoemDataset4Labels(poem4_train_raw, tokenizer)
    poem1_train_ds = PoemDataset1Label(poem1_train_raw, tokenizer)
    
    char_test_ds = CharPairDataset(char_test_raw, tokenizer)
    coup_test_ds = CoupletDataset(coup_test_raw, tokenizer)
    poem4_test_ds = PoemDataset4Labels(poem4_test_raw, tokenizer)
    poem1_test_ds = PoemDataset1Label(poem1_test_raw, tokenizer)
    
    # Train models
    char_model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=2)
    char_model = train_model_single(char_model, char_train_ds, epochs=1)
    
    coup_model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=2)
    coup_model = train_model_single(coup_model, coup_train_ds, epochs=1)
    
    poem4_model = PoemParallelismClassifier.create_initial(
        pretrained_name=PRETRAINED_MODEL_NAME,
        tokenizer=tokenizer,
        couplet_tokens=couplet_tokens,
        num_couplets=4,
        num_labels=2
    )
    poem4_model = train_model_single(poem4_model, poem4_train_ds, epochs=1)
    
    poem1_model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=2)
    poem1_model = train_model_single(poem1_model, poem1_train_ds, epochs=1)
    
    # Evaluate models
    results = {}
    
    results["char_acc"] = evaluate_standard(char_model, char_test_ds)
    results["coup_acc"] = evaluate_standard(coup_model, coup_test_ds)
    results["poem4_overall_acc"] = evaluate_standard(poem4_model, poem4_test_ds)
    results["poem4_inner_acc"] = evaluate_poem4_inner_accuracy(poem4_model, poem4_test_ds)
    results["poem1_acc"] = evaluate_standard(poem1_model, poem1_test_ds)
    results["poem1_inner_acc"] = evaluate_poem1_inner_accuracy(poem1_model, poem1_test_raw, tokenizer)
    results["char_induced_coup_acc"] = evaluate_char_induced_couplet_accuracy(char_model, coup_test_raw, tokenizer)
    results["coup_induced_poem_acc"] = evaluate_couplet_induced_poem_accuracy(coup_model, poem1_test_raw, tokenizer)
    
    # Clean up to free memory
    del char_model, coup_model, poem4_model, poem1_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results


def run_multi_trial_evaluation(num_trials=100, output_file="evaluation_results.json"):
    """Run multiple training/evaluation trials and compute statistics."""
    print(f"\n{'='*60}")
    print(f"Running {num_trials} trials for statistical evaluation")
    print(f"{'='*60}\n")
    
    # Prepare data once (labels are deterministic)
    print("Preparing data...")
    poems = prepare_data(export_silver=False)
    
    # Initialize tokenizer once
    tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_MODEL_NAME)
    couplet_tokens = ["[CP1]", "[CP2]", "[CP3]", "[CP4]"]
    tokenizer.add_special_tokens({"additional_special_tokens": couplet_tokens})
    
    # Collect results from all trials
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
        seed = 42 + trial  # Different seed for each trial
        print(f"\n--- Trial {trial + 1}/{num_trials} (seed={seed}) ---")
        
        trial_results = run_single_trial(poems, seed, tokenizer, couplet_tokens)
        
        for key, value in trial_results.items():
            all_results[key].append(value)
        
        # Print running statistics
        print(f"  Char Acc: {trial_results['char_acc']:.4f}")
        print(f"  Couplet Acc: {trial_results['coup_acc']:.4f}")
        print(f"  Poem4 Overall: {trial_results['poem4_overall_acc']:.4f}")
        print(f"  Poem1 Acc: {trial_results['poem1_acc']:.4f}")
    
    # Compute statistics
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
    
    # Save results to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "num_trials": num_trials,
            "statistics": statistics,
        }, f, indent=2)
    print(f"Results saved to {output_file}")

def main_single():
    """Run single evaluation on pre-trained models."""
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
    
    acc_char = evaluate_standard(char_model, char_test_ds)
    print(f"Char Model Test Acc: {acc_char:.4f}")

    acc_coup = evaluate_standard(coup_model, coup_test_ds)
    print(f"Couplet Model Test Acc: {acc_coup:.4f}")

    acc_poem4_all = evaluate_standard(poem4_model, poem4_test_ds)
    acc_poem4_inner = evaluate_poem4_inner_accuracy(poem4_model, poem4_test_ds)
    print(f"Poem4 Model Overall Acc: {acc_poem4_all:.4f}")
    print(f"Poem4 Model Inner-Couplet Acc: {acc_poem4_inner:.4f}")

    acc_poem1 = evaluate_standard(poem1_model, poem1_test_ds)
    acc_poem1_inner = evaluate_poem1_inner_accuracy(poem1_model, poem1_test_raw, tokenizer)
    print(f"Poem1 Model Test Acc: {acc_poem1:.4f}")
    print(f"Poem1 Model Inner-Couplet Acc: {acc_poem1_inner:.4f}")

    print("\nRunning Cross-Level Evaluations...")
    acc_char_induced = evaluate_char_induced_couplet_accuracy(char_model, coup_test_raw, tokenizer)
    print(f"Couplet Acc (Induced by Char Model): {acc_char_induced:.4f}")

    acc_coup_induced = evaluate_couplet_induced_poem_accuracy(coup_model, poem1_test_raw, tokenizer)
    print(f"Poem Acc (Induced by Couplet Model): {acc_coup_induced:.4f}")


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

