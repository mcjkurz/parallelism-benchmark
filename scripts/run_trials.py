"""
Run training and evaluation trials for parallelism detection models.

Each trial trains all 4 models (char, couplet, poem4, poem1) together on the same
data split. All models AND induced metrics must achieve accuracy > 0.6 for a trial
to be successful.

After training, induced metrics are computed and checked:
- char → couplet: If ≥3/5 char pairs are parallel → couplet is parallel
- couplet → poem: If inner couplets (indices 1,2) are parallel → poem is regulated
- poem4 → poem: If poem4 predicts inner couplets parallel → poem is regulated
- char → poem: Full chain: char→couplet→poem

Usage:
    python scripts/run_trials.py --trials 3     # Run 3 trials for testing
    python scripts/run_trials.py --trials 100   # Run 100 trials
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

from datasets import CharPairDataset, CoupletDataset, PoemDataset4Labels, PoemDataset1Label
from models import PoemParallelismClassifier
from train_utils import (
    get_device, create_tokenizer, train_model, free_memory,
    EPOCHS_CHAR, EPOCHS_COUPLET, EPOCHS_POEM4, EPOCHS_POEM1,
    PRETRAINED_MODEL_NAME
)
from inference import predict_char_pairs, predict_couplet, predict_poem4
from transformers import set_seed, BertForSequenceClassification

# Configuration
MIN_ACCURACY_THRESHOLD = 0.6
TRAIN_RATIO = 0.9

# Sample sizes (same as previous version)
TRAIN_SAMPLES = 9000          # Training samples per model (balanced)
TEST_SAMPLES = 1000           # Test samples per model (balanced)


def load_silver_standard(path=None):
    """Load poems from JSON."""
    if path is None:
        path = os.path.join(PROJECT_ROOT, "data", "silver_standard_train.json")
    if not os.path.exists(path):
        print(f"Error: Data not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def compute_statistics(results_list, metric):
    """Compute mean, std, min, max for a metric."""
    values = [r[metric] for r in results_list]
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def split_poems(poems, seed, train_ratio=0.9):
    """Split poems into train/test sets."""
    poems_copy = list(poems)
    random.seed(seed)
    random.shuffle(poems_copy)
    split_idx = int(len(poems_copy) * train_ratio)
    return poems_copy[:split_idx], poems_copy[split_idx:]


def create_char_data(poems):
    """Create character-level training data from poems."""
    data = []
    for poem in poems:
        for couplet_id, couplet in enumerate(poem["couplets"]):
            char_labels = poem["char_match"][couplet_id]
            line_label = poem["line_match"][couplet_id]
            s = sum(char_labels)
            # Positive: parallel couplet with all chars matching
            if line_label == 1 and s == 5:
                for i in range(5):
                    data.append({"character_pair": (couplet[0][i], couplet[1][i]), "label": 1})
            # Negative: non-parallel couplet with no chars matching
            elif line_label == 0 and s == 0:
                for i in range(5):
                    data.append({"character_pair": (couplet[0][i], couplet[1][i]), "label": 0})
    return data


def create_couplet_data(poems):
    """Create couplet-level training data from poems."""
    data = []
    for poem in poems:
        for couplet_id, couplet in enumerate(poem["couplets"]):
            label = poem["line_match"][couplet_id]
            data.append({"couplet": (couplet[0], couplet[1]), "label": label})
    return data


def create_poem4_data(poems):
    """Create poem 4-label data."""
    data = []
    for poem in poems:
        if len(poem["couplets"]) == 4 and len(poem["line_match"]) == 4:
            data.append({
                "couplets": poem["couplets"],
                "labels": poem["line_match"][:]
            })
    return data


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


def create_bert_classifier(model_seed, num_labels=2):
    """Create BERT classifier."""
    set_seed(model_seed)
    torch.manual_seed(model_seed)
    model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=num_labels)
    model.classifier.bias.data.zero_()
    return model


def create_poem4_classifier(model_seed, tokenizer):
    """Create Poem4 classifier."""
    set_seed(model_seed)
    torch.manual_seed(model_seed)
    from train_utils import COUPLET_TOKENS
    model = PoemParallelismClassifier.create_initial(
        pretrained_name=PRETRAINED_MODEL_NAME,
        tokenizer=tokenizer,
        couplet_tokens=COUPLET_TOKENS,
        num_couplets=4,
        num_labels=2
    )
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
            if logits.dim() == 3:
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.reshape(-1).cpu().tolist())
                all_labels.extend(labels.reshape(-1).cpu().tolist())
            else:
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
    
    return compute_metrics(all_preds, all_labels)


def compute_induced_char_to_couplet(char_model, tokenizer, coup_test_data, device):
    """Compute char→couplet induced metrics on balanced couplet test data.
    
    For each couplet: if ≥3/5 char pairs are parallel → couplet is parallel.
    """
    preds = []
    labels = []
    
    for item in coup_test_data:
        couplet = item["couplet"]  # (line1, line2)
        label = item["label"]
        labels.append(label)
        
        # Get char predictions
        char_preds = predict_char_pairs(char_model, tokenizer, couplet, device)
        # Induce: if ≥3 chars are parallel → couplet is parallel
        induced_pred = 1 if sum(char_preds) >= 3 else 0
        preds.append(induced_pred)
    
    return compute_metrics(preds, labels)


def compute_induced_couplet_to_poem(couplet_model, tokenizer, poem1_test_data, device):
    """Compute couplet→poem induced metrics on balanced poem1 test data.
    
    For each poem: if inner couplets (indices 1,2) are both parallel → poem is regulated.
    """
    preds = []
    labels = []
    
    for item in poem1_test_data:
        couplets = item["couplets"]
        label = item["label"]  # Already: 1 if inner couplets parallel, else 0
        labels.append(label)
        
        # Get couplet predictions for inner couplets (indices 1, 2)
        pred_inner1 = predict_couplet(couplet_model, tokenizer, couplets[1], device)
        pred_inner2 = predict_couplet(couplet_model, tokenizer, couplets[2], device)
        
        # Induce: if both inner couplets are parallel → poem is regulated
        induced_pred = 1 if (pred_inner1 == 1 and pred_inner2 == 1) else 0
        preds.append(induced_pred)
    
    return compute_metrics(preds, labels)


def compute_induced_poem4_to_poem(poem4_model, tokenizer, poem1_test_data, device):
    """Compute poem4→poem induced metrics on balanced poem1 test data.
    
    For each poem: if inner couplets (indices 1,2) are both predicted parallel → poem is regulated.
    Uses the poem4 model which predicts all 4 couplet labels simultaneously.
    """
    preds = []
    labels = []
    
    for item in poem1_test_data:
        couplets = item["couplets"]
        label = item["label"]  # Already: 1 if inner couplets parallel, else 0
        labels.append(label)
        
        # Get poem4 predictions for all 4 couplets
        poem4_preds = predict_poem4(poem4_model, tokenizer, couplets, device)
        
        # Induce: if both inner couplets (indices 1, 2) are parallel → poem is regulated
        induced_pred = 1 if (poem4_preds[1] == 1 and poem4_preds[2] == 1) else 0
        preds.append(induced_pred)
    
    return compute_metrics(preds, labels)


def compute_induced_char_to_poem(char_model, tokenizer, poem1_test_data, device):
    """Compute char→poem induced metrics (full chain) on balanced poem1 test data.
    
    For each poem:
    1. Use char model to predict 5 char pairs for each inner couplet (indices 1, 2)
    2. Apply ≥3/5 rule: if ≥3 char pairs parallel → couplet is parallel
    3. If BOTH inner couplets are induced as parallel → poem is regulated
    """
    preds = []
    labels = []
    
    for item in poem1_test_data:
        couplets = item["couplets"]
        label = item["label"]
        labels.append(label)
        
        # Get char predictions for inner couplets (indices 1 and 2)
        char_preds_inner1 = predict_char_pairs(char_model, tokenizer, couplets[1], device)
        char_preds_inner2 = predict_char_pairs(char_model, tokenizer, couplets[2], device)
        
        # Induce couplet-level: ≥3/5 char pairs parallel → couplet is parallel
        induced_coup1 = 1 if sum(char_preds_inner1) >= 3 else 0
        induced_coup2 = 1 if sum(char_preds_inner2) >= 3 else 0
        
        # Induce poem-level: both inner couplets parallel → poem is regulated
        induced_pred = 1 if (induced_coup1 == 1 and induced_coup2 == 1) else 0
        preds.append(induced_pred)
    
    return compute_metrics(preds, labels)


MAX_RETRIES_PER_EXPERIMENT = 10  # Max retries for each direct experiment


def train_with_retry(model_name, create_model_fn, train_ds, test_ds, epochs,
                     device, data_seed, seed_counter, tokenizer=None, use_balanced_batches=True):
    """Train a single model with retries on failure.
    
    Returns (model, metrics, updated_seed_counter) or (None, None, updated_seed_counter).
    seed_counter is incremented for each attempt to ensure unique seeds.
    """
    for attempt in range(MAX_RETRIES_PER_EXPERIMENT):
        model_seed = seed_counter
        seed_counter += 1
        if attempt > 0:
            print(f"    [{model_name}] Retry {attempt}/{MAX_RETRIES_PER_EXPERIMENT-1} (seed={model_seed})")
        
        if tokenizer is not None:
            model = create_model_fn(model_seed, tokenizer)
        else:
            model = create_model_fn(model_seed)
        
        set_seed(data_seed)
        model = train_model(model, train_ds, epochs=epochs, device=device,
                           verbose=False, use_balanced_batches=use_balanced_batches)
        metrics = evaluate_model(model, test_ds, device)
        
        print(f"    {model_name}:{' '*(8-len(model_name))}acc={metrics['accuracy']:.4f} prec={metrics['precision']:.4f} "
              f"rec={metrics['recall']:.4f} f1={metrics['f1']:.4f}")
        
        if metrics["accuracy"] >= MIN_ACCURACY_THRESHOLD:
            return model, metrics, seed_counter
        
        print(f"    ✗ {model_name} accuracy < {MIN_ACCURACY_THRESHOLD}")
        del model
        free_memory(device)
    
    print(f"    ✗ {model_name} failed after {MAX_RETRIES_PER_EXPERIMENT} attempts")
    return None, None, seed_counter


def run_single_trial(poems, tokenizer, device, seed_counter, data_seed):
    """Run a single trial: train all 4 models, evaluate, compute induced metrics.
    
    Returns (result, updated_seed_counter) or (None, updated_seed_counter).
    seed_counter ensures unique model seeds across all experiments and trials.
    """
    print(f"\n  [Trial] data_seed={data_seed}, model_seed_start={seed_counter}")
    random.seed(data_seed)
    
    # Split poems into train/test (90/10)
    train_poems, test_poems = split_poems(poems, data_seed, TRAIN_RATIO)
    
    # Create training data from train poems
    char_train_all = create_char_data(train_poems)
    coup_train_all = create_couplet_data(train_poems)
    poem4_train_all = create_poem4_data(train_poems)
    poem1_train_all = create_poem1_data(train_poems)
    
    # Create test data from test poems
    char_test_all = create_char_data(test_poems)
    coup_test_all = create_couplet_data(test_poems)
    poem4_test_all = create_poem4_data(test_poems)
    poem1_test_all = create_poem1_data(test_poems)
    
    # Balance training data and limit size
    char_train = balance_data(char_train_all, key="label", max_samples=TRAIN_SAMPLES)
    coup_train = balance_data(coup_train_all, key="label", max_samples=TRAIN_SAMPLES)
    poem1_train = balance_data(poem1_train_all, key="label", max_samples=TRAIN_SAMPLES)
    poem4_train = balance_poem4_data_per_position(poem4_train_all, target_samples=TRAIN_SAMPLES, force_target=True, label="poem4_train")
    
    # Balance test data and limit size
    char_test = balance_data(char_test_all, key="label", max_samples=TEST_SAMPLES)
    coup_test = balance_data(coup_test_all, key="label", max_samples=TEST_SAMPLES)
    poem1_test = balance_data(poem1_test_all, key="label", max_samples=TEST_SAMPLES)
    poem4_test = balance_poem4_data_per_position(poem4_test_all, target_samples=TEST_SAMPLES, force_target=False, label="poem4_test")

    print(f"    Data sizes: char={len(char_train)}/{len(char_test)}, "
          f"coup={len(coup_train)}/{len(coup_test)}, "
          f"poem4={len(poem4_train)}/{len(poem4_test)}, "
          f"poem1={len(poem1_train)}/{len(poem1_test)}")
    
    # Train char model (with retry)
    char_train_ds = CharPairDataset(char_train, tokenizer)
    char_test_ds = CharPairDataset(char_test, tokenizer)
    char_model, char_metrics, seed_counter = train_with_retry(
        "char", create_bert_classifier, char_train_ds, char_test_ds,
        EPOCHS_CHAR, device, data_seed, seed_counter, use_balanced_batches=True
    )
    if char_model is None:
        return None, seed_counter
    
    # Train couplet model (with retry)
    coup_train_ds = CoupletDataset(coup_train, tokenizer)
    coup_test_ds = CoupletDataset(coup_test, tokenizer)
    coup_model, coup_metrics, seed_counter = train_with_retry(
        "couplet", create_bert_classifier, coup_train_ds, coup_test_ds,
        EPOCHS_COUPLET, device, data_seed, seed_counter, use_balanced_batches=True
    )
    if coup_model is None:
        del char_model
        free_memory(device)
        return None, seed_counter
    
    # Train poem4 model (with retry)
    poem4_train_ds = PoemDataset4Labels(poem4_train, tokenizer)
    poem4_test_ds = PoemDataset4Labels(poem4_test, tokenizer)
    poem4_model, poem4_metrics, seed_counter = train_with_retry(
        "poem4", create_poem4_classifier, poem4_train_ds, poem4_test_ds,
        EPOCHS_POEM4, device, data_seed, seed_counter, tokenizer=tokenizer, use_balanced_batches=False
    )
    if poem4_model is None:
        del char_model, coup_model
        free_memory(device)
        return None, seed_counter
    
    # Train poem1 model (with retry)
    poem1_train_ds = PoemDataset1Label(poem1_train, tokenizer)
    poem1_test_ds = PoemDataset1Label(poem1_test, tokenizer)
    poem1_model, poem1_metrics, seed_counter = train_with_retry(
        "poem1", create_bert_classifier, poem1_train_ds, poem1_test_ds,
        EPOCHS_POEM1, device, data_seed, seed_counter, use_balanced_batches=True
    )
    if poem1_model is None:
        del char_model, coup_model, poem4_model
        free_memory(device)
        return None, seed_counter
    
    # Compute induced metrics (using same balanced test data as direct models)
    print("    Computing induced metrics...")
    char_to_coup_metrics = compute_induced_char_to_couplet(char_model, tokenizer, coup_test, device)
    print(f"    char→coup: acc={char_to_coup_metrics['accuracy']:.4f} prec={char_to_coup_metrics['precision']:.4f} "
          f"rec={char_to_coup_metrics['recall']:.4f} f1={char_to_coup_metrics['f1']:.4f}")
    if char_to_coup_metrics["accuracy"] < MIN_ACCURACY_THRESHOLD:
        print(f"    ✗ Failed: char→coup accuracy < {MIN_ACCURACY_THRESHOLD}")
        del char_model, coup_model, poem4_model, poem1_model
        free_memory(device)
        return None, seed_counter
    
    coup_to_poem_metrics = compute_induced_couplet_to_poem(coup_model, tokenizer, poem1_test, device)
    print(f"    coup→poem: acc={coup_to_poem_metrics['accuracy']:.4f} prec={coup_to_poem_metrics['precision']:.4f} "
          f"rec={coup_to_poem_metrics['recall']:.4f} f1={coup_to_poem_metrics['f1']:.4f}")
    if coup_to_poem_metrics["accuracy"] < MIN_ACCURACY_THRESHOLD:
        print(f"    ✗ Failed: coup→poem accuracy < {MIN_ACCURACY_THRESHOLD}")
        del char_model, coup_model, poem4_model, poem1_model
        free_memory(device)
        return None, seed_counter
    
    poem4_to_poem_metrics = compute_induced_poem4_to_poem(poem4_model, tokenizer, poem1_test, device)
    print(f"    poem4→poem: acc={poem4_to_poem_metrics['accuracy']:.4f} prec={poem4_to_poem_metrics['precision']:.4f} "
          f"rec={poem4_to_poem_metrics['recall']:.4f} f1={poem4_to_poem_metrics['f1']:.4f}")
    if poem4_to_poem_metrics["accuracy"] < MIN_ACCURACY_THRESHOLD:
        print(f"    ✗ Failed: poem4→poem accuracy < {MIN_ACCURACY_THRESHOLD}")
        del char_model, coup_model, poem4_model, poem1_model
        free_memory(device)
        return None, seed_counter
    
    char_to_poem_metrics = compute_induced_char_to_poem(char_model, tokenizer, poem1_test, device)
    print(f"    char→poem: acc={char_to_poem_metrics['accuracy']:.4f} prec={char_to_poem_metrics['precision']:.4f} "
          f"rec={char_to_poem_metrics['recall']:.4f} f1={char_to_poem_metrics['f1']:.4f}")
    if char_to_poem_metrics["accuracy"] < MIN_ACCURACY_THRESHOLD:
        print(f"    ✗ Failed: char→poem accuracy < {MIN_ACCURACY_THRESHOLD}")
        del char_model, coup_model, poem4_model, poem1_model
        free_memory(device)
        return None, seed_counter
    
    print(f"    ✓ Success")
    
    result = {
        "data_seed": data_seed,
        "char": char_metrics,
        "couplet": coup_metrics,
        "poem4": poem4_metrics,
        "poem1": poem1_metrics,
        "char_to_couplet": char_to_coup_metrics,
        "couplet_to_poem": coup_to_poem_metrics,
        "poem4_to_poem": poem4_to_poem_metrics,
        "char_to_poem": char_to_poem_metrics,
        # Keep models for potential saving (will be cleaned up by caller)
        "_models": {
            "char": char_model,
            "couplet": coup_model,
            "poem4": poem4_model,
            "poem1": poem1_model,
        }
    }
    
    return result, seed_counter


def run_trials(poems, tokenizer, device, target_trials, model_seed_start, data_seed_start, max_attempts=500):
    """Run trials until we get target_trials successful ones."""
    print(f"\n{'='*60}")
    print(f"Running trials (target: {target_trials} successful)")
    print(f"Model seed start: {model_seed_start}, Data seed start: {data_seed_start}")
    print(f"Min accuracy threshold: {MIN_ACCURACY_THRESHOLD}")
    print(f"{'='*60}")
    
    successful_results = []
    failed_seeds = []
    current_data_seed = data_seed_start
    seed_counter = model_seed_start  # Tracks unique model seeds across all trials/experiments
    
    # Track best model for EACH type separately
    model_types = ["char", "couplet", "poem4", "poem1"]
    best_models = {t: None for t in model_types}
    best_accuracies = {t: 0.0 for t in model_types}
    best_seeds = {t: None for t in model_types}
    
    while len(successful_results) < target_trials and (current_data_seed - data_seed_start) < max_attempts:
        result, seed_counter = run_single_trial(poems, tokenizer, device, seed_counter, current_data_seed)
        
        if result is not None:
            # Track best model for each type separately
            models_to_keep = set()
            for model_type in model_types:
                acc = result[model_type]["accuracy"]
                if acc > best_accuracies[model_type]:
                    # Clean up previous best for this type
                    if best_models[model_type] is not None:
                        del best_models[model_type]
                    # Keep new best
                    best_models[model_type] = result["_models"][model_type]
                    best_accuracies[model_type] = acc
                    best_seeds[model_type] = result["data_seed"]
                    models_to_keep.add(model_type)
                    print(f"    ★ New best {model_type}! acc={acc:.4f}")
            
            # Clean up models that weren't kept
            for model_type in model_types:
                if model_type not in models_to_keep and "_models" in result:
                    if result["_models"][model_type] is not best_models[model_type]:
                        del result["_models"][model_type]
            
            free_memory(device)
            successful_results.append(result)
            print(f"\n  Progress: {len(successful_results)}/{target_trials} successful trials")
        else:
            failed_seeds.append(current_data_seed)
        
        current_data_seed += 1
    
    # Package best models info
    best_info = {
        "models": best_models,
        "accuracies": best_accuracies,
        "seeds": best_seeds,
    }
    
    return successful_results, failed_seeds, best_info


def aggregate_results(results):
    """Aggregate results across trials."""
    if not results:
        return {}
    
    # Metrics to aggregate
    model_keys = ["char", "couplet", "poem4", "poem1", "char_to_couplet", "couplet_to_poem", "poem4_to_poem", "char_to_poem"]
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    
    aggregated = {"num_trials": len(results), "trials": results, "statistics": {}}
    
    for model_key in model_keys:
        aggregated["statistics"][model_key] = {}
        for metric in metric_keys:
            values = [r[model_key][metric] for r in results if model_key in r]
            if values:
                aggregated["statistics"][model_key][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
    
    return aggregated


def save_results(aggregated, output_dir):
    """Save results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)
    print(f"Results saved to {results_path}")


def save_best_models(best_info, output_dir):
    """Save the best model for each type (each may come from different trials)."""
    if best_info is None or "models" not in best_info:
        print("No best models to save")
        return
    
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"\nSaving best models (each from its best trial):")
    
    for model_type, model in best_info["models"].items():
        if model is not None:
            model_path = os.path.join(models_dir, f"{model_type}_model")
            model.save_pretrained(model_path)
            acc = best_info["accuracies"][model_type]
            seed = best_info["seeds"][model_type]
            print(f"  {model_type}_model: acc={acc:.4f} (seed={seed})")


def print_summary(aggregated):
    """Print summary of results."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    stats = aggregated.get("statistics", {})
    
    print(f"\nTrials: {aggregated.get('num_trials', 0)}")
    print()
    print(f"{'Model/Induction':<20} {'Accuracy':>15} {'Precision':>15} {'Recall':>15} {'F1':>15}")
    print("-" * 80)
    
    display_names = {
        "char": "Character",
        "couplet": "Couplet",
        "poem4": "Poem-4",
        "poem1": "Poem-1",
        "char_to_couplet": "Char → Couplet",
        "couplet_to_poem": "Coup → Poem",
        "poem4_to_poem": "Poem4 → Poem",
        "char_to_poem": "Char → Poem",
    }
    
    for key, name in display_names.items():
        if key in stats:
            s = stats[key]
            row = f"{name:<20}"
            for metric in ["accuracy", "precision", "recall", "f1"]:
                if metric in s:
                    row += f" {s[metric]['mean']:.4f}±{s[metric]['std']:.4f}"
                else:
                    row += f" {'N/A':>14}"
            print(row)
    
    print("-" * 80)
    
    # Print best models info
    best_models = aggregated.get("best_models", {})
    if best_models:
        print("\nBest models saved:")
        for model_type in ["char", "couplet", "poem4", "poem1"]:
            if model_type in best_models:
                info = best_models[model_type]
                print(f"  {display_names.get(model_type, model_type)}: "
                      f"acc={info['accuracy']:.4f} (seed={info['seed']})")
    
    print("=" * 80)


def main():
    default_output = os.path.join(PROJECT_ROOT, "results")
    default_data = os.path.join(PROJECT_ROOT, "data", "silver_standard_train.json")
    
    parser = argparse.ArgumentParser(description="Run parallelism model training trials")
    parser.add_argument("--trials", type=int, default=100, help="Target successful trials (default: 100)")
    parser.add_argument("--model-seed", type=int, default=1, help="Model initialization seed (default: 1)")
    parser.add_argument("--data-seed", type=int, default=100, help="Starting data seed (default: 100)")
    parser.add_argument("--data", type=str, default=default_data, help="Path to training data")
    parser.add_argument("--output", type=str, default=default_output, help="Output directory")
    args = parser.parse_args()
    
    device = get_device()
    print(f"Device: {device}")
    
    # Load data
    poems = load_silver_standard(args.data)
    print(f"Loaded {len(poems)} poems")
    
    # Create tokenizer
    tokenizer = create_tokenizer()
    tokenizer_path = os.path.join(args.output, "models", "tokenizer")
    os.makedirs(tokenizer_path, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")
    
    # Run trials
    results, failed, best_info = run_trials(
        poems, tokenizer, device,
        target_trials=args.trials,
        model_seed_start=args.model_seed,
        data_seed_start=args.data_seed
    )
    
    # Save best models (each from its best trial)
    save_best_models(best_info, args.output)
    
    # Clean up model references before saving JSON (models can't be serialized)
    for r in results:
        if "_models" in r:
            del r["_models"]
    
    # Aggregate and save
    aggregated = aggregate_results(results)
    aggregated["failed_seeds"] = failed
    
    # Add best model info for each type
    if best_info:
        aggregated["best_models"] = {
            model_type: {
                "accuracy": best_info["accuracies"][model_type],
                "seed": best_info["seeds"][model_type],
            }
            for model_type in best_info["accuracies"]
        }
    
    save_results(aggregated, args.output)
    
    # Print summary
    print_summary(aggregated)


if __name__ == "__main__":
    main()
