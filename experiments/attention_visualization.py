"""
Attention Visualization Experiment

Train a poem1 model and visualize attention distribution from [CLS] token
to all poem tokens for successfully predicted examples.

Usage:
    python experiments/attention_visualization.py                    # Random selection
    python experiments/attention_visualization.py --reg 42           # Specific regulated poem
    python experiments/attention_visualization.py --nonreg 100       # Specific non-regulated poem
    python experiments/attention_visualization.py --reg 42 --heads 1 3 5  # Specific heads only
"""

import argparse
import json
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from datasets import PoemDataset1Label
from train_utils import get_device, train_model, PRETRAINED_MODEL_NAME
from transformers import BertForSequenceClassification, BertTokenizerFast, set_seed

# Try to load fonts for Chinese text rendering
try:
    from qhchina.helpers import load_fonts
    load_fonts()
except ImportError:
    print("Warning: qhchina not installed, using default fonts")

# =============================================================================
# CONFIGURATION
# =============================================================================
TOTAL_POEMS = 15000  # 7500 regulated + 7500 non-regulated
TRAIN_RATIO = 2/3    # 10000 train, 5000 test
DATA_SEED = 42
MODEL_SEED = 42
EPOCHS = 1

# =============================================================================

# Directories
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "attention_output")
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")
MODEL_DIR = os.path.join(ARTIFACTS_DIR, "model")
TOKENIZER_DIR = os.path.join(ARTIFACTS_DIR, "tokenizer")
DATA_FILE = os.path.join(ARTIFACTS_DIR, "data.json")


def load_poems():
    """Load poems from silver_standard_train.json."""
    path = os.path.join(PROJECT_ROOT, "data", "silver_standard_train.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_couplet_signature(couplet):
    """Create unique signature for a couplet to check overlap."""
    return couplet[0] + "||" + couplet[1]


def is_regulated(poem):
    """Check if poem is regulated (both inner couplets are parallel)."""
    line_match = poem["line_match"]
    return line_match[1] == 1 and line_match[2] == 1


def collect_balanced_poems(poems, total_count, seed):
    """Collect balanced set of regulated and non-regulated poems."""
    random.seed(seed)
    
    regulated = [p for p in poems if is_regulated(p)]
    non_regulated = [p for p in poems if not is_regulated(p)]
    
    print(f"Available: {len(regulated)} regulated, {len(non_regulated)} non-regulated")
    
    random.shuffle(regulated)
    random.shuffle(non_regulated)
    
    n_each = total_count // 2
    selected_regulated = regulated[:n_each]
    selected_non_regulated = non_regulated[:n_each]
    
    print(f"Selected: {len(selected_regulated)} regulated, {len(selected_non_regulated)} non-regulated")
    
    balanced = selected_regulated + selected_non_regulated
    random.shuffle(balanced)
    
    return balanced


def split_no_couplet_overlap(poems, train_ratio, seed):
    """Split poems into train/test ensuring no couplet overlap."""
    random.seed(seed)
    poems_shuffled = list(poems)
    random.shuffle(poems_shuffled)
    
    target_test = round(len(poems) * (1 - train_ratio))  # Use round() to avoid float precision issues
    
    test_poems = []
    test_couplet_sigs = set()
    train_poems = []
    train_couplet_sigs = set()
    
    for poem in poems_shuffled:
        if len(test_poems) >= target_test:
            break
        
        poem_sigs = [get_couplet_signature(c) for c in poem["couplets"]]
        has_overlap = any(sig in train_couplet_sigs for sig in poem_sigs)
        
        if not has_overlap:
            test_poems.append(poem)
            test_couplet_sigs.update(poem_sigs)
    
    for poem in poems_shuffled:
        if poem in test_poems:
            continue
            
        poem_sigs = [get_couplet_signature(c) for c in poem["couplets"]]
        has_overlap = any(sig in test_couplet_sigs for sig in poem_sigs)
        
        if not has_overlap:
            train_poems.append(poem)
            train_couplet_sigs.update(poem_sigs)
    
    train_reg = sum(1 for p in train_poems if is_regulated(p))
    train_nonreg = len(train_poems) - train_reg
    test_reg = sum(1 for p in test_poems if is_regulated(p))
    test_nonreg = len(test_poems) - test_reg
    
    print(f"Train: {len(train_poems)} poems ({train_reg} reg, {train_nonreg} non-reg)")
    print(f"Test: {len(test_poems)} poems ({test_reg} reg, {test_nonreg} non-reg)")
    
    overlap = train_couplet_sigs & test_couplet_sigs
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping couplets!"
    
    return train_poems, test_poems


def create_poem1_data(poems):
    """Create poem 1-label data (regulated if inner couplets are parallel)."""
    data = []
    for idx, poem in enumerate(poems):
        if len(poem["couplets"]) == 4 and len(poem["line_match"]) == 4:
            label = 1 if is_regulated(poem) else 0
            data.append({
                "id": idx,
                "couplets": poem["couplets"],
                "label": label,
                "line_match": poem["line_match"]
            })
    return data


def save_artifacts(model, tokenizer, train_data, test_data):
    """Save model, tokenizer, and data for future reuse."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    model.save_pretrained(MODEL_DIR)
    print(f"Saved model to {MODEL_DIR}")
    
    tokenizer.save_pretrained(TOKENIZER_DIR)
    print(f"Saved tokenizer to {TOKENIZER_DIR}")
    
    data = {"train": train_data, "test": test_data}
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved data to {DATA_FILE}")


def load_artifacts(device):
    """Load saved model, tokenizer, and data."""
    print("Loading saved artifacts...")
    
    tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_DIR)
    print(f"Loaded tokenizer from {TOKENIZER_DIR}")
    
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    print(f"Loaded model from {MODEL_DIR}")
    
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    train_data = data["train"]
    test_data = data["test"]
    print(f"Loaded data: {len(train_data)} train, {len(test_data)} test")
    
    return model, tokenizer, train_data, test_data


def artifacts_exist():
    """Check if all artifacts exist."""
    return (os.path.exists(MODEL_DIR) and 
            os.path.exists(TOKENIZER_DIR) and 
            os.path.exists(DATA_FILE))


def predict_with_attention(model, tokenizer, couplets, device):
    """Get prediction and attention weights."""
    text = "".join([l1 + "，" + l2 + "。" for l1, l2 in couplets])
    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=52,
        return_tensors="pt"
    ).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(**encoded, output_attentions=True)
        pred = outputs.logits.argmax(dim=-1).item()
        attentions = torch.stack(outputs.attentions, dim=0).squeeze(1)
    
    return pred, attentions, encoded["input_ids"].squeeze(0)


def visualize_attention_heatmaps(attentions, input_ids, tokenizer, couplets, 
                                  line_match, output_path, poem_id=None, layer=-1):
    """Create 12 heatmap visualizations (one per attention head)."""
    layer_attention = attentions[layer]  
    num_heads = layer_attention.shape[0]
    cls_attention = layer_attention[:, 0, :]
    
    # Extract attention to poem tokens (skip [CLS] at position 0)
    # Each couplet has 12 tokens: 5 chars + comma + 5 chars + period
    full_attention = cls_attention[:, 1:49].cpu().numpy()  # (num_heads, 48)
    poem_attention_reshaped = full_attention.reshape(num_heads, 4, 12)  # Keep all 12 columns
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    axes = axes.flatten()
    
    y_labels = []
    for i, match in enumerate(line_match):
        status = "∥" if match == 1 else "≠"
        y_labels.append(f"C{i+1} ({status})")
    
    # X-axis labels: 1 2 3 4 5 ， 1 2 3 4 5 。
    x_labels = ["1", "2", "3", "4", "5", "，", "1", "2", "3", "4", "5", "。"]
    
    for head_idx in range(num_heads):
        ax = axes[head_idx]
        heatmap_data = poem_attention_reshaped[head_idx]
        
        # Each head gets its own color normalization (no shared vmin/vmax)
        ax.imshow(heatmap_data, cmap='Blues', aspect='auto')
        
        ax.set_xticks(range(12))
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_yticks(range(4))
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.set_title(f"Head {head_idx + 1}", fontsize=11, fontweight='bold')
    
    # Build title - poem on separate lines
    pattern_str = "".join(str(m) for m in line_match)
    poem_lines = []
    for i, (l1, l2) in enumerate(couplets):
        status = "∥" if line_match[i] == 1 else "≠"
        poem_lines.append(f"{l1}，{l2}。 ({status})")
    poem_display = "\n".join(poem_lines)
    
    id_str = f" (Test ID: {poem_id})" if poem_id is not None else ""
    title = f"Attention from [CLS] — Pattern: {pattern_str}{id_str}\n{poem_display}"
    
    # Use fig.text for title positioning instead of suptitle
    fig.text(0.5, 0.98, title, ha='center', va='top', fontsize=12, linespacing=1.2)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.05)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def visualize_single_head_heatmap(attentions, input_ids, tokenizer, couplets,
                                   line_match, output_path, head_idx, poem_id=None, layer=-1):
    """Create a single heatmap visualization for one attention head."""
    layer_attention = attentions[layer]
    cls_attention = layer_attention[:, 0, :]
    
    # Extract attention to poem tokens (skip [CLS] at position 0)
    full_attention = cls_attention[:, 1:49].cpu().numpy()  # (num_heads, 48)
    poem_attention_reshaped = full_attention.reshape(12, 4, 12)  # (num_heads, 4 couplets, 12 tokens)
    
    fig, ax = plt.subplots(figsize=(8, 3))
    
    y_labels = []
    for i, match in enumerate(line_match):
        status = "∥" if match == 1 else "≠"
        y_labels.append(f"C{i+1} ({status})")
    
    x_labels = ["1", "2", "3", "4", "5", "，", "1", "2", "3", "4", "5", "。"]
    
    heatmap_data = poem_attention_reshaped[head_idx]
    im = ax.imshow(heatmap_data, cmap='Blues', aspect='auto')
    
    ax.set_xticks(range(12))
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_yticks(range(4))
    ax.set_yticklabels(y_labels, fontsize=10)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Build title (no poem text)
    pattern_str = "".join(str(m) for m in line_match)
    id_str = f" (Test ID: {poem_id})" if poem_id is not None else ""
    title = f"Head {head_idx + 1} — Pattern: {pattern_str}{id_str}"
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def analyze_single_poem(model, tokenizer, test_data, poem_id, device, file_prefix, heads=None):
    """Analyze and visualize a single poem by ID."""
    if poem_id < 0 or poem_id >= len(test_data):
        print(f"Error: ID {poem_id} out of range (0-{len(test_data)-1})")
        return
    
    item = test_data[poem_id]
    couplets = item["couplets"]
    line_match = item["line_match"]
    true_label = item["label"]
    label_name = "regulated" if true_label == 1 else "non-regulated"
    
    pred, attentions, input_ids = predict_with_attention(model, tokenizer, couplets, device)
    
    print(f"\n{label_name.upper()} Poem (ID: {poem_id}):")
    for i, (l1, l2) in enumerate(couplets):
        parallel = "∥" if line_match[i] == 1 else "≠"
        print(f"   Couplet {i+1}: {l1}，{l2} ({parallel})")
    print(f"True label: {true_label}, Predicted: {pred}")
    
    # Always generate full 12-head heatmap
    output_path = os.path.join(OUTPUT_DIR, f"{file_prefix}_{poem_id}_full.png")
    visualize_attention_heatmaps(
        attentions, input_ids, tokenizer, couplets, line_match,
        output_path, poem_id=poem_id
    )
    
    # Generate individual heatmaps for specified heads
    if heads is not None:
        for head in heads:
            head_idx = head - 1  # Convert 1-indexed to 0-indexed
            output_path = os.path.join(OUTPUT_DIR, f"{file_prefix}_{poem_id}_{head}.png")
            visualize_single_head_heatmap(
                attentions, input_ids, tokenizer, couplets, line_match,
                output_path, head_idx, poem_id=poem_id
            )


def main():
    parser = argparse.ArgumentParser(description="Visualize attention heatmaps for poems")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--reg", type=int, default=None, 
                        help="ID of regulated poem to analyze")
    group.add_argument("--nonreg", type=int, default=None,
                        help="ID of non-regulated poem to analyze")
    parser.add_argument("--heads", type=int, nargs="+", default=None,
                        help="List of head indices (1-12) to visualize individually")
    parser.add_argument("--retrain", action="store_true",
                        help="Force retraining with a new random seed")
    args = parser.parse_args()
    
    # Validate arguments
    if args.retrain and (args.reg is not None or args.nonreg is not None):
        raise ValueError("Cannot use --retrain with --reg or --nonreg. "
                        "Retrain first, then run again with specific IDs.")
    
    if args.heads is not None:
        if args.reg is None and args.nonreg is None:
            raise ValueError("--heads requires either --reg or --nonreg to be specified.")
        for h in args.heads:
            if h < 1 or h > 12:
                raise ValueError(f"Head index {h} out of range. Must be 1-12.")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = get_device()
    print(f"Device: {device}")
    
    # Check if artifacts exist (or force retrain)
    if artifacts_exist() and not args.retrain:
        print("\n=== Loading saved artifacts ===")
        model, tokenizer, train_data, test_data = load_artifacts(device)
    else:
        if args.retrain:
            print("\n=== Retraining model (--retrain flag) ===")
        else:
            print("\n=== Training new model ===")
        
        print("\n1. Loading poems...")
        all_poems = load_poems()
        print(f"Total poems available: {len(all_poems)}")
        
        print("\n2. Collecting balanced poems...")
        balanced_poems = collect_balanced_poems(all_poems, TOTAL_POEMS, DATA_SEED)
        
        print("\n3. Splitting into train/test (no couplet overlap)...")
        train_poems, test_poems = split_no_couplet_overlap(balanced_poems, TRAIN_RATIO, DATA_SEED)
        
        train_data = create_poem1_data(train_poems)
        test_data = create_poem1_data(test_poems)
        
        print(f"\nFinal: {len(train_data)} train, {len(test_data)} test")
        
        print("\n4. Creating tokenizer and datasets...")
        # Use plain tokenizer without special [CP1]-[CP4] tokens (those are for poem4 model only)
        tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_MODEL_NAME)
        train_ds = PoemDataset1Label(train_data, tokenizer)
        
        print("\n5. Training poem1 model...")
        # Use random seed if retraining, otherwise use fixed seed
        if args.retrain:
            model_seed = random.randint(1, 100000)
            print(f"   Using random seed: {model_seed}")
        else:
            model_seed = MODEL_SEED
        set_seed(model_seed)
        model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=2)
        
        model = train_model(model, train_ds, epochs=EPOCHS, device=device)
        
        print("\n6. Saving artifacts...")
        save_artifacts(model, tokenizer, train_data, test_data)
    
    # If specific IDs are provided, just analyze those poems directly
    if args.reg is not None or args.nonreg is not None:
        print("\n=== Generating heatmaps for specified poem ===")
        
        if args.reg is not None:
            analyze_single_poem(model, tokenizer, test_data, args.reg, device, 
                              "attn_reg", heads=args.heads)
        
        if args.nonreg is not None:
            analyze_single_poem(model, tokenizer, test_data, args.nonreg, device,
                              "attn_nonreg", heads=args.heads)
    else:
        # No IDs provided - evaluate all and pick random correct predictions
        print("\n=== Evaluating test data for random selection ===")
        evaluated = []
        for item in tqdm(test_data, desc="Evaluating"):
            couplets = item["couplets"]
            true_label = item["label"]
            pred, _, _ = predict_with_attention(model, tokenizer, couplets, device)
            evaluated.append({
                **item,
                "pred": pred,
                "correct": pred == true_label
            })
        
        correct_regulated = [e for e in evaluated if e["correct"] and e["label"] == 1]
        correct_nonregulated = [e for e in evaluated if e["correct"] and e["label"] == 0]
        
        print(f"Correctly predicted: {len(correct_regulated)} regulated, {len(correct_nonregulated)} non-regulated")
        
        print("\n=== Generating heatmaps ===")
        
        # Random regulated
        if len(correct_regulated) > 0:
            item = random.choice(correct_regulated)
            couplets = item["couplets"]
            line_match = item["line_match"]
            poem_id = item["id"]
            
            pred, attentions, input_ids = predict_with_attention(model, tokenizer, couplets, device)
            
            print(f"\nREGULATED Poem (ID: {poem_id}):")
            for i, (l1, l2) in enumerate(couplets):
                parallel = "∥" if line_match[i] == 1 else "≠"
                print(f"   Couplet {i+1}: {l1}，{l2} ({parallel})")
            print(f"True label: {item['label']}, Predicted: {pred}")
            
            output_path = os.path.join(OUTPUT_DIR, f"attn_reg_{poem_id}_full.png")
            visualize_attention_heatmaps(attentions, input_ids, tokenizer, couplets, 
                                        line_match, output_path, poem_id=poem_id)
        else:
            print("No correctly predicted regulated poems found!")
        
        # Random non-regulated
        if len(correct_nonregulated) > 0:
            item = random.choice(correct_nonregulated)
            couplets = item["couplets"]
            line_match = item["line_match"]
            poem_id = item["id"]
            
            pred, attentions, input_ids = predict_with_attention(model, tokenizer, couplets, device)
            
            print(f"\nNON-REGULATED Poem (ID: {poem_id}):")
            for i, (l1, l2) in enumerate(couplets):
                parallel = "∥" if line_match[i] == 1 else "≠"
                print(f"   Couplet {i+1}: {l1}，{l2} ({parallel})")
            print(f"True label: {item['label']}, Predicted: {pred}")
            
            output_path = os.path.join(OUTPUT_DIR, f"attn_nonreg_{poem_id}_full.png")
            visualize_attention_heatmaps(attentions, input_ids, tokenizer, couplets,
                                        line_match, output_path, poem_id=poem_id)
        else:
            print("No correctly predicted non-regulated poems found!")
        
        print(f"\n  - Regulated (label=1): {sum(1 for e in evaluated if e['label']==1)} poems")
        print(f"  - Non-regulated (label=0): {sum(1 for e in evaluated if e['label']==0)} poems")
    
    print("\nDone!")
    print(f"Test data has {len(test_data)} poems (IDs: 0-{len(test_data)-1})")


if __name__ == "__main__":
    main()
