"""
Attention Visualization Experiment

Visualize attention distribution from [CLS] token to all poem tokens
for specific poems.

Prerequisites:
    1. Run poem1_trials.py first to train and save a model:
       python experiments/poem1_trials.py --trials 1 --epochs 1
    
    2. Run attention_correlation.py to analyze poems and create data.json:
       python experiments/attention_correlation.py

Usage:
    python experiments/attention_visualization.py --id 42              # Visualize poem with ID 42
    python experiments/attention_visualization.py --id 42 --heads 1 3 5  # Specific heads only
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import torch

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from train_utils import get_device
from transformers import BertForSequenceClassification, BertTokenizerFast

# Try to load fonts for Chinese text rendering
try:
    from qhchina.helpers import load_fonts
    load_fonts()
except ImportError:
    print("Warning: qhchina not installed, using default fonts")

# Directories
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "attention_output")
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts_poem1")
MODEL_DIR = os.path.join(ARTIFACTS_DIR, "model")
TOKENIZER_DIR = os.path.join(ARTIFACTS_DIR, "tokenizer")
DATA_FILE = os.path.join(ARTIFACTS_DIR, "data.json")


def check_artifacts_exist():
    """Check if all required artifacts exist."""
    missing = []
    
    if not os.path.exists(MODEL_DIR):
        missing.append(f"Model directory: {MODEL_DIR}")
    if not os.path.exists(TOKENIZER_DIR):
        missing.append(f"Tokenizer directory: {TOKENIZER_DIR}")
    if not os.path.exists(DATA_FILE):
        missing.append(f"Data file: {DATA_FILE}")
    
    if missing:
        error_msg = "Missing required artifacts:\n"
        for m in missing:
            error_msg += f"  - {m}\n"
        error_msg += "\nPlease run the following commands first:\n"
        error_msg += "  1. python experiments/poem1_trials.py --trials 1 --epochs 1\n"
        error_msg += "  2. python experiments/attention_correlation.py"
        raise FileNotFoundError(error_msg)


def load_artifacts(device):
    """Load saved model, tokenizer, and data."""
    check_artifacts_exist()
    
    print("Loading artifacts...")
    
    tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_DIR)
    print(f"  Loaded tokenizer from {TOKENIZER_DIR}")
    
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    print(f"  Loaded model from {MODEL_DIR}")
    
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"  Loaded {len(test_data)} test poems from {DATA_FILE}")
    
    return model, tokenizer, test_data


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
    
    plt.tight_layout()
    
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
    id_str = f" (ID: {poem_id})" if poem_id is not None else ""
    title = f"Head {head_idx + 1} — Pattern: {pattern_str}{id_str}"
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def find_poem_by_id(test_data, poem_id):
    """Find poem by ID in test data."""
    for item in test_data:
        if item["id"] == poem_id:
            return item
    return None


def analyze_single_poem(model, tokenizer, test_data, poem_id, device, heads=None):
    """Analyze and visualize a single poem by ID."""
    item = find_poem_by_id(test_data, poem_id)
    if item is None:
        # Try to find by index if ID not found
        if poem_id < len(test_data):
            item = test_data[poem_id]
            print(f"Warning: ID {poem_id} not found, using index {poem_id} instead")
        else:
            print(f"Error: ID {poem_id} not found in test data")
            print(f"Available IDs: 0-{len(test_data)-1}")
            return
    
    couplets = item["couplets"]
    line_match = item["line_match"]
    true_label = item["label"]
    label_name = "regulated" if true_label == 1 else "non-regulated"
    
    pred, attentions, input_ids = predict_with_attention(model, tokenizer, couplets, device)
    
    print(f"\n{label_name.upper()} Poem (ID: {item['id']}):")
    for i, (l1, l2) in enumerate(couplets):
        parallel = "∥" if line_match[i] == 1 else "≠"
        print(f"   Couplet {i+1}: {l1}，{l2} ({parallel})")
    print(f"True label: {true_label}, Predicted: {pred}")
    
    # Always generate full 12-head heatmap
    output_path = os.path.join(OUTPUT_DIR, f"attn_{item['id']}_full.png")
    visualize_attention_heatmaps(
        attentions, input_ids, tokenizer, couplets, line_match,
        output_path, poem_id=item['id']
    )
    
    # Generate individual heatmaps for specified heads
    if heads is not None:
        for head in heads:
            head_idx = head - 1  # Convert 1-indexed to 0-indexed
            output_path = os.path.join(OUTPUT_DIR, f"attn_{item['id']}_{head}.png")
            visualize_single_head_heatmap(
                attentions, input_ids, tokenizer, couplets, line_match,
                output_path, head_idx, poem_id=item['id']
            )


def main():
    parser = argparse.ArgumentParser(description="Visualize attention heatmaps for poems")
    parser.add_argument("--id", type=int, required=True,
                        help="ID of poem to visualize")
    parser.add_argument("--heads", type=int, nargs="+", default=None,
                        help="List of head indices (1-12) to visualize individually")
    args = parser.parse_args()
    
    # Validate head arguments
    if args.heads is not None:
        for h in args.heads:
            if h < 1 or h > 12:
                raise ValueError(f"Head index {h} out of range. Must be 1-12.")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = get_device()
    print(f"Device: {device}")
    
    # Load artifacts
    print("\n=== Loading artifacts ===")
    model, tokenizer, test_data = load_artifacts(device)
    
    # Generate heatmaps for specified poem
    print("\n=== Generating heatmaps ===")
    analyze_single_poem(model, tokenizer, test_data, args.id, device, heads=args.heads)
    
    print("\nDone!")
    print(f"Test data has {len(test_data)} poems")


if __name__ == "__main__":
    main()
