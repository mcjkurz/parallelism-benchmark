"""
Attention Correlation Experiment

Find regulated poems where attention to the first line correlates highly
with attention to the second line in the inner couplets (couplets 2 and 3).

Uses Spearman rank correlation to measure similarity of attention patterns.

Prerequisites:
    - Run poem1_trials.py first to train and save a model to experiments/artifacts/

Usage:
    python experiments/attention_correlation.py
"""

import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from tqdm.auto import tqdm

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from transformers import BertForSequenceClassification, BertTokenizerFast
from train_utils import get_device

# Directories
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")
MODEL_DIR = os.path.join(ARTIFACTS_DIR, "model")
TOKENIZER_DIR = os.path.join(ARTIFACTS_DIR, "tokenizer")
DATA_FILE = os.path.join(ARTIFACTS_DIR, "data.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "attention_output")

# Configuration
TOP_N = 20  # Number of top poems to show
MAX_PUNCT_HEADS = 4  # Filter out poems where more than this many heads focus on punctuation
PUNCT_THRESHOLD = 0.40  # Attention ratio threshold for "punctuation-focused" head (40% attention on punct)
MIN_COUPLET_BALANCE = 0.25  # Minimum attention ratio for the less-attended inner couplet


def check_model_exists():
    """Check if trained model exists in artifacts directory."""
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(
            f"Model not found at {MODEL_DIR}\n"
            "Please run poem1_trials.py first to train a model:\n"
            "  python experiments/poem1_trials.py --trials 1 --epochs 1"
        )
    if not os.path.exists(TOKENIZER_DIR):
        raise FileNotFoundError(
            f"Tokenizer not found at {TOKENIZER_DIR}\n"
            "Please run poem1_trials.py first to train a model:\n"
            "  python experiments/poem1_trials.py --trials 1 --epochs 1"
        )


def load_model(device):
    """Load saved model and tokenizer from artifacts."""
    check_model_exists()
    
    print("Loading model and tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    print(f"  Loaded model from {MODEL_DIR}")
    print(f"  Loaded tokenizer from {TOKENIZER_DIR}")
    
    return model, tokenizer


def load_test_poems():
    """Load test poems from silver_standard_test.json."""
    test_path = os.path.join(PROJECT_ROOT, "data", "silver_standard_test.json")
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Test data not found at {test_path}\n"
            "Please run prepare_data.py first to generate test data:\n"
            "  python scripts/prepare_data.py"
        )
    
    with open(test_path, "r", encoding="utf-8") as f:
        poems = json.load(f)
    
    # Convert to expected format with IDs
    test_data = []
    for idx, poem in enumerate(poems):
        if len(poem["couplets"]) == 4 and len(poem["line_match"]) == 4:
            # Regulated if inner couplets (indices 1,2) are both parallel
            label = 1 if (poem["line_match"][1] == 1 and poem["line_match"][2] == 1) else 0
            test_data.append({
                "id": idx,
                "couplets": poem["couplets"],
                "label": label,
                "line_match": poem["line_match"]
            })
    
    return test_data


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


def get_couplet_attention_correlation(attentions, couplet_idx, layer=-1):
    """Calculate Spearman and Pearson correlation between attention to first and second lines.
    
    Args:
        attentions: Tensor of shape (num_layers, num_heads, seq_len, seq_len)
        couplet_idx: Which couplet to analyze (0-3)
        layer: Which layer to use (-1 = last layer)
        
    Returns:
        spearman_correlations: List of (head_idx, correlation, p_value) for each head
        mean_spearman: Mean Spearman correlation across all heads
        pearson_correlations: List of (head_idx, correlation, p_value) for each head
        mean_pearson: Mean Pearson correlation across all heads
        line1_attention: Attention to first 5 chars
        line2_attention: Attention to second 5 chars
    """
    # Get last layer attention from [CLS]
    layer_attention = attentions[layer]  # (num_heads, seq_len, seq_len)
    num_heads = layer_attention.shape[0]
    cls_attention = layer_attention[:, 0, :]  # (num_heads, seq_len)
    
    # Calculate token positions for the specified couplet
    # Structure: [CLS] + couplet0 (12 tokens) + couplet1 (12 tokens) + ...
    # Each couplet: 5 chars + comma + 5 chars + period = 12 tokens
    couplet_start = 1 + couplet_idx * 12  # +1 to skip [CLS]
    
    # First line: positions 0-4 within couplet (5 chars)
    # Comma: position 5
    # Second line: positions 6-10 within couplet (5 chars)
    # Period: position 11
    
    line1_start = couplet_start
    line1_end = couplet_start + 5
    line2_start = couplet_start + 6  # Skip comma at position 5
    line2_end = couplet_start + 11
    
    line1_attention = cls_attention[:, line1_start:line1_end].cpu().numpy()  # (num_heads, 5)
    line2_attention = cls_attention[:, line2_start:line2_end].cpu().numpy()  # (num_heads, 5)
    
    # Calculate Spearman and Pearson correlation for each head
    spearman_correlations = []
    pearson_correlations = []
    for head_idx in range(num_heads):
        # Spearman
        sp_corr, sp_p = spearmanr(line1_attention[head_idx], line2_attention[head_idx])
        spearman_correlations.append((head_idx, sp_corr, sp_p))
        # Pearson
        pe_corr, pe_p = pearsonr(line1_attention[head_idx], line2_attention[head_idx])
        pearson_correlations.append((head_idx, pe_corr, pe_p))
    
    # Mean correlations across heads
    mean_spearman = np.mean([c[1] for c in spearman_correlations if not np.isnan(c[1])])
    mean_pearson = np.mean([c[1] for c in pearson_correlations if not np.isnan(c[1])])
    
    return spearman_correlations, mean_spearman, pearson_correlations, mean_pearson, line1_attention, line2_attention


def count_punctuation_focused_heads(attentions, layer=-1, threshold=0.15):
    """Count heads where attention to punctuation (commas/periods) is high.
    
    A head is considered "punctuation-focused" if the sum of attention to all
    commas and periods exceeds the threshold (relative to total sequence).
    
    Args:
        attentions: Tensor of shape (num_layers, num_heads, seq_len, seq_len)
        layer: Which layer to use (-1 = last layer)
        threshold: Minimum attention ratio on punctuation to be considered focused
        
    Returns:
        num_focused_heads: Number of heads focused on punctuation
        head_punct_scores: List of (head_idx, punct_attention_ratio) for all heads
    """
    layer_attention = attentions[layer]  # (num_heads, seq_len, seq_len)
    num_heads = layer_attention.shape[0]
    cls_attention = layer_attention[:, 0, :]  # (num_heads, seq_len)
    
    # Structure: [CLS] + 4 couplets * 12 tokens each + [SEP]
    # Each couplet: 5 chars + comma + 5 chars + period
    # Punctuation positions (0-indexed from start of sequence):
    # Comma positions: 6, 18, 30, 42 (couplet_start + 5)
    # Period positions: 12, 24, 36, 48 (couplet_start + 11)
    punct_positions = []
    for couplet_idx in range(4):
        couplet_start = 1 + couplet_idx * 12
        punct_positions.append(couplet_start + 5)   # comma
        punct_positions.append(couplet_start + 11)  # period
    
    head_punct_scores = []
    num_focused = 0
    
    for head_idx in range(num_heads):
        head_att = cls_attention[head_idx].cpu().numpy()
        # Sum attention on punctuation positions
        punct_attention = sum(head_att[pos] for pos in punct_positions if pos < len(head_att))
        # Total attention (excluding [CLS] and [SEP] for fairer comparison)
        total_attention = head_att[1:-1].sum()
        ratio = punct_attention / total_attention if total_attention > 0 else 0
        
        head_punct_scores.append((head_idx, ratio))
        if ratio > threshold:
            num_focused += 1
    
    return num_focused, head_punct_scores


def get_inner_couplet_balance(attentions, layer=-1):
    """Calculate attention balance between the two inner couplets (2nd and 3rd).
    
    Returns the ratio of attention on each inner couplet relative to the total
    attention on both inner couplets combined.
    
    Args:
        attentions: Tensor of shape (num_layers, num_heads, seq_len, seq_len)
        layer: Which layer to use (-1 = last layer)
        
    Returns:
        couplet2_ratio: Ratio of attention on couplet 2 (vs both inner couplets)
        couplet3_ratio: Ratio of attention on couplet 3 (vs both inner couplets)
        min_ratio: The smaller of the two ratios (balance metric)
        per_head_balance: List of (head_idx, c2_ratio, c3_ratio) for each head
    """
    layer_attention = attentions[layer]  # (num_heads, seq_len, seq_len)
    num_heads = layer_attention.shape[0]
    cls_attention = layer_attention[:, 0, :]  # (num_heads, seq_len)
    
    # Couplet token positions (excluding punctuation for content-only comparison)
    # Couplet 2 (index 1): tokens 13-17 (line1) and 19-23 (line2)
    # Couplet 3 (index 2): tokens 25-29 (line1) and 31-35 (line2)
    c2_start = 1 + 1 * 12  # = 13
    c3_start = 1 + 2 * 12  # = 25
    
    # Content positions (excluding comma at +5 and period at +11)
    c2_positions = list(range(c2_start, c2_start + 5)) + list(range(c2_start + 6, c2_start + 11))
    c3_positions = list(range(c3_start, c3_start + 5)) + list(range(c3_start + 6, c3_start + 11))
    
    per_head_balance = []
    total_c2_att = 0
    total_c3_att = 0
    
    for head_idx in range(num_heads):
        head_att = cls_attention[head_idx].cpu().numpy()
        c2_att = sum(head_att[pos] for pos in c2_positions if pos < len(head_att))
        c3_att = sum(head_att[pos] for pos in c3_positions if pos < len(head_att))
        
        total = c2_att + c3_att
        c2_ratio = c2_att / total if total > 0 else 0.5
        c3_ratio = c3_att / total if total > 0 else 0.5
        
        per_head_balance.append((head_idx, c2_ratio, c3_ratio))
        total_c2_att += c2_att
        total_c3_att += c3_att
    
    # Overall balance across all heads
    total_inner = total_c2_att + total_c3_att
    overall_c2_ratio = total_c2_att / total_inner if total_inner > 0 else 0.5
    overall_c3_ratio = total_c3_att / total_inner if total_inner > 0 else 0.5
    min_ratio = min(overall_c2_ratio, overall_c3_ratio)
    
    return overall_c2_ratio, overall_c3_ratio, min_ratio, per_head_balance


def analyze_poems(poems, model, tokenizer, device, desc="Analyzing"):
    """Analyze a list of poems and return results with correlations."""
    results = []
    
    for item in tqdm(poems, desc=desc):
        couplets = item["couplets"]
        
        # Get attention
        pred, attentions, input_ids = predict_with_attention(
            model, tokenizer, couplets, device
        )
        
        # Calculate correlations for both inner couplets (indices 1 and 2)
        sp_corrs_c2, mean_sp_c2, pe_corrs_c2, mean_pe_c2, l1_att_c2, l2_att_c2 = \
            get_couplet_attention_correlation(attentions, couplet_idx=1)
        sp_corrs_c3, mean_sp_c3, pe_corrs_c3, mean_pe_c3, l1_att_c3, l2_att_c3 = \
            get_couplet_attention_correlation(attentions, couplet_idx=2)
        
        # Average across both inner couplets
        mean_spearman = (mean_sp_c2 + mean_sp_c3) / 2
        mean_pearson = (mean_pe_c2 + mean_pe_c3) / 2
        
        # Count punctuation-focused heads
        num_punct_heads, punct_scores = count_punctuation_focused_heads(
            attentions, threshold=PUNCT_THRESHOLD
        )
        
        # Calculate inner couplet attention balance
        c2_ratio, c3_ratio, min_balance, head_balance = get_inner_couplet_balance(attentions)
        
        # Find max correlations
        max_sp_head = max(sp_corrs_c2 + sp_corrs_c3, key=lambda x: x[1] if not np.isnan(x[1]) else -1)
        max_pe_head = max(pe_corrs_c2 + pe_corrs_c3, key=lambda x: x[1] if not np.isnan(x[1]) else -1)
        
        results.append({
            "id": item["id"],
            "couplets": couplets,
            "line_match": item["line_match"],
            "label": item["label"],
            "pred": pred,
            "correct": pred == item["label"],
            "mean_spearman": mean_spearman,
            "mean_pearson": mean_pearson,
            "mean_spearman_c2": mean_sp_c2,
            "mean_spearman_c3": mean_sp_c3,
            "mean_pearson_c2": mean_pe_c2,
            "mean_pearson_c3": mean_pe_c3,
            "max_spearman": max_sp_head[1],
            "max_pearson": max_pe_head[1],
            "spearman_correlations_c2": sp_corrs_c2,
            "spearman_correlations_c3": sp_corrs_c3,
            "pearson_correlations_c2": pe_corrs_c2,
            "pearson_correlations_c3": pe_corrs_c3,
            "num_punct_heads": num_punct_heads,
            "punct_scores": punct_scores,
            "couplet2_ratio": c2_ratio,
            "couplet3_ratio": c3_ratio,
            "min_couplet_balance": min_balance,
            "head_balance": head_balance,
        })
    
    return results


def result_to_json(r):
    """Convert result to JSON-serializable format."""
    def safe_float(v):
        return float(v) if not np.isnan(v) else None
    return {
        "id": r["id"],
        "couplets": r["couplets"],
        "line_match": r["line_match"],
        "label": r["label"],
        "pred": r["pred"],
        "correct": r["correct"],
        "mean_spearman": safe_float(r["mean_spearman"]),
        "mean_pearson": safe_float(r["mean_pearson"]),
        "mean_spearman_c2": safe_float(r["mean_spearman_c2"]),
        "mean_spearman_c3": safe_float(r["mean_spearman_c3"]),
        "mean_pearson_c2": safe_float(r["mean_pearson_c2"]),
        "mean_pearson_c3": safe_float(r["mean_pearson_c3"]),
        "max_spearman": safe_float(r["max_spearman"]),
        "max_pearson": safe_float(r["max_pearson"]),
        "num_punct_heads": r["num_punct_heads"],
        "punct_scores": [(h, float(s)) for h, s in r["punct_scores"]],
        "couplet2_ratio": float(r["couplet2_ratio"]),
        "couplet3_ratio": float(r["couplet3_ratio"]),
        "min_couplet_balance": float(r["min_couplet_balance"]),
    }


def save_data_json(test_data):
    """Save test data to data.json for visualization script."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f"Saved test data to {DATA_FILE}")


def main():
    device = get_device()
    print(f"Device: {device}")
    
    # Load model and tokenizer
    model, tokenizer = load_model(device)
    
    # Load test poems from silver_standard_test.json
    print("\nLoading test poems from silver_standard_test.json...")
    test_data = load_test_poems()
    print(f"Loaded {len(test_data)} test poems")
    
    # Save test data to data.json (for visualization script)
    save_data_json(test_data)
    
    # Split by label
    regulated_poems = [p for p in test_data if p["label"] == 1]
    nonregulated_poems = [p for p in test_data if p["label"] == 0]
    print(f"Found {len(regulated_poems)} regulated poems, {len(nonregulated_poems)} non-regulated poems")
    
    # Analyze both groups
    print(f"\nAnalyzing attention correlations for inner couplets (2 and 3)...")
    results_regulated = analyze_poems(regulated_poems, model, tokenizer, device, "Regulated")
    results_nonregulated = analyze_poems(nonregulated_poems, model, tokenizer, device, "Non-regulated")
    
    # Combined results for regulated (for backward compatibility)
    results = results_regulated
    
    # ========================================================================
    # COMPARISON: REGULATED vs NON-REGULATED (correctly identified)
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"CORRELATION COMPARISON: REGULATED vs NON-REGULATED")
    print(f"{'='*80}")
    
    # Get correctly identified poems from each group
    correct_regulated = [r for r in results_regulated if r["correct"]]
    correct_nonregulated = [r for r in results_nonregulated if r["correct"]]
    
    print(f"\nCorrectly identified regulated: {len(correct_regulated)}/{len(results_regulated)}")
    print(f"Correctly identified non-regulated: {len(correct_nonregulated)}/{len(results_nonregulated)}")
    
    # Calculate average correlations
    def safe_mean(values):
        valid = [v for v in values if not np.isnan(v)]
        return np.mean(valid) if valid else float('nan')
    
    reg_spearman = safe_mean([r["mean_spearman"] for r in correct_regulated])
    reg_pearson = safe_mean([r["mean_pearson"] for r in correct_regulated])
    nonreg_spearman = safe_mean([r["mean_spearman"] for r in correct_nonregulated])
    nonreg_pearson = safe_mean([r["mean_pearson"] for r in correct_nonregulated])
    
    spearman_diff = reg_spearman - nonreg_spearman
    pearson_diff = reg_pearson - nonreg_pearson
    
    print(f"\n{'Metric':<20} {'Regulated':<12} {'Non-regulated':<15} {'Difference':<12}")
    print(f"{'-'*60}")
    print(f"{'Avg Spearman ρ':<20} {reg_spearman:>10.4f}   {nonreg_spearman:>12.4f}   {spearman_diff:>+10.4f}")
    print(f"{'Avg Pearson r':<20} {reg_pearson:>10.4f}   {nonreg_pearson:>12.4f}   {pearson_diff:>+10.4f}")
    
    # ========================================================================
    # MULTI-STEP FILTERING PIPELINE (for regulated poems)
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"MULTI-STEP FILTERING PIPELINE (Regulated Poems)")
    print(f"{'='*80}")
    
    # Step 0: All analyzed regulated poems
    print(f"\nStep 0: Total analyzed poems: {len(results)}")
    
    # Step 1: Filter for correctly predicted poems only
    step1_results = [r for r in results if r["correct"]]
    print(f"Step 1: Correctly predicted: {len(step1_results)} poems")
    
    # Step 2: Filter out poems where more than MAX_PUNCT_HEADS focus on punctuation
    step2_results = [r for r in step1_results if r["num_punct_heads"] <= MAX_PUNCT_HEADS]
    print(f"Step 2: Punctuation filter (≤{MAX_PUNCT_HEADS} heads at {PUNCT_THRESHOLD:.0%}): {len(step2_results)} poems")
    
    # Step 3: Filter out poems where attention focuses on one couplet, not two
    step3_results = [r for r in step2_results if r["min_couplet_balance"] >= MIN_COUPLET_BALANCE]
    print(f"Step 3: Balance filter (min couplet ratio ≥{MIN_COUPLET_BALANCE:.0%}): {len(step3_results)} poems")
    
    # Step 4: Sort by Pearson correlation (descending)
    results_filtered = sorted(
        step3_results, 
        key=lambda x: x["mean_pearson"] if not np.isnan(x["mean_pearson"]) else -1, 
        reverse=True
    )
    print(f"Step 4: Sorted by Pearson correlation: {len(results_filtered)} poems")
    
    # Also sort all results for full output
    results_sorted = sorted(
        results, 
        key=lambda x: x["mean_pearson"] if not np.isnan(x["mean_pearson"]) else -1, 
        reverse=True
    )
    
    # Print top results (filtered)
    print(f"\n{'='*80}")
    print(f"TOP {TOP_N} POEMS BY MEAN PEARSON CORRELATION (Inner Couplets)")
    print(f"(Filtered: correct pred → ≤{MAX_PUNCT_HEADS} punct heads → balance ≥{MIN_COUPLET_BALANCE:.0%} → sorted by r)")
    print(f"{'='*80}")
    
    for rank, result in enumerate(results_filtered[:TOP_N], 1):
        couplet2 = result["couplets"][1]
        couplet3 = result["couplets"][2]
        line_match = result["line_match"]
        pattern = "".join(str(m) for m in line_match)
        
        print(f"\n{rank}. ID: {result['id']} | Pearson r: {result['mean_pearson']:.4f} | Spearman ρ: {result['mean_spearman']:.4f}")
        print(f"   Pattern: {pattern} | Correct: {result['correct']} | "
              f"Punct: {result['num_punct_heads']} | Balance: C2={result['couplet2_ratio']:.1%} C3={result['couplet3_ratio']:.1%}")
        print(f"   Couplet 2: {couplet2[0]}，{couplet2[1]}  (r={result['mean_pearson_c2']:.3f}, ρ={result['mean_spearman_c2']:.3f})")
        print(f"   Couplet 3: {couplet3[0]}，{couplet3[1]}  (r={result['mean_pearson_c3']:.3f}, ρ={result['mean_spearman_c3']:.3f})")
    
    # Save results to JSON (full and filtered)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save full results
    output_file = os.path.join(OUTPUT_DIR, "correlation_results.json")
    json_results = [result_to_json(r) for r in results_sorted]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved full results to: {output_file}")
    
    # Save filtered results
    output_file_filtered = os.path.join(OUTPUT_DIR, "correlation_results_filtered.json")
    json_filtered = [result_to_json(r) for r in results_filtered]
    with open(output_file_filtered, "w", encoding="utf-8") as f:
        json.dump(json_filtered, f, ensure_ascii=False, indent=2)
    print(f"Saved filtered results to: {output_file_filtered}")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS (ALL REGULATED POEMS)")
    print(f"{'='*80}")
    all_spearman = [r["mean_spearman"] for r in results if not np.isnan(r["mean_spearman"])]
    all_pearson = [r["mean_pearson"] for r in results if not np.isnan(r["mean_pearson"])]
    print(f"Mean Spearman ρ: {np.mean(all_spearman):.4f} (std: {np.std(all_spearman):.4f})")
    print(f"Mean Pearson r:  {np.mean(all_pearson):.4f} (std: {np.std(all_pearson):.4f})")
    
    # Count poems with high correlation (> 0.5)
    high_sp = sum(1 for c in all_spearman if c > 0.5)
    high_pe = sum(1 for c in all_pearson if c > 0.5)
    print(f"Poems with mean ρ > 0.5: {high_sp} ({100*high_sp/len(all_spearman):.1f}%)")
    print(f"Poems with mean r > 0.5: {high_pe} ({100*high_pe/len(all_pearson):.1f}%)")
    
    # Prediction accuracy
    correct_count = sum(1 for r in results if r["correct"])
    print(f"\nPrediction accuracy: {correct_count}/{len(results)} ({100*correct_count/len(results):.1f}%)")
    
    # Punctuation focus distribution
    punct_dist = {}
    for r in results:
        n = r["num_punct_heads"]
        punct_dist[n] = punct_dist.get(n, 0) + 1
    print(f"\nPunctuation-focused heads distribution:")
    for n in sorted(punct_dist.keys()):
        print(f"  {n} heads: {punct_dist[n]} poems ({100*punct_dist[n]/len(results):.1f}%)")
    
    # Balance distribution
    all_balances = [r["min_couplet_balance"] for r in results]
    print(f"\nCouplet balance distribution (min of C2, C3 ratios):")
    print(f"  Mean: {np.mean(all_balances):.1%}, Std: {np.std(all_balances):.1%}")
    print(f"  Min: {np.min(all_balances):.1%}, Max: {np.max(all_balances):.1%}")
    balanced = sum(1 for b in all_balances if b >= MIN_COUPLET_BALANCE)
    print(f"  Poems with balance ≥{MIN_COUPLET_BALANCE:.0%}: {balanced} ({100*balanced/len(results):.1f}%)")
    
    # Summary for filtered poems
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS (FILTERED: 4-step pipeline)")
    print(f"{'='*80}")
    print(f"Pipeline: Correct → ≤{MAX_PUNCT_HEADS} punct heads → balance ≥{MIN_COUPLET_BALANCE:.0%} → sorted by r")
    filtered_sp = [r["mean_spearman"] for r in results_filtered if not np.isnan(r["mean_spearman"])]
    filtered_pe = [r["mean_pearson"] for r in results_filtered if not np.isnan(r["mean_pearson"])]
    if filtered_pe:
        print(f"Mean Spearman ρ: {np.mean(filtered_sp):.4f} (std: {np.std(filtered_sp):.4f})")
        print(f"Mean Pearson r:  {np.mean(filtered_pe):.4f} (std: {np.std(filtered_pe):.4f})")
        high_sp_filtered = sum(1 for c in filtered_sp if c > 0.5)
        high_pe_filtered = sum(1 for c in filtered_pe if c > 0.5)
        print(f"Poems with mean ρ > 0.5: {high_sp_filtered} ({100*high_sp_filtered/len(filtered_sp):.1f}%)")
        print(f"Poems with mean r > 0.5: {high_pe_filtered} ({100*high_pe_filtered/len(filtered_pe):.1f}%)")
    else:
        print("No poems passed all filters!")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
