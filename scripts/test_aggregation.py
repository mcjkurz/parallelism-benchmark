"""
Test different aggregation methods for char → couplet induction.

Compares:
1. Hard voting: ≥k/5 character matches → parallel
2. Mean probability with threshold
3. Poisson binomial: P(≥k successes) with threshold

Uses silver_standard_test.json char_match as oracle predictions,
or optionally loads the trained char model for real predictions.
"""

import json
import os
import sys
from itertools import combinations
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)


def load_test_data():
    """Load test couplets with char_match and line_match labels."""
    path = os.path.join(PROJECT_ROOT, "data", "silver_standard_test.json")
    with open(path, "r", encoding="utf-8") as f:
        poems = json.load(f)
    
    # Flatten to couplet level
    data = []
    for poem in poems:
        for i, couplet in enumerate(poem["couplets"]):
            data.append({
                "couplet": couplet,
                "char_match": poem["char_match"][i],  # 5 binary values
                "line_match": poem["line_match"][i],  # ground truth
            })
    return data


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
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# =============================================================================
# AGGREGATION METHODS
# =============================================================================

def hard_voting(char_matches, k=3):
    """Original method: ≥k/5 matches → parallel."""
    return 1 if sum(char_matches) >= k else 0


def mean_threshold(char_probs, threshold=0.5):
    """Mean of probabilities with threshold."""
    return 1 if np.mean(char_probs) >= threshold else 0


def poisson_binomial(probs, k=3):
    """P(at least k successes) for independent Bernoullis with different p_i."""
    n = len(probs)
    total = 0.0
    
    for j in range(k, n + 1):
        for subset in combinations(range(n), j):
            p = 1.0
            for i in range(n):
                if i in subset:
                    p *= probs[i]
                else:
                    p *= (1 - probs[i])
            total += p
    return total


def poisson_binomial_threshold(char_probs, k=3, threshold=0.5):
    """Poisson binomial P(≥k) with threshold."""
    p = poisson_binomial(char_probs, k)
    return 1 if p >= threshold else 0


def weighted_mean(char_probs, weights=None, threshold=0.5):
    """Weighted mean with threshold. Default: position 3 (middle) weighted higher."""
    if weights is None:
        weights = [1.0, 1.0, 1.5, 1.0, 1.0]  # Middle position slightly more important
    weighted_sum = sum(p * w for p, w in zip(char_probs, weights))
    weighted_avg = weighted_sum / sum(weights)
    return 1 if weighted_avg >= threshold else 0


# =============================================================================
# TEST RUNNER
# =============================================================================

def test_with_oracle(data):
    """Test aggregation methods using ground truth char_match as oracle."""
    print("\n" + "=" * 70)
    print("ORACLE TEST: Using ground truth char_match values")
    print("=" * 70)
    
    labels = [d["line_match"] for d in data]
    
    # For oracle, char_match values are binary (0 or 1)
    # We'll also test with simulated probabilities
    
    results = {}
    
    # 1. Hard voting with different k
    for k in [2, 3, 4]:
        preds = [hard_voting(d["char_match"], k=k) for d in data]
        metrics = compute_metrics(preds, labels)
        results[f"hard_≥{k}/5"] = metrics
    
    # 2. Mean threshold (on binary values, same as sum/5 ≥ threshold)
    for thresh in [0.4, 0.5, 0.6]:
        preds = [mean_threshold(d["char_match"], threshold=thresh) for d in data]
        metrics = compute_metrics(preds, labels)
        results[f"mean≥{thresh}"] = metrics
    
    # 3. Poisson binomial (on binary values = same as hard voting, but let's verify)
    preds = [poisson_binomial_threshold(d["char_match"], k=3, threshold=0.5) for d in data]
    metrics = compute_metrics(preds, labels)
    results["poisson_k3_t0.5"] = metrics
    
    print_results(results)
    return results


def test_with_noisy_oracle(data, noise_level=0.1):
    """Test with simulated probabilities: add noise to binary char_match."""
    print(f"\n" + "=" * 70)
    print(f"NOISY ORACLE TEST: char_match + noise (level={noise_level})")
    print("=" * 70)
    
    labels = [d["line_match"] for d in data]
    
    # Convert binary to noisy probabilities
    np.random.seed(42)
    noisy_data = []
    for d in data:
        # Binary 1 → ~0.9, Binary 0 → ~0.1, with noise
        probs = []
        for match in d["char_match"]:
            if match == 1:
                p = np.clip(0.9 + np.random.uniform(-noise_level, noise_level), 0.5, 1.0)
            else:
                p = np.clip(0.1 + np.random.uniform(-noise_level, noise_level), 0.0, 0.5)
            probs.append(p)
        noisy_data.append({"probs": probs, "line_match": d["line_match"]})
    
    results = {}
    
    # 1. Hard voting (threshold probs at 0.5 first)
    for k in [2, 3, 4]:
        preds = [hard_voting([1 if p >= 0.5 else 0 for p in d["probs"]], k=k) for d in noisy_data]
        metrics = compute_metrics(preds, labels)
        results[f"hard_≥{k}/5"] = metrics
    
    # 2. Mean threshold
    for thresh in [0.4, 0.5, 0.6]:
        preds = [mean_threshold(d["probs"], threshold=thresh) for d in noisy_data]
        metrics = compute_metrics(preds, labels)
        results[f"mean≥{thresh}"] = metrics
    
    # 3. Poisson binomial
    for k in [2, 3]:
        for thresh in [0.4, 0.5, 0.6]:
            preds = [poisson_binomial_threshold(d["probs"], k=k, threshold=thresh) for d in noisy_data]
            metrics = compute_metrics(preds, labels)
            results[f"poisson_k{k}_t{thresh}"] = metrics
    
    # 4. Weighted mean
    preds = [weighted_mean(d["probs"], threshold=0.5) for d in noisy_data]
    metrics = compute_metrics(preds, labels)
    results["weighted_mean"] = metrics
    
    print_results(results)
    return results


def test_with_model(data):
    """Test with actual trained char model predictions."""
    print("\n" + "=" * 70)
    print("MODEL TEST: Using trained char model predictions")
    print("=" * 70)
    
    import torch
    from transformers import BertForSequenceClassification
    from train_utils import get_device, create_tokenizer
    from inference import MAX_LEN_CHAR
    
    device = get_device()
    print(f"Device: {device}")
    
    # Load model
    model_path = os.path.join(PROJECT_ROOT, "results", "models", "char_model")
    if not os.path.exists(model_path):
        print(f"ERROR: Char model not found at {model_path}")
        print("Run the training pipeline first: ./scripts/pipeline.sh")
        return None
    
    tokenizer = create_tokenizer()
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    print("Getting model predictions...")
    labels = [d["line_match"] for d in data]
    all_probs = []
    all_preds = []
    
    for d in data:
        couplet = d["couplet"]
        pairs = [(couplet[0][i], couplet[1][i]) for i in range(5)]
        
        encoded = tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN_CHAR,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
        
        all_probs.append(probs)
        all_preds.append(preds)
    
    results = {}
    
    # 1. Hard voting
    for k in [2, 3, 4]:
        preds = [hard_voting(p, k=k) for p in all_preds]
        metrics = compute_metrics(preds, labels)
        results[f"hard_≥{k}/5"] = metrics
    
    # 2. Mean threshold
    for thresh in [0.4, 0.5, 0.6]:
        preds = [mean_threshold(p, threshold=thresh) for p in all_probs]
        metrics = compute_metrics(preds, labels)
        results[f"mean≥{thresh}"] = metrics
    
    # 3. Poisson binomial
    for k in [2, 3]:
        for thresh in [0.4, 0.5, 0.6]:
            preds = [poisson_binomial_threshold(p, k=k, threshold=thresh) for p in all_probs]
            metrics = compute_metrics(preds, labels)
            results[f"poisson_k{k}_t{thresh}"] = metrics
    
    # 4. Weighted mean
    preds = [weighted_mean(p, threshold=0.5) for p in all_probs]
    metrics = compute_metrics(preds, labels)
    results["weighted_mean"] = metrics
    
    print_results(results)
    return results


def print_results(results):
    """Print results in a table."""
    print(f"\n{'Method':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 70)
    
    # Sort by F1
    sorted_results = sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True)
    
    for name, m in sorted_results:
        print(f"{name:<25} {m['accuracy']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
    
    print("-" * 70)
    best = sorted_results[0]
    print(f"Best: {best[0]} (F1={best[1]['f1']:.4f})")


def test_chunked_hypothesis(data):
    """Test if 'one match per chunk' matters vs 'any 3 matches'."""
    print("\n" + "=" * 70)
    print("CHUNK HYPOTHESIS TEST: Does position distribution matter?")
    print("=" * 70)
    
    # Assume 2+1+2 structure: chunks = [0,1], [2], [3,4]
    def chunks_covered_212(char_match):
        """Count how many of the 3 chunks (2+1+2) have at least one match."""
        chunk1 = char_match[0] or char_match[1]  # positions 0-1
        chunk2 = char_match[2]                    # position 2 (pivot)
        chunk3 = char_match[3] or char_match[4]  # positions 3-4
        return chunk1 + chunk2 + chunk3
    
    # Assume 2+2+1 structure: chunks = [0,1], [2,3], [4]
    def chunks_covered_221(char_match):
        chunk1 = char_match[0] or char_match[1]
        chunk2 = char_match[2] or char_match[3]
        chunk3 = char_match[4]
        return chunk1 + chunk2 + chunk3
    
    labels = [d["line_match"] for d in data]
    
    results = {}
    
    # Test: all 3 chunks covered (2+1+2)
    preds = [1 if chunks_covered_212(d["char_match"]) == 3 else 0 for d in data]
    results["3_chunks_212"] = compute_metrics(preds, labels)
    
    # Test: ≥2 chunks covered (2+1+2)
    preds = [1 if chunks_covered_212(d["char_match"]) >= 2 else 0 for d in data]
    results["≥2_chunks_212"] = compute_metrics(preds, labels)
    
    # Test: all 3 chunks covered (2+2+1)
    preds = [1 if chunks_covered_221(d["char_match"]) == 3 else 0 for d in data]
    results["3_chunks_221"] = compute_metrics(preds, labels)
    
    # Test: ≥2 chunks covered (2+2+1)
    preds = [1 if chunks_covered_221(d["char_match"]) >= 2 else 0 for d in data]
    results["≥2_chunks_221"] = compute_metrics(preds, labels)
    
    # Baseline: your ≥3/5 rule
    preds = [1 if sum(d["char_match"]) >= 3 else 0 for d in data]
    results["≥3/5_any"] = compute_metrics(preds, labels)
    
    # Analyze distribution of match patterns
    print("\nMatch pattern distribution (oracle char_match):")
    patterns = {}
    for d in data:
        pattern = tuple(d["char_match"])
        n_matches = sum(pattern)
        key = (n_matches, pattern)
        if key not in patterns:
            patterns[key] = {"total": 0, "parallel": 0}
        patterns[key]["total"] += 1
        patterns[key]["parallel"] += d["line_match"]
    
    # Show patterns with exactly 3 matches
    print("\nPatterns with exactly 3 matches:")
    print(f"{'Pattern':<20} {'Count':>8} {'Parallel%':>10} {'Chunks_212':>12}")
    three_match_patterns = [(k, v) for k, v in patterns.items() if k[0] == 3]
    three_match_patterns.sort(key=lambda x: x[1]["total"], reverse=True)
    
    for (n, pattern), stats in three_match_patterns[:10]:
        pct = 100 * stats["parallel"] / stats["total"]
        chunks = chunks_covered_212(list(pattern))
        print(f"{str(pattern):<20} {stats['total']:>8} {pct:>9.1f}% {chunks:>12}")
    
    print_results(results)


def main():
    print("Loading test data...")
    data = load_test_data()
    print(f"Loaded {len(data)} couplets from {len(data)//4} poems")
    
    # Count class balance
    n_parallel = sum(1 for d in data if d["line_match"] == 1)
    print(f"Class balance: {n_parallel} parallel ({100*n_parallel/len(data):.1f}%), "
          f"{len(data)-n_parallel} non-parallel ({100*(len(data)-n_parallel)/len(data):.1f}%)")
    
    # Test chunk hypothesis first
    test_chunked_hypothesis(data)
    
    # Test with trained char model
    test_with_model(data)
    
    # Also show oracle results for comparison
    print("\n" + "=" * 70)
    print("COMPARISON: Oracle (ground truth char_match)")
    print("=" * 70)
    test_with_oracle(data)


if __name__ == "__main__":
    main()

