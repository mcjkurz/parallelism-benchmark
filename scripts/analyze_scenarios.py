"""
Analyze model predictions across different scenarios.

This script can be run standalone to analyze the saved best models,
or can work with comparison files generated during run_trials.py.

Usage:
    python scripts/analyze_scenarios.py                     # Analyze best models
    python scripts/analyze_scenarios.py --seed 42           # Analyze specific comparison file
    python scripts/analyze_scenarios.py --aggregate         # Aggregate all comparison files
"""

import argparse
import json
import os
import sys
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from tqdm.auto import tqdm
import pickle
import numpy as np

# Add parent directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from models import PoemParallelismClassifier
from train_utils import get_device
from inference import predict_char_level, predict_couplet_level, predict_poem4_level, predict_poem1_level

device = get_device()
print(f"Using device: {device}")


def generate_comparison_data(data, poem4_model, poem1_model, coup_model, char_model, tokenizer):
    """Generate comparison data from saved models."""
    print("Generating comparison data...")
    results = []

    poem4_model.eval()
    poem1_model.eval()
    coup_model.eval()
    char_model.eval()

    for idx, item in enumerate(tqdm(data)):
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

    return results


def row_to_example(row):
    """Convert a DataFrame row to a simple example dict."""
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


def analyze_models(results):
    """Analyze all models with pairwise comparisons."""
    total = len(results)
    inner_only = [r for r in results if r['couplet_idx'] in [1, 2]]
    
    # Accuracy for each model
    accuracy = {
        "char": float(np.mean([r['pred_char'] == r['truth'] for r in results])),
        "couplet": float(np.mean([r['pred_coup'] == r['truth'] for r in results])),
        "poem4": float(np.mean([r['pred_poem4'] == r['truth'] for r in results])),
        "poem1": float(np.mean([r['pred_poem1_implicit'] == r['truth'] for r in inner_only]))
    }
    
    # All 6 pairwise comparisons
    pairwise = {
        "char_vs_couplet": pairwise_comparison(
            results, "char", "couplet", "pred_char", "pred_coup"
        ),
        "char_vs_poem4": pairwise_comparison(
            results, "char", "poem4", "pred_char", "pred_poem4"
        ),
        "char_vs_poem1": pairwise_comparison(
            results, "char", "poem1", "pred_char", "pred_poem1_implicit", inner_only=True
        ),
        "couplet_vs_poem4": pairwise_comparison(
            results, "couplet", "poem4", "pred_coup", "pred_poem4"
        ),
        "couplet_vs_poem1": pairwise_comparison(
            results, "couplet", "poem1", "pred_coup", "pred_poem1_implicit", inner_only=True
        ),
        "poem4_vs_poem1": pairwise_comparison(
            results, "poem4", "poem1", "pred_poem4", "pred_poem1_implicit", inner_only=True
        ),
    }
    
    return {
        "summary": {
            "total_couplets": total,
            "inner_couplets": len(inner_only),
            "accuracy": accuracy
        },
        "pairwise": pairwise
    }


def aggregate_comparisons(comparisons_dir):
    """Aggregate all comparison files into summary statistics."""
    comparison_files = [f for f in os.listdir(comparisons_dir) if f.startswith("comparison_seed_")]
    
    if not comparison_files:
        print("No comparison files found!")
        return None
    
    print(f"Found {len(comparison_files)} comparison files")
    
    all_accuracies = {"char": [], "couplet": [], "poem4": [], "poem1": []}
    all_pairwise = {}
    
    for fname in tqdm(comparison_files, desc="Loading comparisons"):
        fpath = os.path.join(comparisons_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for model, acc in data["summary"]["accuracy"].items():
            all_accuracies[model].append(acc)
        
        for pair_name, pair_data in data["pairwise"].items():
            if pair_name not in all_pairwise:
                all_pairwise[pair_name] = {
                    "total": [],
                    "both_correct": [],
                    "both_wrong": [],
                }
                # Get the model-specific keys
                for key in pair_data.keys():
                    if key.endswith("_only_correct") and key != "examples":
                        all_pairwise[pair_name][key] = []
            
            for key in all_pairwise[pair_name].keys():
                if key in pair_data:
                    all_pairwise[pair_name][key].append(pair_data[key])
    
    # Compute statistics
    accuracy_stats = {}
    for model, values in all_accuracies.items():
        arr = np.array(values)
        accuracy_stats[model] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    
    pairwise_stats = {}
    for pair_name, pair_data in all_pairwise.items():
        pairwise_stats[pair_name] = {}
        for key, values in pair_data.items():
            arr = np.array(values)
            pairwise_stats[pair_name][key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
            }
    
    return {
        "num_comparisons": len(comparison_files),
        "accuracy": accuracy_stats,
        "pairwise": pairwise_stats
    }


def check_artifacts_exist(artifacts_dir):
    """Check if saved model artifacts exist."""
    required = {
        "tokenizer": "tokenizer",
        "char_model": "character model",
        "coup_model": "couplet model", 
        "poem4_model": "poem-4 model",
        "poem1_model": "poem-1 model",
        "poem4_test_raw.pkl": "test data"
    }
    missing = []
    
    if not os.path.exists(artifacts_dir):
        print(f"Error: Artifacts directory not found: {artifacts_dir}")
        print()
        print("Please run the training pipeline first:")
        print("  python scripts/run_trials.py")
        print()
        print("Or run the full pipeline:")
        print("  ./scripts/pipeline.sh")
        return False
    
    for filename, description in required.items():
        path = os.path.join(artifacts_dir, filename)
        if not os.path.exists(path):
            missing.append(f"{filename} ({description})")
    
    if missing:
        print(f"Error: Missing artifacts in {artifacts_dir}:")
        for m in missing:
            print(f"  - {m}")
        print()
        print("Please run the training pipeline first:")
        print("  python scripts/run_trials.py")
        return False
    
    return True


def analyze_best_models():
    """Analyze the best saved models (original functionality)."""
    artifacts_dir = os.path.join(PROJECT_ROOT, "saved_artifacts")
    results_dir = os.path.join(PROJECT_ROOT, "results")
    
    # Check artifacts exist before loading
    if not check_artifacts_exist(artifacts_dir):
        return
    
    print("Loading models and data...")
    tokenizer = BertTokenizerFast.from_pretrained(os.path.join(artifacts_dir, "tokenizer"))
    
    char_model = BertForSequenceClassification.from_pretrained(
        os.path.join(artifacts_dir, "char_model")
    ).to(device)
    coup_model = BertForSequenceClassification.from_pretrained(
        os.path.join(artifacts_dir, "coup_model")
    ).to(device)
    poem4_model = PoemParallelismClassifier.from_pretrained(
        os.path.join(artifacts_dir, "poem4_model")
    ).to(device)
    poem1_model = BertForSequenceClassification.from_pretrained(
        os.path.join(artifacts_dir, "poem1_model")
    ).to(device)

    with open(os.path.join(artifacts_dir, "poem4_test_raw.pkl"), "rb") as f:
        poem4_test_raw = pickle.load(f)

    results = generate_comparison_data(
        poem4_test_raw,
        poem4_model,
        poem1_model,
        coup_model,
        char_model,
        tokenizer
    )

    analysis = analyze_models(results)
    
    # Print summary
    print()
    print("=" * 60)
    print("MODEL COMPARISON ANALYSIS")
    print("=" * 60)
    
    summary = analysis['summary']
    print(f"\nTotal couplets: {summary['total_couplets']}")
    print(f"Inner couplets (for poem1): {summary['inner_couplets']}")
    
    print("\nAccuracy:")
    for model, acc in summary['accuracy'].items():
        print(f"  {model}: {acc:.4f}")
    
    print("\nPairwise Comparisons:")
    for pair_name, data in analysis['pairwise'].items():
        print(f"\n  {pair_name} (n={data['total']}):")
        print(f"    Both correct: {data['both_correct']}")
        print(f"    Both wrong: {data['both_wrong']}")
        # Get the two "only correct" keys
        only_keys = [k for k in data.keys() if k.endswith('_only_correct') and k != 'examples']
        for key in only_keys:
            print(f"    {key}: {data[key]}")
    
    # Create summary version (without examples)
    summary_results = {
        "summary": analysis['summary'],
        "pairwise": {}
    }
    for pair_name, data in analysis['pairwise'].items():
        summary_results['pairwise'][pair_name] = {
            k: v for k, v in data.items() if k != 'examples'
        }
    
    # Save summary JSON
    summary_path = os.path.join(results_dir, "model_comparison_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved to {summary_path}")
    
    # Save full results with all examples
    full_path = os.path.join(results_dir, "model_comparison_full.json")
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f"Full results (with all examples) saved to {full_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze model predictions"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Analyze a specific seed's comparison file"
    )
    parser.add_argument(
        "--aggregate", action="store_true",
        help="Aggregate all comparison files into summary statistics"
    )
    args = parser.parse_args()
    
    results_dir = os.path.join(PROJECT_ROOT, "results")
    comparisons_dir = os.path.join(results_dir, "comparisons")
    
    if args.aggregate:
        if not os.path.exists(comparisons_dir):
            print(f"Comparisons directory not found: {comparisons_dir}")
            print("Run run_trials.py first to generate comparison files.")
            return
        
        stats = aggregate_comparisons(comparisons_dir)
        if stats:
            print()
            print("=" * 60)
            print("AGGREGATED STATISTICS")
            print("=" * 60)
            
            print(f"\nFrom {stats['num_comparisons']} comparison files:")
            
            print("\nAccuracy (mean ± std):")
            for model, stat in stats['accuracy'].items():
                print(f"  {model}: {stat['mean']:.4f} ± {stat['std']:.4f} [{stat['min']:.4f}, {stat['max']:.4f}]")
            
            output_path = os.path.join(results_dir, "comparison_aggregate.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            print(f"\nSaved to {output_path}")
    
    elif args.seed is not None:
        comparison_file = os.path.join(comparisons_dir, f"comparison_seed_{args.seed}.json")
        if not os.path.exists(comparison_file):
            print(f"Comparison file not found: {comparison_file}")
            return
        
        with open(comparison_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print()
        print("=" * 60)
        print(f"COMPARISON FOR SEED {args.seed}")
        print("=" * 60)
        
        summary = data['summary']
        print(f"\nTotal couplets: {summary['total_couplets']}")
        print(f"Inner couplets: {summary['inner_couplets']}")
        
        print("\nAccuracy:")
        for model, acc in summary['accuracy'].items():
            print(f"  {model}: {acc:.4f}")
    
    else:
        # Default: analyze best saved models
        analyze_best_models()


if __name__ == "__main__":
    main()
