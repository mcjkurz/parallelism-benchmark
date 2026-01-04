import json
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from tqdm.auto import tqdm
import pickle
import pandas as pd

from models import PoemParallelismClassifier

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

def predict_char_level(l1, l2, model, tokenizer):
    pairs = list(zip(l1, l2))
    if not pairs:
        return 0, []
    encoded = tokenizer(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        truncation=True,
        padding=True,
        max_length=16,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        preds = model(**encoded).logits.argmax(dim=-1).cpu().tolist()
    ratio = sum(preds) / len(preds)
    return (1 if ratio >= 0.6 else 0), preds

def predict_couplet_level(l1, l2, model, tokenizer):
    text = l1 + "，" + l2
    encoded = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        return model(**encoded).logits.argmax(dim=-1).item()

def predict_poem4_level(couplets, model, tokenizer):
    tokens = ["[CLS]"]
    for i, (l1, l2) in enumerate(couplets):
        tokens += [f"[CP{i+1}]"] + list(l1) + ["，"] + list(l2) + ["。"]
    tokens += ["[SEP]"]
    encoded = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=256,
        add_special_tokens=False
    ).to(device)
    with torch.no_grad():
        return model(**encoded)["logits"].argmax(dim=-1).cpu().tolist()[0]

def predict_poem1_level(couplets, model, tokenizer):
    text = "".join([l1 + "，" + l2 + "。" for l1, l2 in couplets])
    encoded = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        return model(**encoded).logits.argmax(dim=-1).item()

def generate_comparison_data(data, poem4_model, poem1_model, coup_model, char_model, tokenizer):
    print("Generating comparison data...")
    results = []

    poem4_model.eval()
    poem1_model.eval()
    coup_model.eval()
    char_model.eval()

    for idx, item in enumerate(tqdm(data)):
        couplets = item["couplets"]
        labels = item["labels"]

        poem4_preds = predict_poem4_level(couplets, poem4_model, tokenizer)
        poem1_pred = predict_poem1_level(couplets, poem1_model, tokenizer)

        full_text_lines = []
        for c in couplets:
            full_text_lines.extend([c[0], c[1]])

        for i in range(4):
            l1, l2 = couplets[i]
            truth = labels[i]

            coup_pred = predict_couplet_level(l1, l2, coup_model, tokenizer)
            char_cons, char_dets = predict_char_level(l1, l2, char_model, tokenizer)

            poem1_implicit = -1
            if i in [1, 2]:
                poem1_implicit = 1 if poem1_pred == 1 else 0

            results.append({
                "poem_id": idx,
                "dynasty": item["dynasty"],
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

    return pd.DataFrame(results)

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


def pairwise_comparison(df, model_a, model_b, col_a, col_b, inner_only=False):
    """Compare two models and return counts + examples."""
    if inner_only:
        df = df[df['couplet_idx'].isin([1, 2])]
    
    total = len(df)
    
    a_correct = df[col_a] == df['truth']
    b_correct = df[col_b] == df['truth']
    
    both_correct = df[a_correct & b_correct]
    both_wrong = df[~a_correct & ~b_correct]
    a_only = df[a_correct & ~b_correct]
    b_only = df[~a_correct & b_correct]
    
    return {
        "total": total,
        "both_correct": len(both_correct),
        "both_wrong": len(both_wrong),
        f"{model_a}_only_correct": len(a_only),
        f"{model_b}_only_correct": len(b_only),
        "examples": {
            f"{model_a}_only_correct": [row_to_example(row) for _, row in a_only.iterrows()],
            f"{model_b}_only_correct": [row_to_example(row) for _, row in b_only.iterrows()]
        }
    }


def analyze_models(df):
    """Analyze all models with pairwise comparisons."""
    total = len(df)
    inner_only = df[df['couplet_idx'].isin([1, 2])]
    
    # Accuracy for each model
    accuracy = {
        "char": float((df['pred_char'] == df['truth']).mean()),
        "couplet": float((df['pred_coup'] == df['truth']).mean()),
        "poem4": float((df['pred_poem4'] == df['truth']).mean()),
        "poem1": float((inner_only['pred_poem1_implicit'] == inner_only['truth']).mean())
    }
    
    # All 6 pairwise comparisons
    pairwise = {
        "char_vs_couplet": pairwise_comparison(
            df, "char", "couplet", "pred_char", "pred_coup"
        ),
        "char_vs_poem4": pairwise_comparison(
            df, "char", "poem4", "pred_char", "pred_poem4"
        ),
        "char_vs_poem1": pairwise_comparison(
            df, "char", "poem1", "pred_char", "pred_poem1_implicit", inner_only=True
        ),
        "couplet_vs_poem4": pairwise_comparison(
            df, "couplet", "poem4", "pred_coup", "pred_poem4"
        ),
        "couplet_vs_poem1": pairwise_comparison(
            df, "couplet", "poem1", "pred_coup", "pred_poem1_implicit", inner_only=True
        ),
        "poem4_vs_poem1": pairwise_comparison(
            df, "poem4", "poem1", "pred_poem4", "pred_poem1_implicit", inner_only=True
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

def main():
    print("Loading models and data...")
    tokenizer = BertTokenizerFast.from_pretrained("saved_artifacts/tokenizer")
    
    char_model = BertForSequenceClassification.from_pretrained("saved_artifacts/char_model").to(device)
    coup_model = BertForSequenceClassification.from_pretrained("saved_artifacts/coup_model").to(device)
    poem4_model = PoemParallelismClassifier.from_pretrained("saved_artifacts/poem4_model").to(device)
    poem1_model = BertForSequenceClassification.from_pretrained("saved_artifacts/poem1_model").to(device)

    with open("saved_artifacts/poem4_test_raw.pkl", "rb") as f:
        poem4_test_raw = pickle.load(f)

    df_results = generate_comparison_data(
        poem4_test_raw,
        poem4_model,
        poem1_model,
        coup_model,
        char_model,
        tokenizer
    )

    results = analyze_models(df_results)
    
    # Print summary
    print()
    print("=" * 60)
    print("MODEL COMPARISON ANALYSIS")
    print("=" * 60)
    
    summary = results['summary']
    print(f"\nTotal couplets: {summary['total_couplets']}")
    print(f"Inner couplets (for poem1): {summary['inner_couplets']}")
    
    print("\nAccuracy:")
    for model, acc in summary['accuracy'].items():
        print(f"  {model}: {acc:.4f}")
    
    print("\nPairwise Comparisons:")
    for pair_name, data in results['pairwise'].items():
        print(f"\n  {pair_name} (n={data['total']}):")
        print(f"    Both correct: {data['both_correct']}")
        print(f"    Both wrong: {data['both_wrong']}")
        # Get the two "only correct" keys
        only_keys = [k for k in data.keys() if k.endswith('_only_correct') and k != 'examples']
        for key in only_keys:
            print(f"    {key}: {data[key]}")
    
    # Create summary version (without examples)
    summary_results = {
        "summary": results['summary'],
        "pairwise": {}
    }
    for pair_name, data in results['pairwise'].items():
        summary_results['pairwise'][pair_name] = {
            k: v for k, v in data.items() if k != 'examples'
        }
    
    # Save summary JSON
    with open("model_comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved to model_comparison_summary.json")
    
    # Save full results with all examples
    with open("model_comparison_full.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Full results (with all examples) saved to model_comparison_full.json")


if __name__ == "__main__":
    main()

