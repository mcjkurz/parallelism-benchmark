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
                "pred_char": char_cons,
                "pred_char_details": char_dets,
                "pred_coup": coup_pred,
                "pred_poem4": poem4_preds[i],
                "pred_poem4_full": poem4_preds,
                "pred_poem1_global": poem1_pred,
                "pred_poem1_implicit": poem1_implicit
            })

    return pd.DataFrame(results)

def format_example(row, count):
    lines = row['full_text']
    c_idx = row['couplet_idx']
    dynasty = row['dynasty']

    parts = []
    parts.append(f"\n--- Example {count} ---")
    parts.append(f"\nDynasty: {dynasty}")
    parts.append("Poem Context:")
    for k in range(0, 8, 2):
        marker = "→ " if k//2 == c_idx else "  "
        parts.append(f"{marker}{lines[k]}，{lines[k+1]}")

    parts.append(f"\nTarget: {row['l1']}，{row['l2']}")
    parts.append(f"Ground Truth: {row['truth']}")

    c_status = "✅" if row['pred_char'] == row['truth'] else "❌"
    parts.append(f"Char Model:    {row['pred_char']} {c_status} | Details: {row['pred_char_details']}")

    cp_status = "✅" if row['pred_coup'] == row['truth'] else "❌"
    parts.append(f"Couplet Model: {row['pred_coup']} {cp_status} | (Correct Level)")

    p4_status = "✅" if row['pred_poem4'] == row['truth'] else "❌"
    parts.append(f"Poem4 Model:   {row['pred_poem4']} {p4_status} | Details: {row['pred_poem4_full']}")

    if row['couplet_idx'] in [1, 2]:
        p1_global = row['pred_poem1_global']
        p1_desc = "Regulated" if p1_global == 1 else "Not Regulated"
        implicit = row['pred_poem1_implicit']
        p1_status = "✅" if implicit == row['truth'] else "❌"
        parts.append(f"Poem1 Model:   {implicit} {p1_status} | Implies this couplet is {p1_desc}")
    else:
        parts.append("Poem1 Model:   N/A (Outer couplets don't determine regulation)")

    return "\n".join(parts)

def run_scenarios(df):
    def process_subset(subset, name, filename):
        print("\n" + "="*80)
        print(f"{name}")
        print("="*80)

        if subset.empty:
            print("No examples found.")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"{name}\n\nNo examples found.\n")
            return

        count_printed = 0
        for _, row in subset.iterrows():
            count_printed += 1
            print(format_example(row, count_printed))
            if count_printed >= 10:
                break

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"{name}\n")
            f.write("="*80 + "\n")
            for idx, row in subset.iterrows():
                ex_index = idx + 1
                text = format_example(row, ex_index)
                f.write(text)
                f.write("\n")

    df_char_fail = df[
        (df['pred_char'] != df['truth']) &
        (df['pred_coup'] == df['truth'])
    ]
    process_subset(
        df_char_fail,
        "SCENARIO A: Char Model Wrong, Couplet Model Right (Low-level Noise)",
        "scenario_A.txt"
    )

    df_poem4_fail = df[
        (df['pred_poem4'] != df['truth']) &
        (df['pred_coup'] == df['truth'])
    ]
    process_subset(
        df_poem4_fail,
        "SCENARIO B: Poem4 Model Wrong, Couplet Model Right (Contextual Hallucination)",
        "scenario_B.txt"
    )

    df_poem1_fail = df[
        (df['couplet_idx'].isin([1, 2])) &
        (df['truth'] == 0) &
        (df['pred_coup'] == 0) &
        (df['pred_poem1_global'] == 1)
    ]
    process_subset(
        df_poem1_fail,
        "SCENARIO C: Poem1 (Global) Hallucination",
        "scenario_C.txt"
    )

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

    run_scenarios(df_results)

if __name__ == "__main__":
    main()

