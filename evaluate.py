import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
from tqdm.auto import tqdm
import pickle

from datasets import CharPairDataset, CoupletDataset, PoemDataset4Labels, PoemDataset1Label
from models import PoemParallelismClassifier

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

def evaluate_standard(model, dataset, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
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
        for item in tqdm(raw_couplet_data, desc="Char->Couplet Eval"):
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
        for item in tqdm(raw_poem_data, desc="Couplet->Poem Eval"):
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
        for item in tqdm(raw_poem_data, desc="Poem1 Inner-Couplet Eval"):
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

def main():
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

if __name__ == "__main__":
    main()

