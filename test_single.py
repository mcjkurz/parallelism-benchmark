import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from models import PoemParallelismClassifier

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

def test_char_model_single(l1, l2, char_model, tokenizer):
    pairs = list(zip(l1, l2))
    if not pairs:
        return {"consensus": 0, "details": [], "ratio": 0.0}

    encoded = tokenizer(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        truncation=True,
        padding=True,
        max_length=16,
        return_tensors="pt"
    ).to(device)

    char_model.eval()
    with torch.no_grad():
        logits = char_model(**encoded).logits
        preds = logits.argmax(dim=-1).cpu().tolist()

    ratio = sum(preds) / len(preds)
    consensus = 1 if ratio >= 0.6 else 0

    return {"consensus": consensus, "details": preds, "ratio": ratio}

def test_couplet_model_single(l1, l2, coup_model, tokenizer):
    text = l1 + "，" + l2
    encoded = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors="pt"
    ).to(device)

    coup_model.eval()
    with torch.no_grad():
        logits = coup_model(**encoded).logits
        pred = logits.argmax(dim=-1).item()

    return {"pred": int(pred), "text": text}

def test_poem4_model_single(couplets, poem4_model, tokenizer):
    if len(couplets) != 4:
        raise ValueError(f"Expected 4 couplets, got {len(couplets)}")

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

    poem4_model.eval()
    with torch.no_grad():
        logits = poem4_model(**encoded)["logits"]
        preds = logits.argmax(dim=-1).cpu().tolist()

    return {"preds": preds, "tokens": tokens}

def test_poem1_model_single(couplets, poem1_model, tokenizer):
    if len(couplets) != 4:
        raise ValueError(f"Expected 4 couplets, got {len(couplets)}")

    text = "".join([l1 + "，" + l2 + "。" for l1, l2 in couplets])

    encoded = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)

    poem1_model.eval()
    with torch.no_grad():
        logits = poem1_model(**encoded).logits
        pred = logits.argmax(dim=-1).item()

    return {"pred": int(pred), "text": text}

def main():
    print("Loading models...")
    tokenizer = BertTokenizerFast.from_pretrained("saved_artifacts/tokenizer")
    char_model = BertForSequenceClassification.from_pretrained("saved_artifacts/char_model").to(device)
    coup_model = BertForSequenceClassification.from_pretrained("saved_artifacts/coup_model").to(device)
    poem4_model = PoemParallelismClassifier.from_pretrained("saved_artifacts/poem4_model").to(device)
    poem1_model = BertForSequenceClassification.from_pretrained("saved_artifacts/poem1_model").to(device)

    l1 = "重重山树暗"
    l2 = "历历水花幽"

    poem_couplets = [
        ("春眠不觉晓", "处处闻啼鸟"),
        ("夜来风雨声", "花落知多少"),
        ("江碧鸟逾白", "山青花欲燃"),
        ("今春看又过", "何日是归年"),
    ]

    print("\n1. Char-level test:")
    char_result = test_char_model_single(l1, l2, char_model, tokenizer)
    print(char_result)

    print("\n2. Couplet-level test:")
    coup_result = test_couplet_model_single(l1, l2, coup_model, tokenizer)
    print(coup_result)

    print("\n3. Poem4-level test:")
    poem4_result = test_poem4_model_single(poem_couplets, poem4_model, tokenizer)
    print(poem4_result)

    print("\n4. Poem1-level test:")
    poem1_result = test_poem1_model_single(poem_couplets, poem1_model, tokenizer)
    print(poem1_result)

if __name__ == "__main__":
    main()

