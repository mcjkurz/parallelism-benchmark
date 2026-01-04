"""
Shared inference functions for parallelism prediction.

These functions are used by run_trials.py, analyze_scenarios.py, and test_single.py.
"""

import torch


def predict_char_level(l1, l2, model, tokenizer, device):
    """Predict parallelism at character level.
    
    Args:
        l1: First line (string of characters)
        l2: Second line (string of characters)
        model: Character-level classifier
        tokenizer: Tokenizer
        device: torch device
        
    Returns:
        Tuple of (consensus_label, char_predictions_list)
    """
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


def predict_couplet_level(l1, l2, model, tokenizer, device):
    """Predict parallelism at couplet level.
    
    Args:
        l1: First line
        l2: Second line
        model: Couplet-level classifier
        tokenizer: Tokenizer
        device: torch device
        
    Returns:
        Binary prediction (0 or 1)
    """
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


def predict_poem4_level(couplets, model, tokenizer, device):
    """Predict parallelism at poem level with 4 labels.
    
    Args:
        couplets: List of 4 (l1, l2) tuples
        model: Poem 4-label classifier
        tokenizer: Tokenizer
        device: torch device
        
    Returns:
        List of 4 binary predictions
    """
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


def predict_poem1_level(couplets, model, tokenizer, device):
    """Predict parallelism at poem level with 1 label.
    
    Args:
        couplets: List of 4 (l1, l2) tuples
        model: Poem 1-label classifier
        tokenizer: Tokenizer
        device: torch device
        
    Returns:
        Binary prediction (0 or 1)
    """
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

