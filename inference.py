"""
Shared inference functions for parallelism prediction.

These functions are used by run_trials.py, analyze_scenarios.py, and test_single.py.

Max lengths calculated precisely for 五言律诗 (pentasyllabic regulated verse):
- Character pairs: [CLS] c1 [SEP] c2 [SEP] = 5 tokens → MAX_LEN_CHAR = 8
- Couplet: [CLS] + 5 chars + "，" + 5 chars + [SEP] = 13 tokens → MAX_LEN_COUPLET = 16
- Poem4: [CLS] + 4×([CPn] + 5 + "，" + 5 + "。") + [SEP] = 54 tokens → MAX_LEN_POEM4 = 56
- Poem1: [CLS] + 4×(5 + "，" + 5 + "。") + [SEP] = 50 tokens → MAX_LEN_POEM1 = 52
"""

import torch

# Calculated max lengths for each model type
MAX_LEN_CHAR = 8      # [CLS] c1 [SEP] c2 [SEP] = 5 tokens
MAX_LEN_COUPLET = 16  # [CLS] + 5 + 1 + 5 + [SEP] = 13 tokens
MAX_LEN_POEM4 = 56    # [CLS] + 4×(1+5+1+5+1) + [SEP] = 54 tokens
MAX_LEN_POEM1 = 52    # [CLS] + 4×(5+1+5+1) + [SEP] = 50 tokens


def predict_char_pairs(model, tokenizer, couplet, device):
    """Predict parallelism for each character pair in a couplet.
    
    Args:
        model: Character-level classifier
        tokenizer: Tokenizer
        couplet: Tuple of (line1, line2), each 5 characters
        device: torch device
        
    Returns:
        List of 5 binary predictions (one per character position)
    """
    pairs = [(couplet[0][i], couplet[1][i]) for i in range(5)]
    
    # Use sentence-pair format: [CLS] c1 [SEP] c2 [SEP]
    encoded = tokenizer(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN_CHAR,
        return_tensors="pt"
    ).to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(**encoded).logits
        preds = logits.argmax(dim=-1).cpu().tolist()
    return preds


def predict_char_consensus(model, tokenizer, couplet, device, threshold=0.6):
    """Predict couplet parallelism by character consensus.
    
    Args:
        model: Character-level classifier
        tokenizer: Tokenizer
        couplet: Tuple of (line1, line2), each 5 characters
        device: torch device
        threshold: Fraction of chars needed for consensus (default: 0.6 = 3/5)
        
    Returns:
        Tuple of (consensus_label, char_predictions_list)
    """
    preds = predict_char_pairs(model, tokenizer, couplet, device)
    ratio = sum(preds) / len(preds)
    consensus = 1 if ratio >= threshold else 0
    return consensus, preds


def predict_couplet(model, tokenizer, couplet, device):
    """Predict if a couplet is parallel.
    
    Args:
        model: Couplet-level classifier
        tokenizer: Tokenizer
        couplet: Tuple of (line1, line2)
        device: torch device
        
    Returns:
        Binary prediction (0 or 1)
    """
    text = couplet[0] + "，" + couplet[1]
    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN_COUPLET,
        return_tensors="pt"
    ).to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(**encoded).logits
        pred = logits.argmax(dim=-1).item()
    return pred


def predict_poem4(model, tokenizer, couplets, device):
    """Predict parallelism for all 4 couplets using poem4 model.
    
    Args:
        model: Poem 4-label classifier (PoemParallelismClassifier)
        tokenizer: Tokenizer with [CP1]-[CP4] special tokens
        couplets: List of 4 (line1, line2) tuples
        device: torch device
        
    Returns:
        List of 4 binary predictions
    """
    # Encode poem with couplet markers
    lines = []
    for c in couplets:
        lines.append(c[0])
        lines.append(c[1])
    
    tokens = ["[CLS]"]
    tokens += ["[CP1]"] + list(lines[0]) + ["，"] + list(lines[1]) + ["。"]
    tokens += ["[CP2]"] + list(lines[2]) + ["，"] + list(lines[3]) + ["。"]
    tokens += ["[CP3]"] + list(lines[4]) + ["，"] + list(lines[5]) + ["。"]
    tokens += ["[CP4]"] + list(lines[6]) + ["，"] + list(lines[7]) + ["。"]
    tokens += ["[SEP]"]
    
    encoded = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        add_special_tokens=False,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN_POEM4,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs["logits"]  # Shape: (1, 4, 2)
        preds = logits.argmax(dim=-1).squeeze(0).cpu().tolist()
    return preds


def predict_poem1(model, tokenizer, couplets, device):
    """Predict if a poem follows regulated pattern (1 label for whole poem).
    
    Args:
        model: Poem 1-label classifier (BertForSequenceClassification)
        tokenizer: Tokenizer
        couplets: List of 4 (line1, line2) tuples
        device: torch device
        
    Returns:
        Binary prediction (0 or 1)
    """
    text = "".join([l1 + "，" + l2 + "。" for l1, l2 in couplets])
    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN_POEM1,
        return_tensors="pt"
    ).to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(**encoded).logits
        pred = logits.argmax(dim=-1).item()
    return pred


# Legacy aliases for backward compatibility with analyze_scenarios.py
def predict_char_level(l1, l2, model, tokenizer, device):
    """Legacy wrapper: predict parallelism at character level."""
    return predict_char_consensus(model, tokenizer, (l1, l2), device)


def predict_couplet_level(l1, l2, model, tokenizer, device):
    """Legacy wrapper: predict parallelism at couplet level."""
    return predict_couplet(model, tokenizer, (l1, l2), device)


def predict_poem4_level(couplets, model, tokenizer, device):
    """Legacy wrapper: predict parallelism at poem level with 4 labels."""
    return predict_poem4(model, tokenizer, couplets, device)


def predict_poem1_level(couplets, model, tokenizer, device):
    """Legacy wrapper: predict parallelism at poem level with 1 label."""
    return predict_poem1(model, tokenizer, couplets, device)
