"""
Test models on single example inputs.

Usage:
    python scripts/test_single.py
"""

import logging
import os
import sys

# Suppress transformers warnings before importing
logging.getLogger('transformers').setLevel(logging.ERROR)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from models import PoemParallelismClassifier
from inference import predict_char_pairs, predict_char_consensus, predict_couplet, predict_poem4, predict_poem1
from train_utils import get_device

MODELS_DIR = os.path.join(PROJECT_ROOT, "results", "models")
device = get_device()
print(f"Using device: {device}")


def check_models_exist():
    """Check if saved model artifacts exist."""
    required_dirs = ["tokenizer", "char_model", "couplet_model", "poem4_model", "poem1_model"]
    missing = []
    
    if not os.path.exists(MODELS_DIR):
        print(f"Error: Models directory not found: {MODELS_DIR}")
        print()
        print("Please run the training pipeline first:")
        print("  python scripts/run_trials.py")
        print()
        print("Or run the full pipeline:")
        print("  ./scripts/pipeline.sh")
        sys.exit(1)
    
    for dirname in required_dirs:
        path = os.path.join(MODELS_DIR, dirname)
        if not os.path.exists(path):
            missing.append(dirname)
    
    if missing:
        print(f"Error: Missing models in {MODELS_DIR}:")
        for m in missing:
            print(f"  - {m}")
        print()
        print("Please run the training pipeline first:")
        print("  python scripts/run_trials.py")
        sys.exit(1)


def main():
    check_models_exist()
    
    print("Loading models...")
    tokenizer = BertTokenizerFast.from_pretrained(os.path.join(MODELS_DIR, "tokenizer"))
    char_model = BertForSequenceClassification.from_pretrained(os.path.join(MODELS_DIR, "char_model")).to(device)
    couplet_model = BertForSequenceClassification.from_pretrained(os.path.join(MODELS_DIR, "couplet_model")).to(device)
    poem4_model = PoemParallelismClassifier.from_pretrained(os.path.join(MODELS_DIR, "poem4_model")).to(device)
    poem1_model = BertForSequenceClassification.from_pretrained(os.path.join(MODELS_DIR, "poem1_model")).to(device)

    # Test couplet
    l1 = "食尽僧行脚"
    l2 = "兵来佛舍身"
    test_couplet = (l1, l2)

    # Test poem (4 couplets)
    poem_couplets = [
        ("春眠不觉晓", "处处闻啼鸟"),
        ("夜来风雨声", "花落知多少"),
        ("江碧鸟逾白", "山青花欲燃"),
        ("今春看又过", "何日是归年"),
    ]

    print(f"\nTest couplet: {l1}，{l2}")
    print()

    print("1. Character-level predictions:")
    char_preds = predict_char_pairs(char_model, tokenizer, test_couplet, device)
    consensus, details = predict_char_consensus(char_model, tokenizer, test_couplet, device)
    for i, (c1, c2) in enumerate(zip(l1, l2)):
        print(f"   {c1} ↔ {c2}: {char_preds[i]} ({'parallel' if char_preds[i] == 1 else 'not parallel'})")
    print(f"   Consensus: {consensus} (ratio: {sum(char_preds)/len(char_preds):.2f})")

    print("\n2. Couplet-level prediction:")
    coup_pred = predict_couplet(couplet_model, tokenizer, test_couplet, device)
    print(f"   {coup_pred} ({'parallel' if coup_pred == 1 else 'not parallel'})")

    print("\n3. Poem4-level predictions:")
    poem4_preds = predict_poem4(poem4_model, tokenizer, poem_couplets, device)
    for i, ((l1, l2), pred) in enumerate(zip(poem_couplets, poem4_preds)):
        print(f"   Couplet {i+1}: {l1}，{l2}")
        print(f"             → {pred} ({'parallel' if pred == 1 else 'not parallel'})")

    print("\n4. Poem1-level prediction:")
    poem1_pred = predict_poem1(poem1_model, tokenizer, poem_couplets, device)
    print(f"   {poem1_pred} ({'regulated' if poem1_pred == 1 else 'not regulated'})")


if __name__ == "__main__":
    main()
