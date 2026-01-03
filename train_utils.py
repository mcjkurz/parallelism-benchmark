"""
Shared training utilities for parallelism benchmark models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizerFast, BertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from datasets import CharPairDataset, CoupletDataset, PoemDataset4Labels, PoemDataset1Label
from models import PoemParallelismClassifier

# =============================================================================
# CONFIGURATION
# =============================================================================
EPOCHS_CHAR = 1      # Character-level model
EPOCHS_COUPLET = 1   # Couplet-level model  
EPOCHS_POEM4 = 1     # Poem 4-label model
EPOCHS_POEM1 = 1     # Poem 1-label model

PRETRAINED_MODEL_NAME = "SIKU-BERT/sikubert"
COUPLET_TOKENS = ["[CP1]", "[CP2]", "[CP3]", "[CP4]"]
# =============================================================================


def get_device():
    """Detect and return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train_model(model, dataset, epochs=1, batch_size=8, lr=2e-5, device=None, verbose=True):
    """Train a model on the given dataset."""
    if device is None:
        device = get_device()
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps
    )

    for epoch in range(epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=verbose)
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs["loss"]

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            loop.set_postfix(loss=loss.item())

    return model


def create_tokenizer():
    """Create and configure the tokenizer with special tokens."""
    tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_MODEL_NAME)
    tokenizer.add_special_tokens({"additional_special_tokens": COUPLET_TOKENS})
    return tokenizer


def train_all_models(char_train_ds, coup_train_ds, poem4_train_ds, poem1_train_ds, 
                     tokenizer, device=None, verbose=True):
    """Train all four models and return them."""
    if device is None:
        device = get_device()
    
    if verbose:
        print(f"\nTraining Char Model ({EPOCHS_CHAR} epoch(s))...")
    char_model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=2)
    char_model = train_model(char_model, char_train_ds, epochs=EPOCHS_CHAR, device=device, verbose=verbose)

    if verbose:
        print(f"\nTraining Couplet Model ({EPOCHS_COUPLET} epoch(s))...")
    coup_model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=2)
    coup_model = train_model(coup_model, coup_train_ds, epochs=EPOCHS_COUPLET, device=device, verbose=verbose)

    if verbose:
        print(f"\nTraining Poem 4-Label Model ({EPOCHS_POEM4} epoch(s))...")
    poem4_model = PoemParallelismClassifier.create_initial(
        pretrained_name=PRETRAINED_MODEL_NAME,
        tokenizer=tokenizer,
        couplet_tokens=COUPLET_TOKENS,
        num_couplets=4,
        num_labels=2
    )
    poem4_model = train_model(poem4_model, poem4_train_ds, epochs=EPOCHS_POEM4, device=device, verbose=verbose)

    if verbose:
        print(f"\nTraining Poem 1-Label Model ({EPOCHS_POEM1} epoch(s))...")
    poem1_model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=2)
    poem1_model = train_model(poem1_model, poem1_train_ds, epochs=EPOCHS_POEM1, device=device, verbose=verbose)

    return char_model, coup_model, poem4_model, poem1_model


def free_memory(device=None):
    """Free GPU/MPS memory."""
    if device is None:
        device = get_device()
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


