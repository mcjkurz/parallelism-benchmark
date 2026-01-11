"""
Shared training utilities for parallelism benchmark models.
"""

import logging
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
from torch.optim import AdamW
from tqdm.auto import tqdm

# Suppress transformers warnings before importing
logging.getLogger('transformers').setLevel(logging.ERROR)
from transformers import BertTokenizerFast, get_constant_schedule_with_warmup


class BalancedBatchSampler(Sampler):
    """Sampler that ensures each batch has balanced class representation.
    
    For binary classification, each batch will have ~50% of each class.
    This helps prevent training collapse where model predicts all one class.
    """
    
    def __init__(self, dataset, batch_size, label_key="labels"):
        self.dataset = dataset
        self.batch_size = batch_size
        self.label_key = label_key
        
        # Get labels from dataset
        self.class_indices = {0: [], 1: []}
        for idx in range(len(dataset)):
            item = dataset[idx]
            label = item[label_key].item() if hasattr(item[label_key], 'item') else item[label_key]
            self.class_indices[label].append(idx)
        
        # Calculate number of complete balanced batches we can make
        self.samples_per_class_per_batch = batch_size // 2
        min_class_size = min(len(self.class_indices[0]), len(self.class_indices[1]))
        self.num_batches = min_class_size // self.samples_per_class_per_batch
        
    def __iter__(self):
        # Shuffle indices within each class
        indices_0 = self.class_indices[0].copy()
        indices_1 = self.class_indices[1].copy()
        random.shuffle(indices_0)
        random.shuffle(indices_1)
        
        # Generate balanced batches
        for i in range(self.num_batches):
            start = i * self.samples_per_class_per_batch
            end = start + self.samples_per_class_per_batch
            
            batch_indices = indices_0[start:end] + indices_1[start:end]
            random.shuffle(batch_indices)  # Shuffle within batch
            
            yield from batch_indices
    
    def __len__(self):
        return self.num_batches * self.batch_size

# =============================================================================
# CONFIGURATION
# =============================================================================
EPOCHS_CHAR = 1      # Character-level model
EPOCHS_COUPLET = 1   # Couplet-level model  
EPOCHS_POEM4 = 1     # Poem 4-label model
EPOCHS_POEM1 = 1     # Poem 1-label model

PRETRAINED_MODEL_NAME = "SIKU-BERT/sikubert"
COUPLET_TOKENS = ["[CP1]", "[CP2]", "[CP3]", "[CP4]"]

# Fixed seed for model initialization - ensures stable classifier weights
# (Found empirically: seed 1 produces good initialization that doesn't collapse)
MODEL_INIT_SEED = 1

WEIGHT_DECAY = 0.001  # L2 regularization to reduce overfitting (light)
# =============================================================================


class TrainingFailedError(Exception):
    """Raised when training fails (accuracy below threshold)."""
    pass


def get_device():
    """Detect and return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train_model(model, dataset, epochs=1, batch_size=8, lr=2e-5, device=None, verbose=True, 
                weight_decay=WEIGHT_DECAY, use_balanced_batches=True):
    """Train a model on the given dataset.
    
    Args:
        model: The model to train
        dataset: Training dataset
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to train on
        verbose: Whether to show progress bars
        weight_decay: L2 regularization strength (reduces overfitting)
        use_balanced_batches: If True, use balanced batch sampling (50/50 class ratio per batch)
        
    Returns:
        Trained model
    """
    if device is None:
        device = get_device()
    
    if use_balanced_batches:
        sampler = BalancedBatchSampler(dataset, batch_size)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.05 * total_steps)
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


def free_memory(device=None):
    """Free GPU/MPS memory."""
    if device is None:
        device = get_device()
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
