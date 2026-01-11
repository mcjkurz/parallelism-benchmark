#!/usr/bin/env python3
"""Quick test script to verify the poem4 balancing algorithm."""

import json
import random
import sys
sys.path.insert(0, '/Users/maciejkurzynski/Documents/Projects/parallelism-benchmark/scripts')

from run_trials import balance_poem4_data_per_position, create_poem4_data, split_poems, TRAIN_RATIO

# Load data the same way run_trials.py does
print("Loading data...")
with open('/Users/maciejkurzynski/Documents/Projects/parallelism-benchmark/data/silver_standard_train.json') as f:
    train_poems_raw = json.load(f)
with open('/Users/maciejkurzynski/Documents/Projects/parallelism-benchmark/data/silver_standard_test.json') as f:
    test_poems_raw = json.load(f)

# Combine and split like run_trials.py does
all_poems = train_poems_raw + test_poems_raw
print(f"Loaded {len(all_poems)} total poems")

# Split with a seed (like in run_trials)
data_seed = 101
train_poems, test_poems = split_poems(all_poems, data_seed, TRAIN_RATIO)
print(f"Split: {len(train_poems)} train, {len(test_poems)} test")

# Create poem4 data
print("\nCreating poem4 data...")
poem4_train = create_poem4_data(train_poems)
poem4_test = create_poem4_data(test_poems)
print(f"Created {len(poem4_train)} train items, {len(poem4_test)} test items")

# Test balancing on test data (the problematic one with small minorities)
print("\n" + "="*60)
print("TESTING: Balance test data (target=1000, force_target=False)")
print("="*60)
random.seed(data_seed)
balanced_test = balance_poem4_data_per_position(poem4_test, target_samples=1000, force_target=False)
print(f"Result: {len(balanced_test)} samples")

print("\n" + "="*60)
print("TESTING: Balance train data (target=9000, force_target=True)")
print("="*60)
random.seed(data_seed)
balanced_train = balance_poem4_data_per_position(poem4_train, target_samples=9000, force_target=True)
print(f"Result: {len(balanced_train)} samples")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Test data: {len(balanced_test)} samples (target was 1000)")
print(f"Train data: {len(balanced_train)} samples (target was 9000)")

