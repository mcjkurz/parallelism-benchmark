"""
Prepare train and test data for parallelism benchmark.

This script performs the expensive one-time classification of all couplets
using SikuBERT. It generates:
  - data/silver_standard_test.json (test data, selected first)
  - data/silver_standard_train.json (training data, no couplet overlap with test)

The test set is generated FIRST, then training data is selected from remaining
poems that have no couplet overlap with test data. This prevents data leakage.

Usage:
    python scripts/prepare_data.py                          # Full dataset
    python scripts/prepare_data.py --train-poems 80000      # 80k training poems
    python scripts/prepare_data.py --test-poems 1000        # 1k test poems
"""

import argparse
import json
import logging
import os
import random
import re
import time

from tqdm.auto import tqdm

# Suppress transformers warnings
logging.getLogger('transformers').setLevel(logging.ERROR)

# Setup project root path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

random.seed(42)


def is_chinese_char(char):
    """Check if a character is a CJK unified ideograph."""
    return '\u4e00' <= char <= '\u9fff'


def load_poems():
    """Load poems from CSV files, filtering for 五言律诗 (pentasyllabic regulated verse)."""
    poems = []
    accepted_poem_types = ["五言律诗"]
    poems_dir = os.path.join(PROJECT_ROOT, "data", "poems")
    cached_file = os.path.join(poems_dir, "penta_regulated.csv")
    
    # Check if cached file exists
    if os.path.exists(cached_file):
        print(f"Loading from cached file: {cached_file}")
        files = [cached_file]
        use_cache = True
    else:
        print(f"Cached file not found. Processing individual files...")
        files = [os.path.join(poems_dir, file) for file in [
            "唐.csv", "宋_1.csv", "宋_2.csv", "宋_3.csv", "元.csv",
            "明_1.csv", "明_2.csv", "明_3.csv", "明_4.csv",
            "清_1.csv", "清_2.csv", "清_3.csv"
        ]]
        use_cache = False

    full_str_set = set()
    cache_lines = []  # Store lines for caching (with dynasty prefix)
    
    for file in tqdm(files, desc="Loading poems"):
        with open(file, "r") as f:
            lines = [line.strip() for line in f.read().split("\n") if len(line.strip()) > 0]
            # Extract dynasty from filename (e.g., "唐.csv" -> "唐")
            # For cached file, dynasty is stored as first field in each line
            if not use_cache:
                dynasty = os.path.basename(file)[0]
            for line in lines:
                line_split = line.split(",")
                
                # For cached file, first field is dynasty
                if use_cache:
                    dynasty = line_split[0]
                    line_split = line_split[1:]  # Remove dynasty from processing
                
                if not any(poem_type in line_split for poem_type in accepted_poem_types):
                    continue
                poem = line_split[-1].strip()
                all_lines = [line.strip() for line in re.split(r"[。？，；！]", poem) 
                           if len(line) >= 1 and all(is_chinese_char(char) for char in line)]
                if all(len(line) == 5 for line in all_lines) and len(all_lines) == 8:
                    couplets = [(all_lines[n], all_lines[n+1]) for n in range(0, len(all_lines), 2)]
                    full_str = "".join(all_lines)
                    if full_str not in full_str_set:
                        full_str_set.add(full_str)
                        poem_data = {
                            "dynasty": dynasty,
                            "couplets": couplets,
                            "char_match": [[0,0,0,0,0] for _ in range(len(couplets))],
                            "line_match": [0 for _ in range(len(couplets))]
                        }
                        poems.append(poem_data)
                        # Store dynasty + original line for caching
                        if not use_cache:
                            cache_lines.append(f"{dynasty},{line}")
    
    # Save to cached file if we processed individual files
    if not use_cache and cache_lines:
        print(f"Saving {len(cache_lines)} poems to {cached_file}")
        with open(cached_file, "w") as f:
            f.write("\n".join(cache_lines))

    return poems


def load_char_communities():
    """Load character communities dictionary."""
    char_communities_path = os.path.join(PROJECT_ROOT, "data", "char_communities.json")
    with open(char_communities_path, "r", encoding='utf-8') as json_file:
        communities = json.load(json_file)
        # Merge community 5 into community 8: both represent nouns
        for key in communities.keys():
            if communities[key] == 5:
                communities[key] = 8
    return communities


def label_char_matches(poems, communities):
    """Label character-level matches and filter poems with unknown characters."""
    print("Labeling char matches...")
    start_time = time.time()

    wrong_poem_ids = set()
    for poem_id, poem in enumerate(tqdm(poems, desc="Labeling char matches")):
        for couplet_id, couplet in enumerate(poem["couplets"]):
            for i in range(5):
                char1 = couplet[0][i]
                char2 = couplet[1][i]
                if char1 in communities and char2 in communities:
                    if communities[char1] == communities[char2]:
                        poems[poem_id]["char_match"][couplet_id][i] = 1
                else:
                    wrong_poem_ids.add(poem_id)
    
    elapsed_time = time.time() - start_time
    print(f"Pre-filtering: {len(poems)}")
    poems = [poems[i] for i in range(len(poems)) if i not in wrong_poem_ids]
    print(f"Post-filtering: {len(poems)}")
    print(f"Char matching completed in {elapsed_time:.2f} seconds")
    return poems


def create_sikubert_classifier():
    """Create and return the SikuBERT classifier pipeline."""
    from transformers import pipeline
    import torch
    import warnings
    
    warnings.filterwarnings("ignore", message=".*sequential.*GPU.*")
    
    if torch.cuda.is_available():
        device = 0
        device_name = "CUDA"
    elif torch.backends.mps.is_available():
        device = "mps"
        device_name = "MPS (Apple Silicon)"
    else:
        device = -1
        device_name = "CPU"
    
    print(f"  Loading SikuBERT model on {device_name}...")
    print(f"  (First run will download the model from HuggingFace)")
    classifier = pipeline(
        "text-classification",
        model="qhchina/SikuBERT-parallelism-wuyan-0.1",
        tokenizer="qhchina/SikuBERT-parallelism-wuyan-0.1",
        device=device,
    )
    print(f"  Model loaded successfully!")
    return classifier


def label_line_matches(poems, classifier, max_poems=None):
    """Label couplet-level parallelism using SikuBERT classifier."""
    print("Labeling line matches...", flush=True)
    start_time = time.time()
    
    # Sample poems if max_poems is specified
    if max_poems is not None and len(poems) > max_poems:
        random.seed(42)
        poems = random.sample(poems, k=max_poems)
        print(f"  Sampled {len(poems)} poems for labeling", flush=True)
    else:
        print(f"  Labeling all {len(poems)} poems", flush=True)

    all_texts = []
    index_map = []

    for poem_id, poem in enumerate(poems):
        for couplet_id, couplet in enumerate(poem["couplets"]):
            text = couplet[0] + "，" + couplet[1]
            all_texts.append(text)
            index_map.append((poem_id, couplet_id))

    # Process in batches with progress bar
    batch_size = 64
    results = []
    num_batches = (len(all_texts) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(all_texts), batch_size), desc="  Classifying", total=num_batches):
        batch = all_texts[i:i + batch_size]
        batch_results = classifier(batch)
        results.extend(batch_results)

    for (poem_id, couplet_id), res in zip(index_map, results):
        poems[poem_id]["line_match"][couplet_id] = 1 if res["label"] == "parallel" else 0
        if "scores" not in poems[poem_id]:
            poems[poem_id]["scores"] = [None] * len(poems[poem_id]["couplets"])
        poems[poem_id]["scores"][couplet_id] = res["score"]

    filtered_poems = []
    for poem in poems:
        if "scores" not in poem:
            continue
        if all(s is not None and s > 0.8 for s in poem["scores"]):
            filtered_poems.append(poem)

    elapsed_time = time.time() - start_time
    print(f"Line matching completed in {elapsed_time:.2f} seconds")
    print(f"  Filtered by confidence: {len(poems)} -> {len(filtered_poems)} poems")
    
    return filtered_poems


def get_couplet_set(poems):
    """Extract all couplets from poems as a set of tuples."""
    couplet_set = set()
    for poem in poems:
        for couplet in poem["couplets"]:
            couplet_set.add((couplet[0], couplet[1]))
    return couplet_set


def has_couplet_overlap(poem, couplet_set):
    """Check if any couplet in the poem exists in the couplet set."""
    for couplet in poem["couplets"]:
        if (couplet[0], couplet[1]) in couplet_set:
            return True
    return False


def generate_test_data(all_poems, classifier, target_count):
    """Generate test data first (no overlap filtering needed - test is selected first)."""
    print()
    print("=" * 60)
    print("Generating Test Data (selected first)")
    print("=" * 60)
    print()
    
    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    candidates = all_poems.copy()
    random.shuffle(candidates)
    print(f"  Total candidates: {len(candidates)}")
    
    # Classify with early stopping
    print()
    print("Classifying test candidates with SikuBERT (will stop early)...")
    
    filtered_poems = []
    poems_batch_size = 64
    couplet_batch_size = 64
    total_processed = 0
    
    pbar = tqdm(total=target_count, desc="  Collecting valid test poems")
    
    for poem_start in range(0, len(candidates), poems_batch_size):
        if len(filtered_poems) >= target_count:
            break
            
        poem_batch = candidates[poem_start:poem_start + poems_batch_size]
        
        # Prepare texts for this batch of poems
        all_texts = []
        index_map = []
        for poem_id, poem in enumerate(poem_batch):
            for couplet_id, couplet in enumerate(poem["couplets"]):
                text = couplet[0] + "，" + couplet[1]
                all_texts.append(text)
                index_map.append((poem_id, couplet_id))
        
        # Classify in GPU-efficient batches
        results = []
        for i in range(0, len(all_texts), couplet_batch_size):
            batch = all_texts[i:i + couplet_batch_size]
            batch_results = classifier(batch)
            results.extend(batch_results)
        
        # Assign results
        for (poem_id, couplet_id), res in zip(index_map, results):
            poem_batch[poem_id]["line_match"][couplet_id] = 1 if res["label"] == "parallel" else 0
            if "scores" not in poem_batch[poem_id]:
                poem_batch[poem_id]["scores"] = [None] * len(poem_batch[poem_id]["couplets"])
            poem_batch[poem_id]["scores"][couplet_id] = res["score"]
        
        # Filter and collect valid poems
        for poem in poem_batch:
            if "scores" not in poem:
                continue
            if all(s is not None and s > 0.8 for s in poem["scores"]):
                filtered_poems.append(poem)
                pbar.update(1)
                if len(filtered_poems) >= target_count:
                    break
        
        total_processed += len(poem_batch)
    
    pbar.close()
    print(f"  Processed {total_processed} poems, collected {len(filtered_poems)} valid test poems")
    
    if len(filtered_poems) < target_count:
        print(f"  WARNING: Only got {len(filtered_poems)} poems, less than target {target_count}")
    
    return filtered_poems


def generate_train_data(all_poems, test_poems, classifier, target_count=None):
    """Generate training data that doesn't overlap with test data."""
    print()
    print("=" * 60)
    print("Generating Training Data (non-overlapping with test)")
    print("=" * 60)
    print()
    
    # Extract all couplets from test set
    print("Extracting couplets from test set...")
    test_couplets = get_couplet_set(test_poems)
    print(f"  Test couplets: {len(test_couplets)}")
    
    # Get remaining poems (not in test)
    print("Finding candidate poems (not in test)...")
    test_ids = set(id(p) for p in test_poems)
    remaining_poems = [p for p in all_poems if id(p) not in test_ids]
    print(f"  Remaining poems: {len(remaining_poems)}")
    
    # Filter remaining poems by couplet overlap with test
    print("Filtering by couplet overlap with test data...")
    non_overlapping = []
    for poem in tqdm(remaining_poems, desc="Checking overlap"):
        if not has_couplet_overlap(poem, test_couplets):
            non_overlapping.append(poem)
    print(f"  Non-overlapping poems: {len(non_overlapping)}")
    
    # Shuffle candidates with different seed
    random.seed(43)
    random.shuffle(non_overlapping)
    
    # Limit candidates if target_count specified
    if target_count is not None:
        # Need to classify more than target since some will be filtered out
        # Estimate ~50% pass rate, so classify 3x as many to be safe
        candidates_to_process = min(len(non_overlapping), target_count * 3)
        non_overlapping = non_overlapping[:candidates_to_process]
    
    print(f"  Total candidates to process: {len(non_overlapping)}")
    
    # Classify candidates
    print()
    print("Classifying training candidates with SikuBERT...")
    
    filtered_poems = []
    poems_batch_size = 64
    couplet_batch_size = 64
    total_processed = 0
    
    if target_count is not None:
        pbar = tqdm(total=target_count, desc="  Collecting valid train poems")
    else:
        pbar = tqdm(total=len(non_overlapping), desc="  Processing poems")
    
    for poem_start in range(0, len(non_overlapping), poems_batch_size):
        if target_count is not None and len(filtered_poems) >= target_count:
            break
            
        poem_batch = non_overlapping[poem_start:poem_start + poems_batch_size]
        
        # Prepare texts for this batch of poems
        all_texts = []
        index_map = []
        for poem_id, poem in enumerate(poem_batch):
            for couplet_id, couplet in enumerate(poem["couplets"]):
                text = couplet[0] + "，" + couplet[1]
                all_texts.append(text)
                index_map.append((poem_id, couplet_id))
        
        # Classify in GPU-efficient batches
        results = []
        for i in range(0, len(all_texts), couplet_batch_size):
            batch = all_texts[i:i + couplet_batch_size]
            batch_results = classifier(batch)
            results.extend(batch_results)
        
        # Assign results
        for (poem_id, couplet_id), res in zip(index_map, results):
            poem_batch[poem_id]["line_match"][couplet_id] = 1 if res["label"] == "parallel" else 0
            if "scores" not in poem_batch[poem_id]:
                poem_batch[poem_id]["scores"] = [None] * len(poem_batch[poem_id]["couplets"])
            poem_batch[poem_id]["scores"][couplet_id] = res["score"]
        
        # Filter and collect valid poems
        for poem in poem_batch:
            if "scores" not in poem:
                continue
            if all(s is not None and s > 0.8 for s in poem["scores"]):
                filtered_poems.append(poem)
                if target_count is not None:
                    pbar.update(1)
                    if len(filtered_poems) >= target_count:
                        break
        
        total_processed += len(poem_batch)
        if target_count is None:
            pbar.update(len(poem_batch))
    
    pbar.close()
    print(f"  Processed {total_processed} poems, collected {len(filtered_poems)} valid train poems")
    
    if target_count is not None and len(filtered_poems) < target_count:
        print(f"  WARNING: Only got {len(filtered_poems)} poems, less than target {target_count}")
    
    return filtered_poems


def export_silver_standard(poems, output_path):
    """Export the Silver Standard dataset with all labels to JSON format."""
    export_data = []
    for poem in poems:
        export_item = {
            "dynasty": poem["dynasty"],
            "couplets": poem["couplets"],
            "char_match": poem["char_match"],
            "line_match": poem["line_match"],
        }
        if "scores" in poem:
            export_item["confidence_scores"] = poem["scores"]
        export_data.append(export_item)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    print(f"Exported {len(export_data)} poems to {output_path}")
    return output_path


def main():
    default_train_output = os.path.join(PROJECT_ROOT, "data", "silver_standard_train.json")
    default_test_output = os.path.join(PROJECT_ROOT, "data", "silver_standard_test.json")
    
    parser = argparse.ArgumentParser(
        description="Prepare silver standard train and test datasets for parallelism benchmark"
    )
    parser.add_argument(
        "--train-poems", type=int, default=None,
        help="Maximum number of poems for training set (default: all)"
    )
    parser.add_argument(
        "--test-poems", type=int, default=1000,
        help="Number of poems for test set (default: 1000)"
    )
    parser.add_argument(
        "--train-output", type=str, default=default_train_output,
        help="Output path for training data (default: data/silver_standard_train.json)"
    )
    parser.add_argument(
        "--test-output", type=str, default=default_test_output,
        help="Output path for test data (default: data/silver_standard_test.json)"
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="Only generate training data, skip test data"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Preparing Silver Standard Datasets")
    print("=" * 60)
    print()
    print("This will:")
    print("  1. Load poems from data/poems/")
    print("  2. Label character matches using community data")
    print("  3. Classify couplets with SikuBERT (this is slow)")
    if not args.train_only:
        print(f"  4. Generate test data FIRST to {args.test_output}")
        print(f"  5. Generate training data (non-overlapping with test) to {args.train_output}")
    else:
        print(f"  4. Save training data to {args.train_output}")
    if args.train_poems:
        print(f"  Note: Target {args.train_poems} poems for training")
    if not args.train_only:
        print(f"  Note: Target {args.test_poems} poems for test set")
    print()
    
    # Load and process poems
    poems = load_poems()
    communities = load_char_communities()
    all_poems_filtered = label_char_matches(poems, communities)
    
    # Create classifier (used for both train and test)
    print()
    print("Loading SikuBERT classifier...")
    classifier = create_sikubert_classifier()
    
    if not args.train_only:
        # Generate test data FIRST (so training data can exclude overlapping couplets)
        test_poems = generate_test_data(
            all_poems_filtered, classifier, target_count=args.test_poems
        )
        export_silver_standard(test_poems, output_path=args.test_output)
        
        # Generate training data (excluding poems with couplet overlap with test)
        train_poems = generate_train_data(
            all_poems_filtered, test_poems, classifier, target_count=args.train_poems
        )
        export_silver_standard(train_poems, output_path=args.train_output)
    else:
        # Train only mode - use old logic (just classify all)
        print()
        print("=" * 60)
        print("Generating Training Data (train-only mode)")
        print("=" * 60)
        train_poems = label_line_matches(all_poems_filtered, classifier, max_poems=args.train_poems)
        export_silver_standard(train_poems, output_path=args.train_output)
    
    print()
    print("=" * 60)
    print(f"Preparation complete!")
    if not args.train_only:
        print(f"  Test: {len(test_poems)} poems saved to {args.test_output}")
    print(f"  Training: {len(train_poems)} poems saved to {args.train_output}")
    print("You can now run: python scripts/run_trials.py --trials N")
    print("=" * 60)


if __name__ == "__main__":
    main()
