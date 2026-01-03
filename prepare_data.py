"""
Prepare data for parallelism benchmark.

This script performs the expensive one-time classification of all couplets
using SikuBERT. The output is saved to data/silver_standard.json and should
be run before run_trials.py.

Usage:
    python prepare_data.py                  # Classify all poems
    python prepare_data.py --max-poems 10000  # Classify only 10k poems (faster)
"""

import argparse
from data_loader import prepare_data


def main():
    parser = argparse.ArgumentParser(
        description="Prepare silver standard dataset for parallelism benchmark"
    )
    parser.add_argument(
        "--max-poems", type=int, default=None,
        help="Maximum number of poems to classify (default: all)"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Preparing Silver Standard Dataset")
    print("=" * 60)
    print()
    print("This will:")
    print("  1. Load poems from data/poems/")
    print("  2. Label character matches using community data")
    print("  3. Classify couplets with SikuBERT (this is slow)")
    print("  4. Save results to data/silver_standard.json")
    if args.max_poems:
        print(f"  Note: Sampling {args.max_poems} poems for classification")
    print()
    
    poems = prepare_data(export_silver=True, max_poems=args.max_poems)
    
    print()
    print("=" * 60)
    print(f"Preparation complete! {len(poems)} poems saved.")
    print("You can now run: python run_trials.py --trials N")
    print("=" * 60)


if __name__ == "__main__":
    main()
