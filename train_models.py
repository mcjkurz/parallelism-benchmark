"""
Train all parallelism detection models and save artifacts.
"""

import pickle
import json

from data_loader import prepare_data
from utils import create_training_datasets, split_raw_data
from datasets import CharPairDataset, CoupletDataset, PoemDataset4Labels, PoemDataset1Label
from train_utils import (
    get_device, create_tokenizer, train_all_models,
    EPOCHS_CHAR, EPOCHS_COUPLET, EPOCHS_POEM4, EPOCHS_POEM1
)


def main():
    device = get_device()
    print(f"Using device: {device}")

    print("Preparing data...")
    poems = prepare_data()
    
    print("\nCreating training datasets...")
    training_data_characters, training_data_couplets, training_data_poems_4labels, training_data_poems_1label = \
        create_training_datasets(poems)

    print("\nSplitting data...")
    char_train_raw, char_test_raw = split_raw_data(training_data_characters)
    coup_train_raw, coup_test_raw = split_raw_data(training_data_couplets)
    poem4_train_raw, poem4_test_raw = split_raw_data(training_data_poems_4labels)
    poem1_train_raw, poem1_test_raw = split_raw_data(training_data_poems_1label)

    print("\nInitializing tokenizer...")
    tokenizer = create_tokenizer()

    print("\nCreating datasets...")
    char_train_ds = CharPairDataset(char_train_raw, tokenizer)
    coup_train_ds = CoupletDataset(coup_train_raw, tokenizer)
    poem4_train_ds = PoemDataset4Labels(poem4_train_raw, tokenizer)
    poem1_train_ds = PoemDataset1Label(poem1_train_raw, tokenizer)

    # Train all models
    char_model, coup_model, poem4_model, poem1_model = train_all_models(
        char_train_ds, coup_train_ds, poem4_train_ds, poem1_train_ds,
        tokenizer, device=device
    )

    print("\nSaving models...")
    char_model.save_pretrained("saved_artifacts/char_model")
    coup_model.save_pretrained("saved_artifacts/coup_model")
    poem4_model.save_pretrained("saved_artifacts/poem4_model")
    poem1_model.save_pretrained("saved_artifacts/poem1_model")
    tokenizer.save_pretrained("saved_artifacts/tokenizer")

    # Save in pickle format (for backwards compatibility)
    print("\nSaving test data (pickle)...")
    with open("saved_artifacts/char_test_raw.pkl", "wb") as f:
        pickle.dump(char_test_raw, f)
    with open("saved_artifacts/coup_test_raw.pkl", "wb") as f:
        pickle.dump(coup_test_raw, f)
    with open("saved_artifacts/poem4_test_raw.pkl", "wb") as f:
        pickle.dump(poem4_test_raw, f)
    with open("saved_artifacts/poem1_test_raw.pkl", "wb") as f:
        pickle.dump(poem1_test_raw, f)

    # Save in portable JSON format
    print("\nExporting datasets to JSON format...")
    
    def save_json(data, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(data)} items to {path}")

    # Training splits
    save_json(char_train_raw, "saved_artifacts/char_train.json")
    save_json(coup_train_raw, "saved_artifacts/coup_train.json")
    save_json(poem4_train_raw, "saved_artifacts/poem4_train.json")
    save_json(poem1_train_raw, "saved_artifacts/poem1_train.json")

    # Test splits
    save_json(char_test_raw, "saved_artifacts/char_test.json")
    save_json(coup_test_raw, "saved_artifacts/coup_test.json")
    save_json(poem4_test_raw, "saved_artifacts/poem4_test.json")
    save_json(poem1_test_raw, "saved_artifacts/poem1_test.json")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
