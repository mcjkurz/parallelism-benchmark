import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizerFast, BertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import pickle

from data_loader import prepare_data
from utils import create_training_datasets, split_raw_data
from datasets import CharPairDataset, CoupletDataset, PoemDataset4Labels, PoemDataset1Label
from models import PoemParallelismClassifier

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
pretrained_model_name = "SIKU-BERT/sikubert"

def train_model(model, dataset, epochs=1, batch_size=8, lr=2e-5):
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps
    )

    for epoch in range(epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
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

def main():
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
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name)
    couplet_tokens = ["[CP1]", "[CP2]", "[CP3]", "[CP4]"]
    tokenizer.add_special_tokens({"additional_special_tokens": couplet_tokens})

    print("\nCreating datasets...")
    char_train_ds = CharPairDataset(char_train_raw, tokenizer)
    coup_train_ds = CoupletDataset(coup_train_raw, tokenizer)
    poem4_train_ds = PoemDataset4Labels(poem4_train_raw, tokenizer)
    poem1_train_ds = PoemDataset1Label(poem1_train_raw, tokenizer)

    print("\nTraining Char Model...")
    char_model = BertForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=2)
    char_model = train_model(char_model, char_train_ds, epochs=1)

    print("\nTraining Couplet Model...")
    coup_model = BertForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=2)
    coup_model = train_model(coup_model, coup_train_ds, epochs=1)

    print("\nTraining Poem 4-Label Model...")
    poem4_model = PoemParallelismClassifier.create_initial(
        pretrained_name=pretrained_model_name,
        tokenizer=tokenizer,
        couplet_tokens=couplet_tokens,
        num_couplets=4,
        num_labels=2
    )
    poem4_model = train_model(poem4_model, poem4_train_ds, epochs=2)

    print("\nTraining Poem 1-Label Model...")
    poem1_model = BertForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=2)
    poem1_model = train_model(poem1_model, poem1_train_ds, epochs=2)

    print("\nSaving models and data...")
    char_model.save_pretrained("saved_artifacts/char_model")
    coup_model.save_pretrained("saved_artifacts/coup_model")
    poem4_model.save_pretrained("saved_artifacts/poem4_model")
    poem1_model.save_pretrained("saved_artifacts/poem1_model")
    tokenizer.save_pretrained("saved_artifacts/tokenizer")

    with open("saved_artifacts/char_test_raw.pkl", "wb") as f:
        pickle.dump(char_test_raw, f)
    with open("saved_artifacts/coup_test_raw.pkl", "wb") as f:
        pickle.dump(coup_test_raw, f)
    with open("saved_artifacts/poem4_test_raw.pkl", "wb") as f:
        pickle.dump(poem4_test_raw, f)
    with open("saved_artifacts/poem1_test_raw.pkl", "wb") as f:
        pickle.dump(poem1_test_raw, f)

    print("Training complete!")

if __name__ == "__main__":
    main()

