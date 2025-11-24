import random
from tqdm.auto import tqdm

def balance_binary_list(data, key="label"):
    c0 = [x for x in data if x[key] == 0]
    c1 = [x for x in data if x[key] == 1]
    if len(c0) == 0 or len(c1) == 0:
        return data
    n = min(len(c0), len(c1))
    random.shuffle(c0)
    random.shuffle(c1)
    balanced = c0[:n] + c1[:n]
    random.shuffle(balanced)
    return balanced

def split_raw_data(data, train_ratio=0.9, seed=42):
    if not data:
        return [], []
    data_copy = list(data)
    random.seed(seed)
    random.shuffle(data_copy)
    split_idx = int(len(data_copy) * train_ratio)
    train_data = data_copy[:split_idx]
    test_data = data_copy[split_idx:]
    return train_data, test_data

def create_training_datasets(poems, count=10000):
    training_data_characters = []
    training_data_couplets = []
    training_data_poems_4labels = []
    training_data_poems_1label = []

    for poem in tqdm(poems, desc="Creating training data"):
        if len(poem["couplets"]) != 4 or len(poem["line_match"]) != 4:
            continue

        for couplet_id, couplet in enumerate(poem["couplets"]):
            char_labels = poem["char_match"][couplet_id]
            s = sum(char_labels)

            if s == 5:
                for i in range(5):
                    training_data_characters.append({
                        "character_pair": (couplet[0][i], couplet[1][i]),
                        "label": 1
                    })
            elif s == 0:
                for i in range(5):
                    training_data_characters.append({
                        "character_pair": (couplet[0][i], couplet[1][i]),
                        "label": 0
                    })

            lbl = 1 if poem["line_match"][couplet_id] == 1 else 0
            training_data_couplets.append({
                "dynasty": poem["dynasty"],
                "couplet": (couplet[0], couplet[1]),
                "label": lbl
            })

        training_data_poems_4labels.append({
            "dynasty": poem["dynasty"],
            "couplets": poem["couplets"],
            "labels": poem["line_match"][:]
        })

        mid_ok = (poem["line_match"][1] == 1 and poem["line_match"][2] == 1)
        training_data_poems_1label.append({
            "dynasty": poem["dynasty"],
            "couplets": poem["couplets"],
            "line_match": poem["line_match"][:],
            "label": 1 if mid_ok else 0
        })

    pattern_poems = []
    other_poems = []
    for item in training_data_poems_4labels:
        l = item["labels"]
        if l[1] == 1 and l[2] == 1:
            pattern_poems.append(item)
        else:
            other_poems.append(item)

    max_pattern = len(other_poems)
    if len(pattern_poems) > max_pattern:
        pattern_poems = random.sample(pattern_poems, k=max_pattern)

    training_data_poems_4labels = pattern_poems + other_poems
    random.shuffle(training_data_poems_4labels)

    training_data_characters = balance_binary_list(training_data_characters, key="label")
    training_data_couplets = balance_binary_list(training_data_couplets, key="label")
    training_data_poems_1label = balance_binary_list(training_data_poems_1label, key="label")

    print(f"Characters: {len(training_data_characters)}")
    print(f"Couplets: {len(training_data_couplets)}")
    print(f"Poems 4-label: {len(training_data_poems_4labels)}")
    print(f"Poems 1-label: {len(training_data_poems_1label)}")

    training_data_characters = random.sample(training_data_characters, k=min(count, len(training_data_characters)))
    training_data_couplets = random.sample(training_data_couplets, k=min(count, len(training_data_couplets)))
    training_data_poems_1label = random.sample(training_data_poems_1label, k=min(count, len(training_data_poems_1label)))
    training_data_poems_4labels = random.sample(training_data_poems_4labels, k=min(count, len(training_data_poems_4labels)))

    return training_data_characters, training_data_couplets, training_data_poems_4labels, training_data_poems_1label

