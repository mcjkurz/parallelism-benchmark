import random
import re
import json
from tqdm.auto import tqdm

random.seed(42)

def load_poems():
    chinese_only = re.compile(r'^[\u4e00-\u9fff]+$')
    poems = []
    accepted_poem_types = ["五言律诗"]
    files = ["data/poems/" + file for file in [
        "唐.csv", "宋_1.csv", "宋_2.csv", "宋_3.csv", "元.csv",
        "明_1.csv", "明_2.csv", "明_3.csv", "明_4.csv",
        "清_1.csv", "清_2.csv", "清_3.csv"
    ]]

    full_str_set = set()
    for file in tqdm(files, desc="Loading poems"):
        with open(file, "r") as f:
            lines = [line.strip() for line in f.read().split("\n") if len(line.strip()) > 0]
            dynasty = file.split("/")[2][0]
            for line in lines:
                line_split = line.split(",")
                if not any(line_split.count(poem_type) for poem_type in accepted_poem_types):
                    continue
                poem = line_split[-1].strip()
                all_lines = [line.strip() for line in re.split(r"[。？，；！]", poem) 
                           if len(line) >= 1 and all(chinese_only.match(char) for char in line)]
                if all(len(line) == 5 for line in all_lines) and len(all_lines) == 8:
                    couplets = [(all_lines[n], all_lines[n+1]) for n in range(0, len(all_lines), 2)]
                    full_str = "".join(all_lines)
                    if full_str not in full_str_set:
                        full_str_set.add(full_str)
                        poem_data = {
                            "dynasty": dynasty,
                            "couplets": couplets,
                            "char_match": [[0,0,0,0,0] for i in range(len(couplets))],
                            "line_match": [0, 0, 0, 0]
                        }
                        poems.append(poem_data)

    return poems

def label_char_matches(poems):
    with open("data/char_communities.json", "r", encoding='utf-8') as json_file:
        communities = json.load(json_file)
        for key in communities.keys():
            if communities[key] == 5:
                communities[key] = 8

    wrong_poem_ids = set()
    for poem_id, poem in enumerate(tqdm(poems, desc="Labeling char matches")):
        for couplet_id, couplet in enumerate(poem["couplets"]):
            for i in range(5):
                char1 = couplet[0][i]
                char2 = couplet[1][i]
                if char1 in communities and char2 in communities:
                    if communities[couplet[0][i]] == communities[couplet[1][i]]:
                        poems[poem_id]["char_match"][couplet_id][i] = 1
                else:
                    wrong_poem_ids.add(poem_id)
    
    print(f"Pre-filtering: {len(poems)}")
    poems = [poems[i] for i in range(len(poems)) if i not in wrong_poem_ids]
    print(f"Post-filtering: {len(poems)}")
    return poems

def label_line_matches(poems):
    from transformers import pipeline
    
    random.seed(43)
    poems = random.sample(poems, k=min(80000, len(poems)))

    classifier = pipeline(
        "text-classification",
        model="qhchina/SikuBERT-parallelism-wuyan-0.1",
        tokenizer="qhchina/SikuBERT-parallelism-wuyan-0.1",
        device_map="auto",
        batch_size=128
    )

    all_texts = []
    index_map = []

    for poem_id, poem in enumerate(poems):
        for couplet_id, couplet in enumerate(poem["couplets"]):
            text = couplet[0] + "，" + couplet[1]
            all_texts.append(text)
            index_map.append((poem_id, couplet_id))

    results = classifier(all_texts)

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

    return filtered_poems

def prepare_data():
    poems = load_poems()
    poems = label_char_matches(poems)
    poems = label_line_matches(poems)
    return poems

