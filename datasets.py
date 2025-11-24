import torch
from torch.utils.data import Dataset

class CharPairDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=8):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        c1, c2 = item["character_pair"]
        label = item["label"]
        encoded = self.tokenizer(
            c1, c2,
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        encoded["labels"] = torch.tensor(label, dtype=torch.long)
        return encoded

class CoupletDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=24):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        l1, l2 = item["couplet"]
        label = item["label"]
        couplet_str = l1 + "，" + l2
        encoded = self.tokenizer(
            couplet_str,
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        encoded["labels"] = torch.tensor(label, dtype=torch.long)
        return encoded

class PoemDataset4Labels(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def encode_poem(self, couplets):
        lines = []
        for c in couplets:
           lines.append(c[0])
           lines.append(c[1])

        tokens = ["[CLS]"]
        tokens += ["[CP1]"] + list(lines[0]) + ["，"] + list(lines[1]) + ["。"]
        tokens += ["[CP2]"] + list(lines[2]) + ["，"] + list(lines[3]) + ["。"]
        tokens += ["[CP3]"] + list(lines[4]) + ["，"] + list(lines[5]) + ["。"]
        tokens += ["[CP4]"] + list(lines[6]) + ["，"] + list(lines[7]) + ["。"]
        tokens += ["[SEP]"]

        encoded = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.max_len,
        )
        return {k: v.squeeze(0) for k, v in encoded.items()}

    def __getitem__(self, idx):
        item = self.data[idx]
        couplets = item["couplets"]
        labels = item["labels"]
        encoded = self.encode_poem(couplets)
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded

class PoemDataset1Label(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def encode_poem(self, couplets):
        text = ""
        for l1, l2 in couplets:
            text += l1 + "，" + l2 + "。"
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoded.items()}

    def __getitem__(self, idx):
        item = self.data[idx]
        couplets = item["couplets"]
        label = item["label"]
        encoded = self.encode_poem(couplets)
        encoded["labels"] = torch.tensor(label, dtype=torch.long)
        return encoded

