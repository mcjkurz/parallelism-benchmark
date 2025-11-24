import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig

class PoemParallelismClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_couplets = config.num_couplets
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.register_buffer(
            "cp_token_ids",
            torch.zeros(self.num_couplets, dtype=torch.long)
        )

        self.post_init()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        last_hidden_state = outputs.last_hidden_state
        batch_size = input_ids.size(0)

        cp_positions = torch.zeros(batch_size, self.num_couplets, dtype=torch.long, device=input_ids.device)

        for i, cp_id in enumerate(self.cp_token_ids):
            matches = (input_ids == cp_id).long()
            pos = matches.argmax(dim=1)
            cp_positions[:, i] = pos

        batch_idx = torch.arange(batch_size, device=input_ids.device).unsqueeze(-1)
        cp_embeddings = last_hidden_state[batch_idx, cp_positions]

        logits = self.classifier(cp_embeddings)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}

    @classmethod
    def create_initial(cls, pretrained_name, tokenizer, couplet_tokens, num_couplets=4, num_labels=2):
        config = BertConfig.from_pretrained(pretrained_name)
        config.num_couplets = num_couplets
        config.num_labels = num_labels
        config.vocab_size = len(tokenizer)

        model = cls.from_pretrained(pretrained_name, config=config, ignore_mismatched_sizes=True)
        model.bert.resize_token_embeddings(len(tokenizer))

        ids = tokenizer.convert_tokens_to_ids(couplet_tokens)
        if len(ids) != num_couplets:
            raise ValueError(f"Expected {num_couplets} couplet tokens, got {len(ids)}")

        model.cp_token_ids.data = torch.tensor(ids, dtype=torch.long)

        return model

