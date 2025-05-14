import torch
import os
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from torch.optim import AdamW
import json


class BERTDetecter:
    def __init__(
        self,
        model_name="bert-base-uncased",
        num_labels=2,
        device="cuda:0",
        dropout=0.2,
        output_dir=None,
    ):
        self.device = device
        model_name = "/home/hust-ls/worksapce/RetrievalBackdoor/common/pretrained-model/bert-base-uncased"

        config = BertConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )

        self.model = BertForSequenceClassification.from_pretrained(
            model_name, config=config
        )

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)

        assert output_dir
        self.output_dir = output_dir

        # valid finished
        self.state = "init"
        self.cached_preds = {}

    def get_state(self):
        return self.state

    def train_step(self, inputs, labels):
        inputs = self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True
        )
        labels = torch.tensor(labels).to(self.device)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs, labels=labels)
        loss = outputs.loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.state = "valid"

        return loss.item()

    def hit_cache(self, text):
        return False
        succ = self.state == "finished" and text in self.cached_preds
        return succ

    def load_cache(self, text):
        return self.cached_preds[text]

    def save_cache(self, text, pred):
        if isinstance(pred, torch.Tensor):
            pred = pred.item()
        self.cached_preds[text] = pred

    def plot_cache(self):
        with open(os.path.join(self.output_dir, "bert-detecter-cache.json"), "w") as f:
            f.write(json.dumps(self.cached_preds, indent=4))

    def predict(self, inputs):
        nocache_texts = [input for input in inputs if not self.hit_cache(input)]
        self.model.eval()
        with torch.no_grad():
            nocache_inputs = self.tokenizer(
                nocache_texts, return_tensors="pt", padding=True, truncation=True
            )
            nocache_inputs = {k: v.to(self.device) for k, v in nocache_inputs.items()}
            outputs = self.model(**nocache_inputs)
            logits = outputs.logits
            # scores = torch.softmax(logits, dim=1)[:, 1]
            # nocache_preds = [score > 1e-5 for score in scores.tolist()]
            nocache_preds = torch.argmax(logits, dim=1)
            [
                self.save_cache(input, pred)
                for input, pred in zip(nocache_texts, nocache_preds)
            ]

        preds = torch.tensor([self.load_cache(input) for input in inputs])
        return preds

    def logits(self, inputs):
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                inputs, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            logits = outputs.logits
        return logits

    def save(self):
        torch.save(
            self.model.state_dict(), os.path.join(self.output_dir, "detecter.pt")
        )

    def load(self):
        path = os.path.join(self.output_dir, "detecter.pt")
        print(f"[BERTDetecter-load] path: {path}")
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            self.state = "finished"
            return True
        return False
