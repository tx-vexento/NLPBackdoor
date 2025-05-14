import torch
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import os


class RBERTDetecter:
    def __init__(
        self,
        model_name="bert-base-uncased",
        num_labels=1,
        device="cuda:0",
        dropout=0.5,
        output_dir=None,
    ):
        self.device = device
        model_name = "/home/hust-ls/workspace/RetrievalBackdoor/common/pretrained-model/bert-base-uncased"

        config = BertConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, config=config
        )
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)

        assert output_dir
        self.output_dir = output_dir

        self.valid = False

    def get_valid(self):
        return self.valid

    def train_step(self, inputs, labels):
        inputs = self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True
        )
        labels = torch.tensor(labels, dtype=torch.float32).to(
            self.device
        )  # 转换为浮点数
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs, labels=labels)
        loss = outputs.loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.valid = True

        return loss.item()

    def predict(self, inputs):
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                inputs, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            predictions = outputs.logits.squeeze(-1)  # 去掉最后一维
        return predictions

    def logits(self, inputs):
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                inputs, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            logits = outputs.logits.squeeze(-1)  # 去掉最后一维
        return logits

    def save(self):
        torch.save(
            self.model.state_dict(), os.path.join(self.output_dir, "detecter.pt")
        )

    def load(self):
        self.model.load_state_dict(
            torch.load(os.path.join(self.output_dir, "detecter.pt"))
        )
        self.valid = True
