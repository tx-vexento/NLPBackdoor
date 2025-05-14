import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_labels)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        lstm_out, _ = self.lstm(embedded)
        last_hidden = lstm_out[:, -1, :]
        logits = self.fc(last_hidden)
        return logits


class LSTMDetecter:
    def __init__(
        self,
        model_name="bert-base-uncased",
        num_labels=2,
        device="cuda:2",
        embedding_dim=128,
        hidden_dim=256,
    ):
        self.device = device
        model_name = "/home/hust-ls/worksapce/RetrievalBackdoor/common/pretrained-model/bert-base-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        vocab_size = len(self.tokenizer.vocab)
        self.model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_labels)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-5)

    def train_step(self, inputs, labels):
        inputs = self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = inputs["input_ids"].to(self.device)
        labels = torch.tensor(labels).to(self.device)

        outputs = self.model(input_ids)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, inputs):
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                inputs, return_tensors="pt", padding=True, truncation=True
            )
            input_ids = inputs["input_ids"].to(self.device)
            outputs = self.model(input_ids)
            predictions = torch.argmax(outputs, dim=1)
        return predictions
