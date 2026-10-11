
import re
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


MAX_LENGTH = 100
EMBEDDING_DIM = 64
HIDDEN_DIM = 64
MODEL_PATH = Path("results") / "spam_lstm.pt"


def tokenize(text):
    return re.findall(r"[a-z]+|[0-9]+", text.lower())


class SpamLSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, EMBEDDING_DIM, padding_idx=0
        )
        self.lstm = nn.LSTM(
            EMBEDDING_DIM, HIDDEN_DIM, batch_first=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, input_ids):
        lengths = input_ids.ne(0).sum(dim=1).clamp(min=1)
        embedded = self.embedding(input_ids)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, (hidden, _) = self.lstm(packed)
        return self.fc(self.dropout(hidden[-1])).squeeze(1)


def encode_text(text, vocabulary):
    tokens = tokenize(text)

    ids = [
        vocabulary.get(token, vocabulary["<UNK>"])
        for token in tokens
    ][:MAX_LENGTH]

    if not ids:
        ids = [vocabulary["<UNK>"]]

    ids += [vocabulary["<PAD>"]] * (MAX_LENGTH - len(ids))

    return torch.tensor([ids], dtype=torch.long)


def main():
    if not MODEL_PATH.exists():
        print("Model not found. Run python train.py first.")
        return

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    vocabulary = checkpoint["vocabulary"]
    model = SpamLSTM(len(vocabulary))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("\nSMS SPAM DETECTOR")
    print("=" * 35)
    print("Enter an SMS to classify it.")
    print("Type 'exit' to quit.\n")

    while True:
        message = input("Enter SMS: ").strip()

        if message.lower() == "exit":
            print("Goodbye!")
            break

        if not message:
            print("Please enter a non-empty message.\n")
            continue

        inputs = encode_text(message, vocabulary)

        with torch.no_grad():
            logit = model(inputs)
            probability = torch.sigmoid(logit).item()

        prediction = "SPAM" if probability >= 0.5 else "HAM"

        print(f"Prediction: {prediction}")
        print(f"Spam probability: {probability:.2%}\n")


if __name__ == "__main__":
    main()
