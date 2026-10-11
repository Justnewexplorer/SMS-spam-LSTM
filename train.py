
import os
import re
import random
import urllib.request
import zipfile

from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# ---------------- CONFIGURATION ----------------

SEED = 42
MAX_VOCAB_SIZE = 10000
MAX_LENGTH = 100
EMBEDDING_DIM = 64
HIDDEN_DIM = 64
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/"
    "228/sms+spam+collection.zip"
)

random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")


# ---------------- DOWNLOAD DATASET ----------------

def download_dataset():
    DATA_DIR.mkdir(exist_ok=True)

    dataset_file = DATA_DIR / "SMSSpamCollection"
    zip_file = DATA_DIR / "sms_spam_collection.zip"

    if dataset_file.exists():
        print("Dataset already downloaded.")
        return dataset_file

    print("Downloading SMS Spam Collection dataset...")

    urllib.request.urlretrieve(DATASET_URL, zip_file)

    print("Extracting dataset...")

    with zipfile.ZipFile(zip_file, "r") as archive:
        archive.extractall(DATA_DIR)

    zip_file.unlink(missing_ok=True)

    if not dataset_file.exists():
        raise FileNotFoundError(
            "SMSSpamCollection was not found after extraction."
        )

    return dataset_file


# ---------------- TEXT PREPROCESSING ----------------

def tokenize(text):
    text = text.lower()
    return re.findall(r"[a-z]+|[0-9]+", text)


def load_messages(dataset_file):
    messages = []
    labels = []

    with open(dataset_file, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.rstrip("\n").split("\t", 1)

            if len(parts) != 2:
                continue

            label, message = parts

            messages.append(message)
            labels.append(1 if label == "spam" else 0)

    print(f"Total messages: {len(messages)}")
    print(f"Spam messages: {sum(labels)}")
    print(f"Ham messages: {len(labels) - sum(labels)}")

    return messages, labels


# ---------------- VOCABULARY ----------------

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def build_vocabulary(texts):
    counts = Counter()

    for text in texts:
        counts.update(tokenize(text))

    vocabulary = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for word, _ in counts.most_common(MAX_VOCAB_SIZE - 2):
        vocabulary[word] = len(vocabulary)

    return vocabulary


def encode_text(text, vocabulary):
    tokens = tokenize(text)
    ids = [
        vocabulary.get(token, vocabulary[UNK_TOKEN])
        for token in tokens
    ]

    ids = ids[:MAX_LENGTH]

    if len(ids) < MAX_LENGTH:
        ids += [vocabulary[PAD_TOKEN]] * (
            MAX_LENGTH - len(ids)
        )

    return ids


# ---------------- PYTORCH DATASET ----------------

class SMSData(Dataset):
    def __init__(self, texts, labels, vocabulary):
        self.texts = [
            encode_text(text, vocabulary)
            for text in texts
        ]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            torch.tensor(self.texts[index], dtype=torch.long),
            torch.tensor(self.labels[index], dtype=torch.float32),
        )


# ---------------- LSTM MODEL ----------------

class SpamLSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            EMBEDDING_DIM,
            padding_idx=0,
        )

        self.lstm = nn.LSTM(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_DIM,
            batch_first=True,
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

        final_hidden = hidden[-1]
        output = self.fc(self.dropout(final_hidden))

        return output.squeeze(1)

# ---------------- TRAINING ----------------

def main():
    print("\nSMS SPAM DETECTION USING LSTM")
    print("=" * 45)
    print("Device:", DEVICE)

    dataset_file = download_dataset()
    messages, labels = load_messages(dataset_file)

    (
        train_texts,
        test_texts,
        train_labels,
        test_labels,
    ) = train_test_split(
        messages,
        labels,
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )

    vocabulary = build_vocabulary(train_texts)

    train_dataset = SMSData(
        train_texts, train_labels, vocabulary
    )

    test_dataset = SMSData(
        test_texts, test_labels, vocabulary
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = SpamLSTM(len(vocabulary)).to(DEVICE)

    spam_weight = len(train_labels) / (2 * sum(train_labels))

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [spam_weight], dtype=torch.float32
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print(f"\nTraining messages: {len(train_dataset)}")
    print(f"Testing messages: {len(test_dataset)}")
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            logits = model(inputs)
            loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * targets.size(0)

            predictions = (logits >= 0).float()
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

        average_loss = total_loss / total
        accuracy = correct / total

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"Training Accuracy: {accuracy:.4f}"
        )

    # ---------------- EVALUATION ----------------

    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            logits = model(inputs)
            predictions = (logits >= 0).long()

            all_predictions.extend(predictions.tolist())
            all_targets.extend(targets.long().tolist())

    print("\nTEST RESULTS")
    print("=" * 45)

    print(
        classification_report(
            all_targets,
            all_predictions,
            labels=[0, 1],
            target_names=["Ham", "Spam"],
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(all_targets, all_predictions, labels=[0, 1]))

    # ---------------- SAVE MODEL ----------------

    RESULTS_DIR.mkdir(exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocabulary": vocabulary,
        "config": {
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "max_length": MAX_LENGTH,
        },
    }

    model_path = RESULTS_DIR / "spam_lstm.pt"
    torch.save(checkpoint, model_path)

    print(f"\nModel saved to: {model_path.resolve()}")
    print("Training and evaluation completed!")


if __name__ == "__main__":
    main()
