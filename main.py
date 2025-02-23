#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import numpy as np

# ---------------------------------------------------------------
# Settings and Paths
# ---------------------------------------------------------------

KANJIDIC2_XML_PATH = 'kanjidic2.xml'  # Path to your XML file
OUTPUT_FILE = 'data/kanji_data.txt'  # Path to save extracted data
MODEL_PATH = 'kanji_generator.pth'  # Path to save the model


# ---------------------------------------------------------------
# Data Extraction from XML
# ---------------------------------------------------------------

def extract_kanji_data(xml_path, output_file):
    """Extract Kanji data from XML and save to a text file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    with open(output_file, 'w', encoding='utf-8') as f:
        for character in root.findall('.//character'):
            literal = character.find('literal').text
            if not literal:
                continue

            meanings = [m.text for m in character.findall('.//meaning')]
            readings = [r.text for r in character.findall('.//reading')]

            f.write(f"Character: {literal}\n")
            f.write(f"Meanings: {', '.join(meanings)}\n")
            f.write(f"Readings: {', '.join(readings)}\n")
            f.write("\n")


# ---------------------------------------------------------------
# Data Preprocessing
# ---------------------------------------------------------------

class KanjiDataset(Dataset):
    def __init__(self, data_file, max_length=100, vocab_size=5000):
        self.max_length = max_length
        self.words, self.vocab = self.build_vocab(data_file, vocab_size)
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for idx, word in enumerate(self.vocab)}
        self.tensor_data = self.process_data(data_file)

    def build_vocab(self, data_file, vocab_size):
        with open(data_file, 'r', encoding='utf-8') as f:
            data = f.read().splitlines()

        # Extract all words from the data
        words = [word for line in data for word in line.split()]

        # Count word frequencies
        word_counts = Counter(words)

        # Sort words by frequency and select the most common ones
        sorted_words = sorted(word_counts, key=lambda w: -word_counts[w])
        vocab = [w for w in sorted_words[:vocab_size] if word_counts[w] > 1]

        return words, vocab  # Now returns both words and vocab as a tuple

    def process_data(self, data_file):
        tensor_list = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokenized = []
                for word in line.split():
                    if word in self.word_to_idx:
                        tokenized.append(self.word_to_idx[word])
                    else:
                        tokenized.append(0)  # OOV token
                tensor = torch.tensor(tokenized)
                if len(tensor) > 0:
                    tensor_list.append(tensor)
        return tensor_list

    def __len__(self):
        return len(self.tensor_data)

    def __getitem__(self, idx):
        return self.tensor_data[idx]


def collate_fn(batch):
    """Pad sequences to the same length."""
    batch_sorted = sorted(batch, key=lambda x: len(x), reverse=True)
    padded = pad_sequence(batch_sorted, batch_first=True, padding_value=0)
    return padded


# ---------------------------------------------------------------
# Model Definition
# ---------------------------------------------------------------

class LSTMGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        return self.fc(output)


# ---------------------------------------------------------------
# Training Function
# ---------------------------------------------------------------

def train_model(dataset, model_path, epochs=10, batch_size=16, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMGenerator(len(dataset.vocab)).to(device)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in data_loader:
            batch = batch[:, :dataset.max_length].to(device)  # Truncate if needed
            optimizer.zero_grad()

            outputs = model(batch[:, :-1])
            targets = batch[:, 1:]

            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_loader)}")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


# ---------------------------------------------------------------
# Text Generation
# ---------------------------------------------------------------

def generate_text(model, dataset, seed_text="Character", num_words=100, temperature=0.8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    generated = []
    input_text = seed_text.split()[:100]
    input_tensor = torch.tensor([dataset.word_to_idx.get(w, 0) for w in input_text]).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(num_words):
            output = model(input_tensor[:, -100:])
            output = F.softmax(output[0, -1, :] / temperature, dim=-1)
            topk = torch.topk(output, 50)[1]
            word_idx = topk[torch.multinomial(torch.ones_like(topk).float(), 1).item()]
            generated.append(dataset.idx_to_word.get(word_idx.item(), "<UNK>"))
            input_tensor = torch.cat([input_tensor, torch.tensor([[word_idx]]).to(device)], dim=1)

    return ' '.join(generated)


# ---------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------

if __name__ == "__main__":
    # Extract data from XML
    extract_kanji_data(KANJIDIC2_XML_PATH, OUTPUT_FILE)

    # Prepare dataset
    dataset = KanjiDataset(OUTPUT_FILE)

    # Train model
    train_model(dataset, MODEL_PATH, epochs=10, batch_size=16, lr=0.001)

    # Load trained model
    model = LSTMGenerator(len(dataset.vocab))
    model.load_state_dict(torch.load(MODEL_PATH))

    # Generate sample text
    generated = generate_text(model, dataset, seed_text="Character 一 Meaning", num_words=50)
    print("Generated Text:")
    print(generated)