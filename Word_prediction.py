import torch
import re
import demoji
import random
import inflect
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

import torchtext
import torch.nn.functional as F
from torch.nn.functional import one_hot
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

# Suppress torchtext deprecation warnings
import torchtext; torchtext.disable_torchtext_deprecation_warning()

# Device checking
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('my device', device)

# Loading data
path = "D:\\conversation.txt"
with open(path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
print(lines[0:2])

# Pre-processing
def preprocess_text(text):
    # Removing HTML tags
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()

    p = inflect.engine()  # Converting num to text
    # Removing emojis
    text = demoji.replace(text, "")

    # Removing 'Human 1' & 'Human 2'
    text = re.sub(r'\b(?:Human 1|Human 2)\b:?', " ", text)

    # Replacing num with words
    text = re.sub(r'\b\d+\b', lambda x: p.number_to_words(x.group()), text)

    # Removing special character and keeping alphabets and space
    text = re.sub('[^a-zA-Z\s]', ' ', text)

    # Replacing specific unicode spaces with standard spaces and trim
    text = text.replace(u'\xe0', u' ').replace('\u200a', ' ').strip()

    return text

preprocessed_lines = [preprocess_text(line) for line in lines]
print(preprocessed_lines[0:5])

tokenizer = get_tokenizer('basic_english')
tokenized_conv = [tokenizer(conv) for conv in preprocessed_lines]
print(tokenized_conv[2])

# Build vocabulary
features_vocab = build_vocab_from_iterator(
    tokenized_conv,
    min_freq=1,
    specials=['<pad>', '<oov>'],
    special_first=True
)
target_vocab = build_vocab_from_iterator(
    tokenized_conv,
    min_freq=1
)

features_vocab_total_words = len(features_vocab)
target_vocab_total_words = len(target_vocab)

print('Total vocabs in feature vocab :', features_vocab_total_words)
print('Total vocabs in target vocab :', target_vocab_total_words)

# Making ngrams from the conversations
def make_ngrams(tokenized_text):
    list_ngrams = []
    for i in range(1, len(tokenized_text)):
        ngram_sequence = tokenized_text[:i+1]
        list_ngrams.append(ngram_sequence)
    return list_ngrams

ngrams_list = []
for tokenized_con in tokenized_conv:
    ngrams_list.extend(make_ngrams(tokenized_con))

# Adding random oov token to let the model handle oov tokens
def add_random_oov_tokens(ngram):
    for idx, word in enumerate(ngram[:-1]):
        if random.uniform(0, 1) < 0.1:
            ngram[idx] = '<oov>'
    return ngram

ngrams_list_oov = []
for ngram in ngrams_list:
    ngrams_list_oov.append(add_random_oov_tokens(ngram))
print(any('<oov>' in ngram for ngram in ngrams_list_oov))

def text_to_numerical_sequence(tokenized_text):
    tokens_list = []
    if tokenized_text[-1] in target_vocab.get_itos():
        for token in tokenized_text[:-1]:
            num_token = features_vocab[token] if token in features_vocab.get_itos() else features_vocab['<oov>']
            tokens_list.append(num_token)
        num_token = target_vocab[tokenized_text[-1]]
        tokens_list.append(num_token)
        return tokens_list
    return None

input_sequences = [text_to_numerical_sequence(sequence) for sequence in ngrams_list_oov if text_to_numerical_sequence(sequence)]

print(f'total input sequence : {len(input_sequences)} ')
print(input_sequences[7:9])

x = [sequence[:-1] for sequence in input_sequences]
y = [sequence[-1] for sequence in input_sequences]
print(x[2], y[2])

longest_sequence_feature = max(len(sequence) for sequence in x)

padded_x = [F.pad(torch.tensor(sequence), (longest_sequence_feature - len(sequence), 0), value=0) for sequence in x]

padded_x = torch.stack(padded_x)
y = torch.tensor(y)
print(type(y))
print(type(padded_x))

y_one_hot = one_hot(y, num_classes=target_vocab_total_words)

data = TensorDataset(padded_x, y_one_hot)

train_size = int(0.8 * len(data))
test_size = len(data) - train_size
batch_size = 16

train_data, test_data = random_split(data, [train_size, test_size])
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

class MyModel(nn.Module):

    def __init__(self, features_vocab_total_words, target_vocab_total_words, embedding_dim, hidden_dim):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(features_vocab_total_words, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, target_vocab_total_words)

    def forward(self, x):
        x = x.to(self.embedding.weight.device)
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(torch.cat((lstm_out[:, -1, :hidden_dim], lstm_out[:, 0, hidden_dim:]), dim=1))
        return output

features_vocab_total_words = 2749
target_vocab_total_words = 2747
embedding_dim = 128
hidden_dim = 200
epochs = 50

model = MyModel(features_vocab_total_words, target_vocab_total_words, embedding_dim=embedding_dim, hidden_dim=hidden_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.007)

def calculate_accuracy(model, data_loader, k=3):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            output = model(batch_x)
            _, predicted_indices = output.topk(k, dim=1)
            correct_predictions += torch.any(predicted_indices == torch.argmax(batch_y, dim=1, keepdim=True), dim=1).sum().item()
            total_predictions += batch_y.size(0)

    accuracy = correct_predictions / total_predictions
    return accuracy

all_accuracy = []
all_losses = []
for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y.argmax(dim=1))
        loss.backward()
        optimizer.step()

    if epoch % 5 == 0:
        accuracy = calculate_accuracy(model, train_loader)
        print(f'Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, Train K_Accuracy: {accuracy * 100:.2f}%')
        all_accuracy.append(accuracy)
        all_losses.append(loss.item())
