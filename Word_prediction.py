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


#device checking

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('my device', device)

#loading data

path = "D:\conversation.txt"
with open(path, 'r',  encoding='utf-8') as file:
    lines = file.readlines()
print(lines[0:2])

#pre=processing

def preprocess_text(text):
    #removing html tags
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()

    p = inflect.engine() #converting num to text
    demoji.download_codes()
    #removeing emojis
    text = demoji.replace(text, "")

    #removing 'human1' & 'human2'
    text = re.sub(r'\b(?:Human 1|Human 2)\b:?', " ", text)

    #replacing num with words
    text = re.sub(r'\b\d+\b', lambda x: p.number_to_words(x.group()), text)

    #removing special character and keeping alphabets and space
    text = re.sub('[^a-zA-Z\s]', ' ', text)

    #replacing specific unicode spaces with standard spaces and trim
    text = text.replace(u'\xe0', u' ').replace('\u200a', ' ').strip()

    return text

preprocessed_lines = [preprocess_text(line) for line in lines]
print(preprocessed_lines[0:5])

tokenizer = get_tokenizer('basic_english')
tokenized_conv = [tokenizer(conv) for conv in preprocessed_lines]
print(tokenized_conv[2])

'''build_vocab_from_iterator function in the torchtext.vocab module is used to create a vocabulary from an iterable of 
tokenized data.This vocabulary is essential for converting textual data into numerical form'''

features_vocab = torchtext.vocab.build_vocab_from_iterator(
    tokenized_conv,
    min_freq=1,
    specials=['<pad>', '<oov>'], #oov= out of vocabulary, if model gets any words which is not known then it will use oov for that word
    special_first=True
)
target_vocab = torchtext.vocab.build_vocab_from_iterator(
    tokenized_conv,
    min_freq=1
)

features_vocab_total_words = len(features_vocab)
target_vocab_total_words = len(target_vocab)

print('Total vocabs in feature vocab :', features_vocab_total_words)
print('Total vocabs in target vocab :', target_vocab_total_words)

#making ngrams from the conversations
def make_ngrams(tokenized_text):
    list_ngrams =[]
    for i in range(1, len(tokenized_text)):
        ngram_sequence = tokenized_text[:i+1]
        list_ngrams.append(ngram_sequence)
    return list_ngrams