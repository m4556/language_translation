import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k  # torchtext torchtext==0.6
from torchtext.data import Field, BucketIterator
from torch.utils.tensorboard import SummaryWriter  
import spacy


spacy_ger = spacy.load("de_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")


def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")

english = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")


train_data, valid_data, test_data = Multi30k.splits(
                                    exts=(".de", ".en"), fields=(german, english))


german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)