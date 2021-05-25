import argparse
import os
import time
import pandas as pd

import torch
import torch.optim as optim
from torch import nn

from models.rnn import RNN
from models.lstm_sa import RNNAtt
from models.electra import Electra
from transformers import ElectraTokenizer
from utils import load_dataframes, load_model, train_loop
from models.tokenizer_and_dataset import \
    FastTextTokenizer, EmbeddingsDataset, HuggingfaceDataset
