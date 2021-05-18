"""
Module that loads the electra model and the electra tokenizer.
"""
import pandas as pd
from torch.hub import load
from transformers import ElectraTokenizer

# Load short forms dataframe
ADAM_DF: pd.DataFrame = pd.read_csv("datasets/adam/valid_adam.txt", sep='\t')

# Load electra MeDAL pretrained model
ELECTRA = load("BruceWen120/medal", "electra")

# Create electra tokenizer object
ELECTRA_TOKENIZER = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
