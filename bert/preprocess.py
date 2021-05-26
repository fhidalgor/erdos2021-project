import pandas as pd
import torch
import numpy as np
from bert.model import ELECTRA_TOKENIZER, BERT_TOKENIZER, ELECTRA_BASE_TOKENIZER, Electra, GlossBert
from icecream import ic
from itertools import compress 


# Tokenizes data and converts to tensor. 
# The data is be preprocessed to remove files that are longer than the max_length.
# Otherwise there will be some errors down the road. 

class MedalDatasetTokenizer(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer, dictionary_file, max_length=256, device='cpu'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.df = df
        label_df = pd.read_csv(dictionary_file, sep='\t', index_col = "EXPANSION")
        self.label_ser = label_df["LABEL"].squeeze()


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idxs):
        
        batch_df = self.df.iloc[idxs]
        locs = batch_df['LOCATION'].values
        label_strings = batch_df['LABEL'].values
        labels = self.label_ser[label_strings].to_numpy()

        batch_encode = self.tokenizer(batch_df['TEXT'].tolist(), max_length=self.max_length, \
                    padding=True, truncation = True)

        tokenized = batch_encode['input_ids']
        return torch.tensor(tokenized).to(self.device), torch.tensor(locs).to(self.device), \
            torch.tensor(labels).to(self.device)

class GlossPreprocessing(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, dictionary_file, max_length=256, device='cpu'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.df = df
        label_df = pd.read_csv(dictionary_file, sep='\t', index_col = "EXPANSION")
        self.label_ser = label_df["LABEL"].squeeze()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idxs):
        
        batch_df = self.df.iloc[idxs]
        locs = batch_df['LOCATION'].values
        label_strings = batch_df['LABEL'].values
        labels = self.label_ser[label_strings].to_numpy()

        batch_encode = self.tokenizer(batch_df['TEXT'].tolist(), max_length=self.max_length, \
                    padding=True, truncation = True)

        tokenized = batch_encode['input_ids']
        return torch.tensor(tokenized).to(self.device), torch.tensor(locs).to(self.device), \
            torch.tensor(labels).to(self.device)

def main():
    df = pd.read_csv("datasets/medal/two_abbr/train.csv")
    dictionary_file = "datasets/medal/two_abbr/dict.txt"
    label_df = pd.read_csv(dictionary_file, sep='\t', index_col = "EXPANSION")
    tokenizer = BERT_TOKENIZER
    expansions = label_df.index
    length_list = []
    for long_form in expansions:
        tokens = tokenizer.tokenize(long_form)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        ids.append(102)
        ic(ids)
        tokenized = tokenizer(long_form)['input_ids']
        ic(tokenized)
        length_list.append(len(tokenized))
    ic(length_list, max(length_list))

    with torch.no_grad():
        # train_data = MedalDatasetTokenizer(df, ELECTRA_TOKENIZER, dictionary_file)
        # train_data_electra_base = MedalDatasetTokenizer(df, ELECTRA_BASE_TOKENIZER, dictionary_file)
        # train_data = MedalDatasetTokenizer(df, BERT_TOKENIZER, dictionary_file)
        # idx = [0, 1, 2]
        # idy = [3, 4, 5]
        # X_1 = train_data[idx][0]
        # X_2 = train_data[idy][0]
        # loc1 = train_data[idx][1]
        # loc2 = train_data[idy][1]
        
        pass

        

if __name__ == "__main__":
    main()