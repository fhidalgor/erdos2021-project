from os import truncate
import pandas as pd
import torch
from bert.model import ELECTRA_TOKENIZER, Electra
from icecream import ic
from itertools import compress 

# Tokenizes data and converts to tensor. 
class MedalDatasetTokenizer(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length=256, device='cpu'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.df = df
        label_df = pd.read_csv("datasets/adam/label_numbers.txt", sep='\t', index_col = "EXPANSION")
        self.label_num_series = label_df.squeeze()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idxs):
        
        # Code to remove entries that are larger than the max length size
        batch_df = self.df.iloc[idxs]
        # ic(batch_df['TEXT'].apply(lambda string: len(string.split())))
        filter = batch_df['TEXT'].apply(lambda string: len(string.split()) < self.max_length).to_list()
        # ic(idxs, filter)
        idxs = list(compress(idxs, filter))


        batch_df = self.df.iloc[idxs]
        locs = batch_df['LOCATION'].values
        label_strings = batch_df['LABEL'].values
        labels = self.label_num_series[label_strings].to_numpy()

        # ic(batch_df['TEXT'].tolist())
        # ic(type(batch_df['TEXT'].tolist()[0]))
        batch_encode = self.tokenizer(batch_df['TEXT'].tolist(), max_length=self.max_length, \
                    padding=True, truncation = True)
        
        # ic(batch_encode)
        # ic(type(batch_encode))

        tokenized = batch_encode['input_ids']
        # decoded = self.tokenizer.batch_decode(tokenized)
        # ic(decoded, len(decoded[0].split()))
        # ic(len(tokenized[0]), len(tokenized[1]), type(tokenized))
        return torch.tensor(tokenized), torch.tensor(locs), torch.tensor(labels)

def main():
    df = pd.read_csv("datasets/medal/test1000.csv")
    data = MedalDatasetTokenizer(df, ELECTRA_TOKENIZER)
    ids = [6, 7]
    ids = [7]
    # ic(data[ids][0].size())

    

if __name__ == "__main__":
    main()