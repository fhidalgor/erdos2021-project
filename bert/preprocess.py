from os import truncate
import pandas as pd
import torch
from bert.model import ELECTRA_TOKENIZER, Electra
from icecream import ic
from itertools import compress 

# Tokenizes data and converts to tensor. 
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
        
        # Code to remove entries that are larger than the max length size
        batch_df = self.df.iloc[idxs]
        # ic(batch_df['TEXT'].apply(lambda string: len(string.split())))
        filter = batch_df['TEXT'].apply(lambda string: len(string.split()) < self.max_length).to_list()
        # ic(idxs, filter)
        idxs = list(compress(idxs, filter))


        batch_df = self.df.iloc[idxs]
        locs = batch_df['LOCATION'].values
        label_strings = batch_df['LABEL'].values
        labels = self.label_ser[label_strings].to_numpy()

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
    df = pd.read_csv("datasets/medal/one_abbr/train_one_abbr.csv")
    dictionary_file = "datasets/medal/one_abbr/dict.txt"
    label_df = pd.read_csv(dictionary_file, sep='\t', index_col = "EXPANSION")
    # batch_df = label_df.iloc[[0, 1]]['LABEL'].values
    # print(label_df.head(10))
    # print(label_df.loc['casein'])

    model = Electra(output_size=65)
    train_data = MedalDatasetTokenizer(df, ELECTRA_TOKENIZER, dictionary_file)
    idx = [0, 1]
    X = train_data[idx][0]
    loc = train_data[idx][1]
    y = train_data[idx][2]
    ic(model(X,loc), y)

if __name__ == "__main__":
    main()