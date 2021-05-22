import pandas as pd
from transformers import ElectraTokenizer
import torch
# from itertools import compress

ELECTRA_TOKENIZER = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')

# Tokenizes data and converts to tensor. 
class MedalDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length=512, device='cpu'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.df = df
        label_df = pd.read_csv("datasets/adam/label_numbers.txt", sep='\t', index_col = "EXPANSION")
        self.label_num_series = label_df.squeeze()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idxs):
        # Line of the original Medal code that I think is unnecessary and time consuming
        # idxs = list(compress(idxs, batch_df['TEXT'].apply(lambda string: len(string.split()) < self.max_length).to_list()))
        
        batch_df = self.df.iloc[idxs]
        locs = batch_df['LOCATION'].values

        label_strings = batch_df['LABEL'].values
        labels = self.label_num_series[label_strings].to_numpy()
        # labels = labels.to(self.device)
        tokenized = self.tokenizer.batch_encode_plus(batch_df['TEXT'].tolist(), max_length=self.max_length, \
                    padding=True)['input_ids']
        return torch.tensor(tokenized), torch.tensor(locs), torch.tensor(labels)

def main():
    df = pd.read_csv("datasets/medal/test1000.csv")
    print(df.head())
    data = MedalDataset(df, ELECTRA_TOKENIZER)
    ids = [0, 1, 2, 3]
    for i in range(4, 1000, 4):
        data[list(range(i-4,i))]
        print(i)
    print("Done")
if __name__ == "__main__":
    main()