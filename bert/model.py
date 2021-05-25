import torch
import pandas as pd
from torch import nn
from transformers import AutoModel, ElectraConfig
from transformers import ElectraTokenizer
from transformers import BertTokenizer, BertModel

ELECTRA_TOKENIZER = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
BERT_TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')

# ELECTRA model with custom output layer.
class Electra(nn.Module):
    def __init__(self, output_size, device='cpu'):
        super().__init__()
        self.device = device
        config = ElectraConfig.from_pretrained('google/electra-small-discriminator')
        self.model = AutoModel.from_config(config).to(device)
        self.output = nn.Linear(self.model.config.hidden_size, output_size).to(device)

    # What happens when passing input into the model.
    def forward(self, sents, locs):
        # sents = torch.tensor(sents).to(self.device)
        # print(self.electra(sents))
        sents = self.model(sents)[0]
        abbs = torch.stack([sents[n, idx, :] for n, idx in enumerate(locs)])  # (B * M)
        return self.output(abbs)

class Bert(nn.Module):
    def __init__(self, output_size, device='cpu'):
        super().__init__()
        self.device = device
        self.model = BertModel.from_pretrained('bert-base-uncased').to(device)
        self.output = nn.Linear(self.model.config.hidden_size, output_size).to(device)

    # What happens when passing input into the model.
    def forward(self, sents, locs):
        # sents = torch.tensor(sents).to(self.device)
        # print(self.electra(sents))
        sents = self.model(sents)[0]
        abbs = torch.stack([sents[n, idx, :] for n, idx in enumerate(locs)])  # (B * M)
        return self.output(abbs)


