import torch
import pandas as pd
from torch import nn
from transformers import AutoModel, ElectraConfig
from transformers import ElectraTokenizer

ELECTRA_TOKENIZER = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')

# ELECTRA model with custom output layer.
class Electra(nn.Module):
    def __init__(self, output_size=4131 , device='cpu'):
        super().__init__()
        self.device = device
        config = ElectraConfig.from_pretrained('google/electra-small-discriminator')
        self.electra = AutoModel.from_config(config).to(device)
        self.output = nn.Linear(self.electra.config.hidden_size, output_size).to(device)

    # What happens when passing input into the model.
    def forward(self, sents, locs):
        # sents = torch.tensor(sents).to(self.device)
        # print(self.electra(sents))
        sents = self.electra(sents)[0]
        abbs = torch.stack([sents[n, idx, :] for n, idx in enumerate(locs)])  # (B * M)
        return self.output(abbs)

