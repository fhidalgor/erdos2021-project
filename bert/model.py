import torch
from torch import nn
from transformers import ElectraModel
from transformers import ElectraTokenizer
from transformers import BertTokenizer, BertModel

ELECTRA_TOKENIZER = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
ELECTRA_BASE_TOKENIZER = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
BERT_TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')

class Electra(nn.Module):
    def __init__(self, output_size, size = 'small', device='cpu'):
        super().__init__()
        self.device = device
        self.model = ElectraModel.from_pretrained(f'google/electra-{size}-discriminator').to(device)
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


