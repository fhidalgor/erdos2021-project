import torch
import os, pickle
from engine.utils.text_processing import locate_short_forms, replace_short_forms
from engine.utils.LSTM_loader import ADAM_DF
from engine.utils.LSTM_loader import LSTM, LSTM_SA, FastTextTokenizer, EmbeddingsDataset, load_dataframes
from engine.wrappers.wrapper import Wrapper


class LSTMWrapper(Wrapper):
    """
    Wrapper for both of the LSTM pretrained models. It takes an input note, and
    it returns the note with the the short forms replaced by the long forms.
    """
    def __init__(self, note: str, addOns: str) -> None:
        self.note = note
        self.df_dictionary = ADAM_DF
        if addOns == 'SA':
            self.model = LSTM_SA
        else:
            self.model = LSTM

    def __call__(self) -> str:
        """
        When executed, will call an instance of this class. NOT EDITED FOR LSTM MODELS
        """
        # Find if token in the abbreviation list, store token and location
        _, span, locations = locate_short_forms(
            self.note, self.df_dictionary['PREFERRED_AB'].to_list()
        )

        # Tokenize text with electra tokenizer
        tokens: list = self.tokenizer.encode(self.note, return_tensors="pt")

        # Predict the long forms of the short forms
        long_forms = self.predict(tokens, locations)

        # Replace the short forms by long forms in the original text
        note_replaced = replace_short_forms(self.note, long_forms, span)

        return note_replaced

    def test_MeDAL_csv(index : int, data_dir, adam_path, emb_path, model='rnn' pretrained=False):
        # Load data
        tokenizer = get_LSTM_tokenizer(data_dir, adam_path, emb_path);

        DEVICE = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

        train = pd.read_csv(os.path.join(data_dir, 'train.csv'), engine='c')
        adam_df = pd.read_csv(adam_path, sep='\t')
        unique_labels = adam_df.EXPANSION.unique()
        label_to_ix = {label: ix for ix, label in enumerate(unique_labels)}
        train['LABEL_NUM'] = train.LABEL.apply(lambda l: label_to_ix[l]);

        train_data = EmbeddingsDataset(train, tokenizer=tokenizer, device=DEVICE);
        #valid_data = EmbeddingsDataset(valid, tokenizer=tokenizer, device=DEVICE);
        #test_data = EmbeddingsDataset(test, tokenizer=tokenizer, device=DEVICE);
        if pretrained == True:
            from utils import load_LSTM_pretrained;
            if model == 'rnn':
                net = load_LSTM_pretrained.lstm()
            elif model == 'rnn_sa':
                net = load_LSTM_pretrained.lstm_sa();
        else:
            if model == 'rnn':
                net = LSTM;
            elif model == 'rnn_sa':
                net = LSTM_SA;
        with torch.no_grad():
            idx = torch.tensor([index]);
            sents, locs, labels = train_data[idx];
            outputs = net(sents, locs);

        pLabels = torch.topk(outputs,20);
        print('Predicted values: ')
        print(adam_df['EXPANSION'].iloc[pLabels[1].numpy()[0]])

        print('Actual: ')
        print(adam_df['EXPANSION'].iloc[labels.numpy()])
        print('Actual: ')
        print(train["LABEL"][index])

        return pLabels, labels;

    def predict(self, tokens: list, locations: list) -> list:
        """
        Use the electra pre-trained model to predict the disambiguation
        given the location of the short form. Will return a list containing
        the long forms of the abbreviations.
        """

        long_forms: list = []
        short_forms: list = []
        # Set the evaluation mode to throw sentences to disambiguate
        self.model.eval()
        with torch.no_grad():  # Speeds up computation
            # You can only input one location of the acronyms at a time
            for location in locations:
                output = self.model(tokens, torch.tensor([location]))
                prediction = torch.argmax(output)

                # trace back using adam dictionary and append to list
                short_form = self.df_dictionary['PREFERRED_AB'].iloc[prediction.numpy()]
                long_form = self.df_dictionary['EXPANSION'].iloc[prediction.numpy()]
                short_forms.append(short_form)
                long_forms.append(long_form)

        return long_forms
