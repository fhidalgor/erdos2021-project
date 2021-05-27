"""
Module with the sub class of the wrappers to use on Electra.
"""
import torch
from engine.utils.wrapper_utils import locate_short_forms, replace_short_forms
from engine.utils.electra_loader import ELECTRA
from engine.utils.electra_loader import ELECTRA_TOKENIZER
from engine.wrappers.wrapper import Wrapper


class ElectraWrapper(Wrapper):
    """
    Wrapper for the electra pretrained model. It takes an input note, and
    it returns the note with the the short forms replaced by the long forms.
    """
    def __init__(self, note: str) -> None:
        super().__init__(note)
        self.tokenizer = ELECTRA_TOKENIZER
        self.model = ELECTRA

    def __call__(self) -> str:
        """
        When executed, will call an instance of this class.
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
