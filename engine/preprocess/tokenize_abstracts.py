"""
Module that contains the code to tokenize and remove digits that are not
part of words in the corpus and punctuation.
"""
import os

from engine.preprocess.preprocessing import Preprocessor
from engine.preprocess.preprocess_superclass import Preprocess


class Tokenize(Preprocess):
    """
    This class will tokenize, remove punctuation and digits that are not
    in contact with a non-digit character.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input_path = self.input_path + "extracted_abstracts"
        self.output_path = self.output_path + "tokenized_abstracts"
        self.preprocessor = Preprocessor(num_words_to_remove=-1, remove_punctuation=True)

    def __call__(self) -> None:
        """
        This class will tokenize, remove punctuation and digits that are not
        in contact with a non-digit character.
        """
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        super().batch_run()

    def tokenize_abstracts(self, filename: str) -> list:
        """
        Will load the txt file with the abstracts and tokenize them.
        """
        # Open file with abstracts
        with open(os.path.join(self.input_path, filename), 'r') as filehandle:
            abstracts = [abstract.rstrip() for abstract in filehandle.readlines()]

        # Use preprocessor to tokenize
        abstracts_tokenized = [self.preprocessor.preprocess(abstract) for abstract in abstracts]

        return abstracts_tokenized

    def single_run(self, filename: str) -> None:
        """
        Will take a list of tokenized abstracts and write it in a .txt file.
        """
        abstracts_tokenized: list = self.tokenize_abstracts(filename)
        new_filename: str = os.path.splitext(filename)[0] + "_nodigits.txt"
        with open(os.path.join(self.output_path, new_filename), 'w') as filehandle:
            filehandle.writelines("%s\n" % abstract for abstract in abstracts_tokenized)
