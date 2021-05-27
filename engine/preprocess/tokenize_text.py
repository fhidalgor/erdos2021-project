"""
Module that contains the code to tokenize and remove digits that are not
part of words in the corpus and punctuation.
"""
import os
import pandas as pd

from engine.utils.preprocessing import Preprocessor
from engine.preprocess.preprocess_superclass import Preprocess


class Tokenize(Preprocess):
    """
    This class will tokenize, remove punctuation and digits that are not
    in contact with a non-digit character.
    """
    def __init__(self, dataset: str) -> None:
        super().__init__(dataset)
        self.input_path = self.input_path + "extracted"
        self.output_path = self.output_path + "tokenized"
        self.preprocessor = Preprocessor(num_words_to_remove=-1, remove_punctuation=True)

    def __call__(self) -> None:
        """
        This class will tokenize, remove punctuation and digits that are not
        in contact with a non-digit character.
        """
        super().__call__()
        super().batch_run()

    def batch_run(self) -> None:
        """
        Empty because the super class method is used.
        """

    def tokenize_abstracts(self, filename: str) -> list:
        """
        Will load the txt/csv file with the abstracts and tokenize them.
        """
        if self.dataset == 'pubmed':
            with open(os.path.join(self.input_path, filename), 'r') as filehandle:
                abstracts = [abstract.rstrip() for abstract in filehandle.readlines()]
        elif self.dataset == 'mimiciii':
            df_abstracts = pd.read_csv(os.path.join(self.input_path, filename))
            abstracts = df_abstracts['TEXT'].to_list()
            
        # Use preprocessor to tokenize
        abstracts_tokenized = [self.preprocessor.preprocess(abstract) for abstract in abstracts]

        return abstracts_tokenized

    def single_run(self, filename: str) -> None:
        """
        Will take a list of tokenized abstracts and write it in a .txt file.
        """
        abstracts_tokenized: list = self.tokenize_abstracts(filename)
        new_filename: str = os.path.splitext(filename)[0] + "_tokenized.txt"
        with open(os.path.join(self.output_path, new_filename), 'w') as filehandle:
            filehandle.writelines("%s\n" % abstract for abstract in abstracts_tokenized)
