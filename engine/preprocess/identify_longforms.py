"""
This Module hosts the class that will identify the long forms present in
the pubmed abstracts according to an input dictionary.
"""

import os
from typing import List, Tuple, Optional, Match
import re
import pandas as pd

from engine.utils.preprocessing import Preprocessor
from engine.preprocess.preprocess_superclass import Preprocess


class IdentifyLongForms(Preprocess):
    """
    This class will identify the long forms present in the dictionary in
    the pubmed abstracts.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input_path = self.input_path + "tokenized_abstracts"
        self.output_path = self.output_path + "identified_abstracts"

        self.preprocessor = Preprocessor(num_words_to_remove=50, remove_punctuation=False)

    def __call__(self) -> None:
        """
        When the instance of the class is executed, it will extract the
        abstracts of pubmed into a txt file.
        """
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        super().batch_run()

    def batch_run(self) -> None:
        """
        Empty because the super class method is used.
        """

    def indentify_long_forms(self, abstract: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Given an abstract, it will identify the long forms and their span.
        If that abbreviation is present already, it wont count it. This is
        done to avoid having the same abbrevation with different meanings
        in a training abstract.
        """
        # Get the long forms that appear in the abstract, only if the short form
        # is not present in text
        long_forms: List[str] = []
        spans: List[Tuple[int, int]] = []
        for long_form, short_form in self.dictionary.items():
            if long_form in abstract:  # Doing string search first speeds up the process
                if re.findall(r"\b%s\b" % re.escape(short_form), abstract,
                              re.IGNORECASE) == []:  # using re.escape to escape special characters
                    expression: Optional[Match[str]] = re.search(
                        r"\b%s\b" % re.escape(long_form),
                        abstract,
                    )
                    # Check that the long form was found in the text
                    if expression is not None:
                        long_forms.append(expression.group())
                        spans.append(expression.span())
        return long_forms, spans

    def single_run(self, filename: str) -> None:
        """
        Will export to csv the replaced abstracts, long forms and the span.
        """
        # Open file with abstracts
        with open(os.path.join(self.input_path, filename), 'r') as filehandle:
            abstracts = [abstract.rstrip() for abstract in filehandle.readlines()]

        # Loop over all abstracts
        df_results: pd.DataFrame = pd.DataFrame(columns=['long_forms', 'span', 'abstract'])
        for abstract in abstracts:
            # Identify long forms
            long_forms, spans = self.indentify_long_forms(abstract)
            # Save in df
            df_results = df_results.append({
                'long_forms': long_forms, 'span': spans, 'abstract': abstract
            },
                                           ignore_index=True)

        # Export to csv
        new_filename: str = os.path.splitext(filename)[0] + "_longforms.csv"
        df_results.to_csv(os.path.join(self.output_path, new_filename), index=False)
