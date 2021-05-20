"""
Module that contains the code to replace the long forms by short forms
in the corpus.
"""
import os
from typing import List, Tuple, Dict
import random
import pandas as pd
import string
from collections import OrderedDict

from engine.utils.preprocessing import Preprocessor, delete_overlapping_tuples
from engine.preprocess.preprocess_superclass import Preprocess


class ReplaceLongForms(Preprocess):
    """
    When the instance of the class is executed, it will replace the
    long forms by short forms.
    You can define the probability of a substitution, and the min length
    of the abstracts.
    """
    def __init__(self, probability: float = 1, length_abstract: int = 200) -> None:
        super().__init__()
        self.input_path = self.input_path + str("identified_abstracts")
        self.output_path = self.output_path + str("replaced_abstracts")
        self.probability = probability
        self.preprocessor = Preprocessor(num_words_to_remove=50, remove_punctuation=False)
        self.length_abstract = length_abstract
        self.len_batch_key = 8
        self.long_form_counts: OrderedDict[str, int] = OrderedDict(dict.fromkeys(self.dictionary.keys(),0))
        self.long_form_loc: OrderedDict[str, list]  = OrderedDict({key: [] for key in self.dictionary.keys()})  
        # ^ Creating the dict with dict.fromkeys was appending to all keys, that is why I changed
        
    def __call__(self) -> None:
        """
        When the instance of the class is executed, it will replace the
        long forms by short forms.
        """
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        super().batch_run()

    def batch_run(self) -> None:
        """
        Empty because the super class method is used.
        """

    def decision(self) -> bool:
        """
        Return True/False based on a given probability.
        """
        return random.random() < self.probability

    def generate_key(self) -> str:
        """
        Function to generate a unique key for each abstract using the
        pubmed file name and the row.
        """
        return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(self.len_batch_key))
    
    def replace_abstract(
        self,
        abstract: str,
        long_forms: List[str],
        span: List[Tuple[int, int]],
    ) -> Tuple[str, List[str], List[Tuple[int, int]]]:
        """
        Given an abstract, it will replace the long forms by short forms.
        If the short form was already in the text, it wont add more short forms.
        Will return the replaced abstract and the list spans.
        """
        replaced_abstract: str = abstract
        span_updated: List[Tuple[int, int]] = []
        long_forms_updated: List[str] = []
        dict_span_lf = dict(zip(span, long_forms))

        # Deal with tuple overlapping
        clean_tuples = delete_overlapping_tuples(span)

        # Iterave over each long form and span
        correction_index: int = 0
        for tup in clean_tuples:
            if self.decision():
                long_form = dict_span_lf[tup]
                replaced_abstract = str(
                    replaced_abstract[: tup[0] + correction_index] + self.dictionary[long_form] +
                    replaced_abstract[tup[1] + correction_index :],
                )
                span_updated.append((
                    tup[0] + correction_index,
                    tup[0] + correction_index + len(self.dictionary[long_form])
                ))
                correction_index = correction_index + len(self.dictionary[long_form]
                                                          ) - len(long_form)
                long_forms_updated.append(long_form)

        return replaced_abstract, long_forms_updated, span_updated

    def single_run(self, filename: str) -> None:
        """
        Will load the csv file with the abstracts and replace the long
        forms by short forms. Will add a unique key to each output row.
        The key is generated using the filename.
        """
        # Open csv file with abstracts
        df_abstracts = pd.read_csv(
            os.path.join(self.input_path, filename), converters={'long_forms': eval, 'span': eval}
        )

        df_results: pd.DataFrame = pd.DataFrame(
            columns=['long_forms', 'span_short_form', 'replaced_abstract', 'unique_key']
        )
        batch_key : str = self.generate_key()
        
        for i, row in df_abstracts.iterrows():
            # check that the list is not empy and the length of the abstract.
            if row['long_forms'] != [] and len(row['abstract']) > self.length_abstract:
                # replace long forms. Need to convert span to tuples
                replaced_abstract, long_forms_updated, span_updated = self.replace_abstract(
                    row['abstract'], row['long_forms'], row['span']
                )
                # Store in dataframe if not empty
                if long_forms_updated != []:
                    # Define unique key
                    unique_key: str = batch_key + "_" + str(i)
                    # Update the dictionaries with the counts and keys
                    for long_form in long_forms_updated:
                        self.long_form_counts[long_form] += 1
                        self.long_form_loc[long_form].append(unique_key)
                    # Export to df
                    df_results = df_results.append({
                        'long_forms': long_forms_updated, 'span_short_form': span_updated,
                        'replaced_abstract': replaced_abstract, 'unique_key': unique_key
                    },
                                                   ignore_index=True)

        # Export df to csv
        new_filename: str = "{}.csv".format(batch_key)
        df_results.to_csv(os.path.join(self.output_path, new_filename), index = False)
        
        # Export dictionary to csv
        df_counts = pd.DataFrame(list(self.long_form_counts.items()),columns = ['long_form','counts'])
        df_counts['unique_key'] = list(self.long_form_loc.values())
        df_counts.to_csv(os.path.join(self.output_path, "counts.csv"), index = False)
