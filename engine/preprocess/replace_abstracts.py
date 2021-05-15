import os
import time
from multiprocessing import Pool
from typing import List, Tuple, Optional
import random
from re import finditer
import pandas as pd
from engine.preprocess.preprocessing import Preprocessor
import csv

# Path to the .txt files and output path
INPUT_PATH: str = ("datasets/pubmed/tokenized_abstracts/")
OUTPUT_PATH: str = ("datasets/pubmed/replaced_abstracts/")

# Load short forms dataframe
ADAM_DF: pd.DataFrame = pd.read_csv("datasets/adam/valid_adam.txt", sep='\t')

# Number of processes in the multipool
PROCESSES = 11


class TokenizePubmedAbstracts:
    """
    This class will tokenize the abstracts of the .txt pubmed files.
    Will remove punctuation, stop words, digits that are not in contact with
    a non-digit character.
    You can define the probability of a substitution, and the min length of
    the abstracts.
    """
    def __init__(self, probability: float, length_abstract: int) -> None:
        self.input_path = INPUT_PATH
        self.output_path = OUTPUT_PATH
        self.df_dictionary = ADAM_DF
        self.dictionary = dict(zip(self.df_dictionary.EXPANSION, self.df_dictionary.PREFERRED_AB))
        self.probability = probability
        self.preprocessor = Preprocessor(num_words_to_remove=50, remove_punctuation=False)
        self.length_abstract = length_abstract
        
    def __call__(self) -> None:
        """
        When the instance of the class is executed, it will extract the
        abstracts of pubmed into a txt file.
        """
        self.batch_run()

    def decision(self) -> bool:
        """
        Return True/False based on a given probability.
        """
        return random.random() < self.probability

    def replace_long_forms(self, abstract: str) -> Tuple[str, List[int]]:
        """
        Given an abstract, it will replace the long forms by short forms.
        If the short form was already in the text, it wont add more short forms.
        Will return the replaced abstract and the list of acronyms.
        """
        # Get the long forms that appear in the abstract, only if the short form is not present
        matches: list = [
            long_form for long_form in self.dictionary.keys() if (" "+long_form.strip()+" ") in abstract
            if self.dictionary[long_form] not in abstract
        ]
        replaced_abstract: str = abstract
        short_forms: List[str] = []

        # For each long form present in the abstract, replace by short form.
        # The replacement is done based on the given probability.
        for long_form in matches:
            if self.decision():
                replaced_abstract = replaced_abstract.replace(
                    " "+long_form.strip()+" ", " "+self.dictionary[long_form]+" ", 1
                )
                short_forms.append(self.dictionary[long_form])
        return short_forms, replaced_abstract

    def replace_abstracts(self, filename: str) -> Optional[Tuple[list, list]]:
        """
        Will load the txt file with the abstracts and replace the long
        forms by short forms.
        """
        # Open file with abstracts
        with open(os.path.join(self.input_path, filename), 'r') as filehandle:
            abstracts = [abstract.rstrip() for abstract in filehandle.readlines()]

        spans: list = []
        replaced_abstracts: list = []
        for abstract in abstracts:
            short_forms, replaced_abstract = self.replace_long_forms(abstract)
    
            if short_forms == [] or len(abstract) < self.length_abstract:
                pass
            else:
                # Eliminate stop words
                replaced_abstract = self.preprocessor.remove_stop_words(replaced_abstract)
                # Save span
                span: list = []
                for short_form in short_forms:
                    try:
                        span.append(list(finditer(short_form, replaced_abstract))[0].span())
                    except: IndexError # This is caused by the dictionary being shitty.
                    
                # Store in list
                replaced_abstracts.append(replaced_abstract)
                spans.append(list(set(span)))
        return replaced_abstracts, spans

    def replace_save(self, filename: str) -> None:
        """
        Will export the replaced abstracts and the span of the short forms.
        """
        replaced_abstract, span = self.replace_abstracts(filename)
        new_filename: str = os.path.splitext(filename)[0] + "_replaced.csv"
        with open(os.path.join(self.output_path, new_filename), 'w') as filehandle:
            csv_writer = csv.writer(filehandle, delimiter=',')
            csv_writer.writerow(['Abstract', 'Span'])
            for abstract, span in zip(replaced_abstract, span):
                csv_writer.writerow([abstract, span])

    def batch_run(self) -> None:
        """
        This function multiprocesses replace_save.
        """
        # Get initial time
        start_time: float = time.time()

        # Extract the abstracts and save to csv in a multiprocess manner
        with Pool(processes=PROCESSES) as pool:
            pool.map(self.replace_save, os.listdir(self.input_path))

        # Print the run time
        print("--- %s seconds ---" % (time.time() - start_time))
        
obj = TokenizePubmedAbstracts(1, 200)
obj.replace_save("pubmed21n1195_tokenized.txt")
