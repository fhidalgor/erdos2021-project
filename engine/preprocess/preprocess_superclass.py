"""
Module that contains the code to preprocess the pubmed dataset for usage
in the training of nlp models.
"""
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
import time
from multiprocessing import Pool
import pandas as pd

# Load short forms dataframe
ADAM_DF: pd.DataFrame = pd.read_csv("datasets/adam/valid_adam.txt", sep='\t')

# Number of processes in the multipool
PROCESSES = 10

# Path to the .txt files and output path
INPUT_PATH: str = ("datasets/pubmed/")
OUTPUT_PATH: str = ("datasets/pubmed/")


class Preprocess(ABC):  # pylint: disable=too-few-public-methods
    """
    Pubmed preprocessing abstract class. Each subclass of Preprocess will
    perform a different funtion in the preprocessing pipeline.
    """
    @abstractmethod
    def __init__(self):
        self.df_dictionary = ADAM_DF
        self.dictionary = OrderedDict(
            zip(self.df_dictionary.EXPANSION, self.df_dictionary.PREFERRED_AB)
        )
        self.processes = PROCESSES
        self.input_path: str = INPUT_PATH
        self.output_path: str = OUTPUT_PATH

    @abstractmethod
    def __call__(self):
        """
        When called, will run the batch_run.
        """
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    @abstractmethod
    def single_run(self, filename: str):
        """
        This method processes one file.
        """

    @abstractmethod
    def batch_run(self) -> None:
        """
        This method multiprocesses single_run.
        """
        # Get initial time
        start_time: float = time.time()

        # Extract the abstracts and save to csv in a multiprocess manner
        with Pool(processes=self.processes) as pool:
            pool.map(self.single_run, os.listdir(self.input_path))

        # Print the run time
        print("--- %s seconds ---" % (time.time() - start_time))
