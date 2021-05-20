"""
This module will sample a corpus containing short forms, so all short
forms appear with a given frequency
"""
import os
from typing import List, Tuple
import time
from multiprocessing import Pool, Manager
import pandas as pd

from engine.preprocess.preprocess_superclass import Preprocess


def divide_chunks(list_to_divide: list, size_chunks: int):
    """
    Chunk list into small pieces
    """
    # looping till length l
    for i in range(0, len(list_to_divide), size_chunks):
        yield list_to_divide[i : i + size_chunks]


class SampleDataset(Preprocess):
    """
    Select subset of dataset where the frequency of each long form is
    equal.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input_path: str = self.input_path + str("replaced_abstracts")
        self.output_path: str = self.output_path + str("sampled_abstracts")
        self.max_long_form_occurrences: int = 10

        # Create global variables
        self.replaced_dfs: dict = Manager().dict()

    def __call__(self) -> None:
        """
        When the instance of the class is executed, it select a subset
        of abstracts.
        """
        super().__call__()
        super().batch_run()
        self.batch_run_sample()

    def batch_run(self) -> None:
        """
        Empty because the super class method is used.
        """

    def get_pairs(self) -> Tuple[list, list]:
        """
        Return dataframe with long forms and keys of abstracts. If an
        abstract is in more than one long form, it will be kept for the
        long form with the smallest frequency.
        """
        df_counts = pd.read_csv(
            os.path.join(self.input_path, "counts.csv"),
            converters={'long_forms': eval, 'unique_key': eval}
        )
        df_counts = df_counts.sort_values(by='counts')
        df_counts = df_counts.explode('unique_key')
        df_counts.dropna(inplace=True)
        df_counts.drop_duplicates(subset='unique_key', inplace=True)
        temp = df_counts.groupby(by=['long_form'])['unique_key'].apply(list)
        long_forms: list = list(temp.index)
        keys: list = list(temp)
        return long_forms, keys

    def sample_data(self, list_tuples: List[tuple]) -> None:
        """
        Will return a dictionary with the counts of each long form.
        The input is a list of tuples containing the long form and the'
        keys.
        """
        df_results: pd.DataFrame = pd.DataFrame(
            columns=['long_forms', 'span_short_form', 'replaced_abstract', 'unique_key']
        )
        for long_form, keys_subset in list_tuples:
            for i, key in enumerate(keys_subset):
                # get out of loop if enough abstracts
                if i > self.max_long_form_occurrences:
                    break
                # look for the abstract and append to df with results
                row = self.replaced_dfs[key.split('_')[0]].loc[self.replaced_dfs[key.split('_')[0]]
                                                               ['unique_key'] == key]
                index_long_form = list(row['long_forms'])[0].index(long_form)
                span: Tuple[int, int] = list(row['span_short_form'])[0][index_long_form]
                df_results = df_results.append({
                    'long_forms':
                    long_form, 'span_short_form':
                    span, 'replaced_abstract':
                    list(row['replaced_abstract'])[0], 'unique_key':
                    key
                },
                                               ignore_index=True)

        # append to csv
        df_results.to_csv(
            os.path.join(self.output_path, "sampled_dataset.csv"),
            index=False,
            header=False,
            mode='a'
        )

    def single_run(self, filename: str) -> None:
        """
        Load csv to internal dictionary
        """

        if filename != 'counts.csv':
            df_replaced = pd.read_csv(
                os.path.join(self.input_path, filename),
                converters={'long_forms': eval, 'span_short_form': eval}
            )
            self.replaced_dfs[os.path.splitext(filename)[0]] = df_replaced

    def batch_run_sample(self) -> None:
        """
        Empty because the super class method is used.
        """
        # Get initial time
        start_time: float = time.time()

        # Get long forms/keys and chunk
        long_forms, keys = self.get_pairs()
        arguments: List[Tuple] = list(zip(long_forms, keys))
        chunked_arguments = list(divide_chunks(arguments, 100))

        # Create empty dictionary
        df_results: pd.DataFrame = pd.DataFrame(
            columns=['long_forms', 'span_short_form', 'replaced_abstract', 'unique_key']
        )
        df_results.to_csv(os.path.join(self.output_path, "sampled_dataset.csv"), index=False)

        # Extract the abstracts and save to csv in a multiprocess manner
        with Pool(processes=self.processes) as pool:
            pool.map(self.sample_data, chunked_arguments)

        # Print the run time
        print("--- %s seconds to sample ---" % (time.time() - start_time))
