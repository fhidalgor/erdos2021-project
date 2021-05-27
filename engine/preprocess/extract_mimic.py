"""
This module will extract the notes from mimic NOTEEVENTS.csv file.
Will chunk it into smaller pieces to facilitate multiprocessing.
"""
import os
import sys
import numpy as np
import pandas as pd

from engine.preprocess.preprocess_superclass import Preprocess


class ExtractMimicNotes(Preprocess):
    """
    This class will extract the notes of the csv MIMICIII file and chunk
    them into smaller csv files.
    """
    def __init__(self, dataset: str = 'mimiciii') -> None:
        super().__init__(dataset)
        self.input_path = self.input_path + "csv_notes"
        self.output_path = self.output_path + "extracted"
        self.input_path_raw_file = os.path.join(self.input_path, "NOTEEVENTS.csv")
        
    def __call__(self) -> None:
        """
        When the instance of the class is executed, it will extract the
        abstracts of the csv file and chunk them.
        """

        self.chunk()
        
    def single_run(self) -> None:
        """
        Empty because the super class method is used.
        """
        
    def batch_run(self) -> None:
        """
        Empty because the super class method is used.
        """
    
    def get_lines_csv(self) -> int:
        """
        Get the number of lines in the input csv.
        """
        with open(self.input_path_raw_file) as f:
            for i, l in enumerate(f):
                pass
        return i
    
    def chunk(self) -> None:
        """
        Chunk notes in smaller sized csvs.
        """
        i: int = 0
        
        for chunk in pd.read_csv(self.input_path_raw_file, chunksize=int(3e5)):
            if sys.getsizeof(chunk) > 100e6:
                for j in np.arange(0,6):
                    mini_chunk = chunk[['TEXT']].iloc[int(j*5e4):int(j*5e4+5e4)]
                    mini_chunk.to_csv(os.path.join(self.output_path, str(i+j).zfill(2)+".csv"))
                i = i+j+1
            else:
                chunk[['TEXT']].to_csv(os.path.join(self.output_path, str(i).zfill(2)+".csv"))
                i = i + 1
