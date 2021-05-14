from engine.preprocess.preprocessing import Preprocessor
import os
import time
from multiprocessing import Pool

# Path to the .txt files and output path
INPUT_PATH: str = ("datasets/pubmed/extracted_abstracts/")
OUTPUT_PATH: str = ("datasets/pubmed/clean_abstracts/")

# Number of processes in the multipool
PROCESSES = 11


class TokenizePubmedAbstracts:
    """
    This class will tokenize the abstracts of the .txt pubmed files.
    Will remove punctuation, stop words, digits that are not in contact with
    a non-digit character.
    """
    def __init__(self) -> None:
        self.input_path = INPUT_PATH
        self.output_path = OUTPUT_PATH
        self.preprocessor = Preprocessor(num_words_to_remove=50, remove_punctuation=True)

    def __call__(self) -> None:
        """
        When the instance of the class is executed, it will extract the
        abstracts of pubmed into a txt file.
        """
        self.batch_tokenize_abstracts()

    def tokenize_abstracts(self, filename: str) -> list:
        """
        Will load the txt file with the abstracts and tokeniz them.
        """
        # Open file with abstracts
        with open(os.path.join(self.input_path, filename), 'r') as filehandle:
            abstracts = [abstract.rstrip() for abstract in filehandle.readlines()]

        # Use preprocessor to tokenize
        abstracts_tokenized = [self.preprocessor.preprocess(abstract) for abstract in abstracts]

        return abstracts_tokenized

    def tokenize_save(self, filename: str) -> None:
        """
        Will take a list of tokenized abstracts and write it in a .txt file.
        """
        abstracts_tokenized: list = self.tokenize_abstracts(filename)
        new_filename: str = os.path.splitext(filename)[0] + "_tokenized.txt"
        with open(os.path.join(self.output_path, new_filename), 'w') as filehandle:
            filehandle.writelines("%s\n" % abstract for abstract in abstracts_tokenized)

    def batch_tokenize_abstracts(self) -> None:
        """
        This function multiprocesses extract_save.
        """
        # Get initial time
        start_time: float = time.time()

        # Extract the abstracts and save to txt in a multiprocess manner
        with Pool(processes=PROCESSES) as pool:
            pool.map(self.tokenize_save, os.listdir(self.input_path))

        # Print the run time
        print("--- %s seconds ---" % (time.time() - start_time))
