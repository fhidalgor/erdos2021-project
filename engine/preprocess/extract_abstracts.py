import xml.etree.ElementTree as ET
import os
import time
from multiprocessing import Pool

# Path to the raw xml files and output path for the txt files
INPUT_PATH: str = ("datasets/pubmed/xml_abstracts/")
OUTPUT_PATH: str = ("datasets/pubmed/extracted_abstracts/")

# Number of processes in the multipool
PROCESSES = 11


class ExtractPubmedAbstracts:
    """
    This class will extract the abstracts of the raw .xml pubmed files
    and return .txt files.
    """
    def __init__(self) -> None:
        self.input_path = INPUT_PATH
        self.output_path = OUTPUT_PATH

    def __call__(self) -> None:
        """
        When the instance of the class is executed, it will extract the
        abstracts of pubmed into a txt file.
        """
        self.batch_run()

    def extract_abstracts(self, filename: str) -> list:
        """
        This function will take an .xml file and extract the pubmed abstracts.
        """
        tree = ET.parse(os.path.join(self.input_path, filename))
        root = tree.getroot()
        abstracts: list = []

        # Iterate over the children of the root and check their tag
        for child in root.iter():
            if child.tag == 'Abstract':
                if child[0].text and child[0].text != "\n":
                    abstracts.append(" ".join(child[0].text.split()))
        return abstracts

    def extract_save(self, filename: str) -> None:
        """
        Will take a list of abstracts and write it in a .txt file.
        Performance wise it was 2.5x faster to save the abstracts on a list and
        then export the list than combine the two steps.
        """
        abstracts: list = self.extract_abstracts(filename)
        new_filename: str = os.path.splitext(filename)[0] + ".txt"
        with open(os.path.join(self.output_path, new_filename), 'w') as filehandle:
            filehandle.writelines("%s\n" % abstract for abstract in abstracts)

    def batch_run(self) -> None:
        """
        This function multiprocesses extract_save.
        """
        # Get initial time
        start_time: float = time.time()

        # Extract the abstracts and save to txt in a multiprocess manner
        with Pool(processes=PROCESSES) as pool:
            pool.map(self.extract_save, os.listdir(self.input_path))

        # Print the run time
        print("--- %s seconds ---" % (time.time() - start_time))
