"""
This Module hosts the class that will extract the abstracts of the raw
.xml pubmed files and return .txt files.
"""
import xml.etree.ElementTree as ET
import os

from engine.preprocess.preprocess_superclass import Preprocess


class ExtractPubmedAbstracts(Preprocess):
    """
    This class will extract the abstracts of the raw .xml pubmed files
    and return .txt files.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input_path = self.input_path + "xml_abstracts"
        self.output_path = self.output_path + "extracted_abstracts"

    def __call__(self) -> None:
        """
        When the instance of the class is executed, it will extract the
        abstracts of pubmed into a txt file.
        """
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        super().batch_run()

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

    def single_run(self, filename: str) -> None:
        """
        Will take a list of abstracts and write it in a .txt file.
        Performance wise it was 2.5x faster to save the abstracts on a list and
        then export the list than combine the two steps.
        """
        abstracts: list = self.extract_abstracts(filename)
        new_filename: str = os.path.splitext(filename)[0] + ".txt"
        with open(os.path.join(self.output_path, new_filename), 'w') as filehandle:
            filehandle.writelines("%s\n" % abstract for abstract in abstracts)
