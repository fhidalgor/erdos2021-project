"""
This module will parse and clean the dictionary containing the short forms.
"""
import re
from typing import Tuple, Dict
from bs4 import BeautifulSoup
import pandas as pd

INPUT_PATH = "datasets/elsevier/sample.xml"
OUTPUT_PATH = "datasets/elsevier/sample.csv"


def parse_elsevier(export: bool = False) -> Tuple[Dict[str, str], pd.DataFrame]:
    """
    This function will parse the xml elsevier short forms dictionary.
    The output will be a dictionary and a dataframe containing the
    dictionary.
    """
    # Read xml file
    with open(INPUT_PATH, "r") as file:
        contents = file.read()

    # Parse with soup
    soup = BeautifulSoup(contents, 'xml')

    # Loop through entries and populate the dictionary
    dictionary: dict = {}
    for entry in soup.find_all('entry'):
        for long_form in list(entry.defgroup.children):
            short_form: str = list(entry)[1].text.strip('\n')
            try:
                if long_form.text != ";":
                    long_form_clean = re.sub(r'\[[^)]*\]', '', long_form.text.strip('\n'))
                    long_form_clean = re.sub(r'\([^)]*\)', '', long_form_clean)
                    dictionary[long_form_clean.strip('\n').strip(' ')] = short_form
            except AttributeError:
                pass

    # Convert to dataframe
    df_dictionary = pd.DataFrame(list(dictionary.items()), columns=['long_form', 'short_form'])

    # Export to csv
    if export:
        df_dictionary.to_csv(OUTPUT_PATH)

    return dictionary, df_dictionary
