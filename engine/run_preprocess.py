"""
Module to run the preprocessing pipeline
"""
import pandas as pd
from engine.preprocess.extract_pubmed import ExtractPubmedAbstracts
from engine.preprocess.extract_mimic import ExtractMimicNotes
from engine.preprocess.tokenize_text import Tokenize
from engine.preprocess.identify_longforms import IdentifyLongForms
from engine.preprocess.replace_longforms import ReplaceLongForms
from engine.preprocess.sample_dataset import SampleDataset

#obj = Tokenize('mimiciii')

# Load short forms dataframe
SUBSET_DF: pd.DataFrame = pd.read_csv("datasets/adam/train_2500_sort_AB_Exp.csv")

#obj = IdentifyLongForms('mimiciii', SUBSET_DF)

obj = ReplaceLongForms('mimiciii', SUBSET_DF)
obj()
#print(SUBSET_DF.columns)
