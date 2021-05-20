from engine.preprocess.extract_abstracts import ExtractPubmedAbstracts
from engine.preprocess.identify_longforms import IdentifyLongForms
from engine.preprocess.tokenize_abstracts import Tokenize
from engine.preprocess.replace_longforms import ReplaceLongForms

obj = ReplaceLongForms()
obj()
