import re
from typing import Iterable, List
from wordfreq import top_n_list, tokenize # type: ignore

TOKENS_REG = re.compile(r"(?u)\b\w+\b")


class Preprocessor:
    """
    This class is responsible to preprocess sentences.
    """
    def __init__(self, num_words_to_remove: int = 0, remove_punctuation: bool = True) -> None:
        self.stopwords = set(top_n_list("en", num_words_to_remove, wordlist='best'))
        self.remove_punctuation = remove_punctuation
        
    def simple_preprocess(self, text: str) -> str:
        """
        Strips punctuation and returns lower case.
        """
        return " ".join(TOKENS_REG.findall(text.lower()))
        
    def preprocess_sentences(self, sentences: Iterable[str]) -> List[str]:
        return [self.preprocess(sent) for sent in sentences]

    def preprocess(self, sentence: str) -> str:
        if self.remove_punctuation:
            sentence = self.simple_preprocess(sentence)
        else:
            sentence = sentence.lower()
        # remove stop words and digits
        tokens: list = [
            word for word in tokenize(sentence, "en") if word not in self.stopwords
            if not word.isdigit()
        ]
        return " ".join(tokens)
