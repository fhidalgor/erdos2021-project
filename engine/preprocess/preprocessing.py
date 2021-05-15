import re
from typing import Iterable, List
from wordfreq import top_n_list, tokenize  # type: ignore

TOKENS_REG = re.compile(r"(?u)\b\w+\b")


class Preprocessor:
    """
    This class is responsible to preprocess sentences.
    Set num_words_to_remove = -1 if you do not want to remove stop words.
    """
    def __init__(self, num_words_to_remove: int = -1, remove_punctuation: bool = True) -> None:
        self.num_words_to_remove = num_words_to_remove
        self.stopwords = set(top_n_list("en", self.num_words_to_remove, wordlist='best'))
        self.remove_punctuation = remove_punctuation

    def simple_preprocess(self, text: str) -> str:
        """
        Strips punctuation and returns lower case.
        """
        return " ".join(TOKENS_REG.findall(text.lower()))

    def preprocess_sentences(self, sentences: Iterable[str]) -> List[str]:
        return [self.preprocess(sent) for sent in sentences]

    def preprocess(self, sentence: str) -> str:
        """
        Remove punctuation, digits and stop words.
        """
        if self.remove_punctuation:
            sentence = self.simple_preprocess(sentence)
        else:
            sentence = sentence.lower()

        if self.num_words_to_remove != -1:
            # remove stop words and digits
            tokens: list = [
                word for word in tokenize(sentence, "en") if word not in self.stopwords
                if not word.isdigit()
            ]
        else:
            # remot stop digits only
            tokens: list = [word for word in tokenize(sentence, "en") if not word.isdigit()]
        return " ".join(tokens)

    def remove_stop_words(self, sentence: str) -> str:
        """
        Remove stop words.
        """
        tokens: list = [word for word in sentence.split() if word not in self.stopwords]
        return " ".join(tokens)
