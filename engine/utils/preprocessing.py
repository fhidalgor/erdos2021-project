"""
Module that contains text preprocessing util functions.
"""
import re
from typing import List, Tuple
from wordfreq import top_n_list, tokenize  # type: ignore

TOKENS_REG = re.compile(r"(?u)\b\w+\b")


def simple_preprocess(text: str) -> str:
    """
    Strips punctuation and returns lower case.
    """
    return " ".join(TOKENS_REG.findall(text.lower()))


def delete_overlapping_tuples(list_tuples: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Given a list of tuples, delete the tuples that overlap.
    Will first sort the tuples and keeps the first tuple.
    """
    # Sort tuples
    sorted_tuples = sorted(list_tuples, key=lambda x: x[0])

    # Create list with first tuple element
    clean_tuples: List[Tuple[int, int]] = [sorted_tuples[0]]
    # Iterate over tuples and sequentially compare with the i tuple
    if len(list_tuples) > 1:
        i: int = 0
        for tup in sorted_tuples[1 :]:
            if (tup[1] >= clean_tuples[i][0] and tup[0] <= clean_tuples[i][1]):
                pass
            else:
                clean_tuples.append(tup)
                i = i + 1
    return clean_tuples


class Preprocessor:
    """
    This class is responsible to preprocess sentences.
    Set num_words_to_remove = -1 if you do not want to remove stop words.
    """
    def __init__(self, num_words_to_remove: int = -1, remove_punctuation: bool = True) -> None:
        self.num_words_to_remove = num_words_to_remove
        self.stopwords = set(top_n_list("en", self.num_words_to_remove, wordlist='best'))
        self.remove_punctuation = remove_punctuation

    def preprocess(self, sentence: str) -> str:
        """
        Remove punctuation, digits and stop words.
        """
        if self.remove_punctuation:
            sentence = simple_preprocess(sentence)
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
            tokens = [word for word in tokenize(sentence, "en") if not word.isdigit()]
        return " ".join(tokens)

    def remove_stop_words(self, sentence: str) -> str:
        """
        Remove stop words.
        """
        tokens: list = [word for word in sentence.split() if word not in self.stopwords]
        return " ".join(tokens)
