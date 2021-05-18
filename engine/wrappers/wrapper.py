"""
Module with the super class of the wrappers to use the trained MeDAL models.
"""
from abc import ABC, abstractmethod
from engine.utils.electra_loader import ADAM_DF


class Wrapper(ABC):  # pylint: disable=too-few-public-methods
    """
    Wrapper interface. Each subclass of Wrapper will adapt to a specific
    pre-trained model in the MeDAL paper.
    """
    @abstractmethod
    def __init__(self, note: str) -> None:
        self.note = note
        self.df_dictionary = ADAM_DF

    @abstractmethod
    def predict(self, tokens: list, locations: list) -> list:
        """
        Use the electra pre-trained model to predict the disambiguation
        given the location of the short form. Will return a list containing
        the long forms of the abbreviations.
        """
