"""
Module that tests the electra wrapper.
"""
import unittest
from engine.wrappers.electra_wrapper import ElectraWrapper


def dummy_note() -> str:  # pylint: disable=missing-function-docstring
    with open('tests/wrappers/dummy_note.txt') as line:
        note = line.readlines()
    return "".join(note)


class TestElectraWrapper(unittest.TestCase):
    """
    Test the wrapper for the electra pre-trained model.
    """
    def test_predict(self) -> None:  # pylint: disable=missing-function-docstring
        note = dummy_note()

        electra_wrapper = ElectraWrapper(note)
        tokens = electra_wrapper.tokenizer.encode(note, return_tensors="pt")
        locations: list = [10, 25]
        long_forms = electra_wrapper.predict(tokens, locations)
        long_forms_solution: list = [
            'acute myocardial infarction', 'transluminal coronary angioplasty'
        ]
        self.assertEqual(long_forms, long_forms_solution)
