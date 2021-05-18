"""
Module that tests the text processing functions used in the electra
wrapper.
"""
import unittest
from engine.utils.text_processing import locate_short_forms, replace_short_forms
from engine.utils.electra_loader import ADAM_DF


def dummy_note() -> str:  # pylint: disable=missing-function-docstring
    with open('tests/utils/dummy_note.txt') as line:
        note = line.readlines()
    return "".join(note)


def dummy_note_replaced() -> str:  # pylint: disable=missing-function-docstring
    with open('tests/utils/dummy_note_replaced.txt') as line:
        note = line.readlines()
    return "".join(note)


class TestTextProcessing(unittest.TestCase):
    """
    Test the utils text processing functions.
    """
    def locate_short_forms(self) -> None:  # pylint: disable=missing-function-docstring
        note = dummy_note()
        df_dictionary = ADAM_DF
        short_forms_intext, span, locations = locate_short_forms(
            note, df_dictionary['PREFERRED_AB'].to_list()
        )

        self.assertEqual(short_forms_intext, ['AIM', 'TCA'])
        self.assertEqual(span, [(91, 94), (196, 199)])
        self.assertEqual(locations, [10, 25])

    def test_replace_short_forms(self) -> None:  # pylint: disable=missing-function-docstring
        note = dummy_note()
        note_replaced_solution = dummy_note_replaced()
        span: list = [(91, 94), (196, 199)]
        long_forms: list = ['acute myocardial infarction', 'transluminal coronary angioplasty']
        note_replaced = replace_short_forms(note, long_forms, span)
        self.assertEqual(note_replaced, note_replaced_solution)
