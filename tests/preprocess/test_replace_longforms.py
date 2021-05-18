"""
Module that tests the replace long forms by short forms of a corpus.
"""
import unittest
from typing import List, Tuple

from engine.preprocess.replace_longforms import ReplaceLongForms

DUMMY_OBJECT = ReplaceLongForms(probability=1, length_abstract=50)


class TestReplaceLongForms(unittest.TestCase):
    """
    Test the replace long forms class.
    """
    def test_replace_abstract(self) -> None:  # pylint: disable=missing-function-docstring
        # Dummy data
        abstract: str = str("The kid had an autism diagnostic interview with the doctor because "\
                        "had a congenital nephrotic syndrome However, the autismplexus "\
                        "blockdiagnostic interview suffered from congenital-nephrotic-syndrome.")
        long_forms: List[str] = ["autism diagnostic interview", "congenital nephrotic syndrome"]
        span: List[Tuple[int, int]] = [(15, 42), (73, 102)]

        # Expected solutions
        replaced_abstract_solution: str = str("The kid had an ADI with the doctor because "\
                                                "had a CNS However, the autismplexus blockdiagnostic interview "\
                                                "suffered from congenital-nephrotic-syndrome.")
        span_solution: List[Tuple[int, int]] = [(15, 18), (49, 52)]

        # Calculated answer
        replaced_abstract, long_forms_updated, span_updated = DUMMY_OBJECT.replace_abstract(
            abstract, long_forms, span
        )

        # Assertion
        self.assertEqual(replaced_abstract, replaced_abstract_solution)
        self.assertEqual(long_forms_updated, long_forms)
        self.assertEqual(span_updated, span_solution)
