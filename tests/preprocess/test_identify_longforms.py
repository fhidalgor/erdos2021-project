"""
Module that tests the identify long forms in a corpus.
"""
import unittest
from typing import List, Tuple

from engine.preprocess.identify_longforms import IdentifyLongForms

DUMMY_OBJECT = IdentifyLongForms()


class TestIdentifyLongForms(unittest.TestCase):
    """
    Test the replace long forms class.
    """
    def test_indentify_long_forms(self) -> None:  # pylint: disable=missing-function-docstring
        # Dummy data
        abstract: str = str("The kid had an autism diagnostic interview with the doctor because "\
                        "had a congenital nephrotic syndrome However, the autismplexus "\
                        "blockdiagnostic interview suffered from congenital-nephrotic-syndrome.")
        long_forms_solution: List[str] = [
            "autism diagnostic interview", "congenital nephrotic syndrome"
        ]
        span_solution: List[Tuple[int, int]] = [(15, 42), (73, 102)]

        # Calculated answer
        long_forms, span = DUMMY_OBJECT.indentify_long_forms(abstract)

        # Assertion
        self.assertEqual(long_forms, long_forms_solution)
        self.assertEqual(span, span_solution)
