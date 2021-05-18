"""
Module that tests the preprocessing utils module.
"""
import unittest
from typing import List, Tuple
from engine.utils.preprocessing import delete_overlapping_tuples


class TestPreprocessing(unittest.TestCase):
    """
    Test the delete overlapping tuples function.
    """
    def test_delete_overlapping_tuples(self) -> None:  # pylint: disable=missing-function-docstring
        tuple_list: List[Tuple[int, int]] = [(0, 5), (6, 10), (8, 15), (14, 20)]
        tuple_solution: List[Tuple[int, int]] = [(0, 5), (6, 10), (14, 20)]
        self.assertEqual(tuple_solution, delete_overlapping_tuples(tuple_list))
