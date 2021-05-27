"""
Module with utils used in the preprocess module.
"""
from typing import Tuple, List


def divide_chunks(list_to_divide: list, size_chunks: int):
    """
    Chunk list into small pieces.
    """
    # looping till length l
    for i in range(0, len(list_to_divide), size_chunks):
        yield list_to_divide[i : i + size_chunks]


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
