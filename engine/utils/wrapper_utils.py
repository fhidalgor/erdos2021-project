"""
Module with some text processing functions used by the wrappers.
"""
import re
from typing import Tuple
from wordfreq import top_n_list

TOKENS_REG = re.compile(r"(?u)\b\w+\b")
STOP_WORDS = set(top_n_list("en", 50, wordlist='best'))  # 50 most common stop words


def locate_short_forms(note: str, short_form_list: list) -> Tuple[list, list, list]:
    """
    Find if token in the short forms list, store token, span and location.
    If the token happen to be a short form and a stop word, it won't get counted.
    """
    locations: list = []
    short_forms_intext: list = []
    span: list = []
    for i, token in enumerate(TOKENS_REG.finditer(note)):
        if token.group() in short_form_list and token.group() not in STOP_WORDS:
            locations.append(i)
            short_forms_intext.append(token.group())
            span.append(token.span())

    return short_forms_intext, span, locations


def replace_short_forms(note: str, long_forms: list, span: list) -> str:
    """
    Given a list of long forms and the span of the short forms, replace
    the short form by long form in note using the string indeces.
    """
    note_replaced: str = note
    # Iterates in reverse order, otherwise we would have to change the span indeces
    for long_form, index_span in zip(long_forms[::-1], span[::-1]):
        note_replaced = note_replaced[: index_span[0]] + long_form + note_replaced[index_span[1]:]

    return note_replaced
