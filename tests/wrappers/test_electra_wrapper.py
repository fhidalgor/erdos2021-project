import unittest
from engine.wrappers.electra_wrapper import ElectraWrapper


class TestElectraWrapper(unittest.TestCase):
    """
    Test the wrapper for the electra pre-trained model.
    """
    def test_predict(self) -> None:
        note = self.dummy_note()

        electra_wrapper = ElectraWrapper(note)
        tokens = electra_wrapper.tokenizer.encode(note, return_tensors="pt")
        locations: list = [10, 25]
        long_forms = electra_wrapper.predict(tokens, locations)
        long_forms_solution: list = [
            'acute myocardial infarction', 'transluminal coronary angioplasty'
        ]
        self.assertEqual(long_forms, long_forms_solution)

    def dummy_note(self) -> str:
        with open('tests/wrappers/dummy_note.txt') as line:
            note = line.readlines()
        return "".join(note)
