from engine.wrappers.electra_wrapper import ElectraWrapper

with open('tests/wrappers/dummy_note.txt') as line:
    note = line.readlines()
NOTE = "".join(note)


def main() -> None:
    """
    Calls the model wrapper and executes it.
    """
    # Initialize model component
    electra_obj = ElectraWrapper(NOTE)

    # Execute method to get note substituted
    note_replaced = electra_obj()

    print(note_replaced)


if __name__ == "__main__":
    main()
