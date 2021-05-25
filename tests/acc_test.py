# from engine.wrappers.electra_wrapper import ElectraWrapper
from engine.utils.electra_loader import ELECTRA_TOKENIZER
from engine.utils.electra_loader import ELECTRA
from engine.utils.electra_loader import ADAM_DF
import torch
import pandas as pd
import time
import numpy as np
from tqdm import tqdm

def main():

    SAVE_DATA = False
    

    valid_pd = pd.read_csv("datasets/medal/two_abbr/valid.csv")
    X_valid = valid_pd['TEXT']
    loc = valid_pd['LOCATION']
    y_valid = valid_pd['LABEL']
    # print(X_valid.head())
    # print(X_valid[0])

    NUMBER_SAMPLES = X_valid.size

    start = time.time()

    tokenizer = ELECTRA_TOKENIZER
    electra_model = ELECTRA
    acc_array = np.zeros(NUMBER_SAMPLES)

    if SAVE_DATA:
        string_output = ""

    for i in tqdm(range(NUMBER_SAMPLES)):
        note = X_valid[i]
        location = loc[i]

        # Tokenize text with electra tokenizer
        tokens: list = tokenizer.encode(note, return_tensors="pt")

        # Switches electra into evaluation mode.
        electra_model.eval()

        # Predict the long forms of the short forms
        with torch.no_grad():

            output = electra_model(tokens, torch.tensor([location]))
            prediction = torch.argmax(output)
            long_form_pred = ADAM_DF['EXPANSION'].iloc[prediction.numpy()]
        
        correct = long_form_pred == y_valid[i]
        acc_array[i] = correct 
        if SAVE_DATA:
            string_output += f"{i}).\n"
            string_output += "Abbr = " + note.split()[location] + "\n"
            string_output += "Prediction = " + long_form_pred + "\n"
            string_output += "Actual = " + y_valid[i] + "\n"
            string_output += "Predict = Actual?: " + f"{correct}" + "\n\n"

    end = time.time()
    print(f"Time: {end-start}")

    acc_score = np.sum(acc_array)/acc_array.size
    print("Accuracy Score =", acc_score)

    if SAVE_DATA:
        string_output += "Accuracy Score = " + str(acc_score)


    if SAVE_DATA:
        with open("sample_output.txt", "w") as text_file:
            text_file.write(string_output)

if __name__ == "__main__":
    main()