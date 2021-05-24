import pandas as pd
import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from bert.model import Electra, ELECTRA_TOKENIZER
from bert.preprocess import MedalDatasetTokenizer
from tqdm import tqdm
from datetime import datetime

# Using the reduced dictionary 
# DICTIONARY: pd.DataFrame = pd.read_csv("datasets/adam/train_2500_sort_AB_Exp.txt", sep='\t')
# DICTIONARY: pd.DataFrame = pd.read_csv("datasets/adam/valid_adam.txt", sep='\t')

# One epoch of training.
# max is the maximum number of batchs. If max is negative, it runs all batches.
def train_loop(train_data, model, loss_fn, optimizer, batch_size, max = -1):
    
    # Switches model to training mode.
    model.train()
    
    # size = len(train_data)

    # List of all values for the loss. Output at the end.
    loss_list = []

    # Loads indexes of training data
    train_loader = DataLoader(
        range(len(train_data)), 
        shuffle=True, 
        batch_size=batch_size
    )
    

    for batch, idx in enumerate(tqdm(train_loader)):
        # print(idx)
        X = train_data[idx][0]
        loc = train_data[idx][1]
        y = train_data[idx][2]

        # Compute prediction and loss
        pred = model(X, loc)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if max > 0:
            if batch >= max:
                print("Max iterations reached.")
                break

        if batch % 2 == 0:
            loss = loss.item()
            print(f"\nloss: {loss:>7f}")
            loss_list.append(loss)

    return loss_list

# Tests the model on the validation data
def valid_loop(valid_data, model, loss_fn):

    # Switches model to evaluation mode
    model.eval()

    size = len(valid_data)
    valid_loss, correct = 0, 0

    with torch.no_grad():
        for id in tqdm(range(size)):
            idx = torch.tensor([id])
            print(id, idx)
            X = valid_data[idx][0]
            loc = valid_data[idx][1]
            y = valid_data[idx][2]
            pred = model(X, loc)
            valid_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    valid_loss /= size
    correct /= size
    print(f"Validation: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {valid_loss:>8f} \n")

def main():

    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu" 
    device = torch.device(dev) 

    # N_CPU_CORES = 2
    # torch.set_num_threads(N_CPU_CORES)
    
    tokenizer = ELECTRA_TOKENIZER

    # Data
    train_df = pd.read_csv("datasets/medal/one_abbr/train.csv")
    dictionary_file = "datasets/medal/one_abbr/dict.txt"
    output_size = 12 # Should be set to the size of the dictionary
    train_data = MedalDatasetTokenizer(train_df, tokenizer, dictionary_file)

    valid_df = pd.read_csv("datasets/medal/one_abbr/valid.csv")
    valid_data = MedalDatasetTokenizer(valid_df, tokenizer, dictionary_file)

    # Hyperparameters
    learning_rate = 2e-5
    batch_size = 16
    epochs = 1
    
    # Model
    model = Electra(output_size=output_size, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    

    # Train the model
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        loss = train_loop(train_data, model, loss_fn, optimizer, batch_size = batch_size,\
            max = 4, device = device)

        # # Testing whether the loss function changes
        # sample_size = 20
        # first100 = np.array(loss[:sample_size])
        # last100 = np.array(loss[-sample_size:])
        # print("Initial", np.mean(first100), np.std(first100))
        # print("Final", np.mean(last100), np.std(last100))
        loss = np.array(loss)
        print(f"\nEpoch {t} | Mean Loss: {np.mean(loss)} | Std: {np.std(loss)}")
        
        # valid_loop(valid_data, model, loss_fn)

    # Save the model in its current state.
    save_dir = "saves"
    now = datetime.now()
    now_formatted = now.strftime("%d")+"_"+now.strftime("%H")+"_"+now.strftime("%M")
    torch.save(model, os.path.join(save_dir, f"{now_formatted}_one_abbr_model.pt"))
    print("Model saved")

    

if __name__ == "__main__":
    main()