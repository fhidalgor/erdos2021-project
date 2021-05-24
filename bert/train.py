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
DICTIONARY: pd.DataFrame = pd.read_csv("datasets/adam/train_2500_sort_AB_Exp.txt", sep='\t')

## Below is the original dictionary.
# DICTIONARY: pd.DataFrame = pd.read_csv("datasets/adam/valid_adam.txt", sep='\t')

# One epoch of training.
def train_loop(train_data, model, loss_fn, optimizer, batch_size = 1, max = -1):

    size = len(train_data)

    # List of all values for the loss. Output at the end.
    loss_list = []

    # Loads indexes of training data
    train_loader = DataLoader(
        range(len(train_data)), 
        shuffle=True, 
        batch_size=batch_size
    )
    

    for batch, idx in enumerate(tqdm(train_loader)):

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
            loss, current = loss.item(), batch * len(X)
            print(f"\nloss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            loss_list.append(loss)

    return loss_list

# I haven't looked at the code from the test_loop yet. It shouldn't work.
def test_loop(dataloader, model, loss_fn):
    size = dataloader.size
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, loc, y in dataloader:
            pred = model(X, loc)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main():

    N_CPU_CORES = 2
    torch.set_num_threads(N_CPU_CORES)

    # Hyperparameters
    learning_rate = 1e-3
    batch_size = 16
    epochs = 1
    
    # Model
    model = Electra()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Data
    df = pd.read_csv("datasets/medal/test10000.csv")
    train_data = MedalDatasetTokenizer(df, ELECTRA_TOKENIZER)

    # Train the model
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        loss = train_loop(train_data, model, loss_fn, optimizer, batch_size = batch_size, max = -1)

        # Testing whether the loss function changes
        sample_size = 100
        first100 = np.array(loss[:sample_size])
        last100 = np.array(loss[-sample_size:])
        print("Initial", np.mean(first100), np.std(first100))
        print("Final", np.mean(last100), np.std(last100))
        # test_loop(test_dataloader, model, loss_fn)

    # Save the model in its current state.
    save_dir = "bert/saves"
    now = datetime.now()
    now_formatted = now.strftime("%d")+"_"+now.strftime("%H")+"_"+now.strftime("%M")
    torch.save(model, os.path.join(save_dir, f"{now_formatted}_10000_model.pt"))
    print("Model saved")

    

if __name__ == "__main__":
    main()