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
    
    size = len(train_data)

    # Switches model to training mode.
    model.train()

    # Loads indexes of training data
    train_loader = DataLoader(
        range(len(train_data)), 
        shuffle=True, 
        batch_size=batch_size
    )
    
    # List of all values for the loss. Output at the end.
    loss_list = []
    # For computing accuracy
    correct = 0

    for batch, idx in enumerate(tqdm(train_loader)):
    # for batch, idx in enumerate(train_loader):
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

        loss_value = loss.item()
        batch_correct = (pred.argmax(1) == y).type(torch.float).sum().item()
        correct += batch_correct
        loss_list.append(loss_value)

        if max > 0:
            if batch >= max:
                print("\nMax iterations reached.")
                size = max*batch_size
                break

        if batch % 5 == 0 and batch != 0:
            print(f"\nbatch loss: {loss_value:>7f} | batch accuracy: {batch_correct/batch_size:>7f}")
    
    loss_list = np.array(loss_list)
    accuracy = correct/size
    print(f"Accuracy: {accuracy} | Average Loss: {np.mean(loss_list):>7f}\n")
    return loss_list, accuracy

# Tests the model on the validation data
def valid_loop(valid_data, model, loss_fn, max = -1):

    # Switches model to evaluation mode
    model.eval()

    size = len(valid_data)
    valid_loss, correct = 0, 0

    with torch.no_grad():
        for id in tqdm(range(size)):
        # for id in range(size):
            idx = torch.tensor([id])
            # print(id, idx)
            X = valid_data[idx][0]
            loc = valid_data[idx][1]
            y = valid_data[idx][2]
            pred = model(X, loc)
            valid_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            if max > 0:
                if id > max:
                    break

    valid_loss /= size
    correct /= size
    print(f"Validation: \nAccuracy: {(100*correct):>0.1f}% | Average loss: {valid_loss:>8f} \n")
    return valid_loss, correct

# Save the model in its current state.
def save_model(model, save_dir):
    now = datetime.now()
    now_formatted = now.strftime("%d")+"_"+now.strftime("%H")+"_"+now.strftime("%M")
    torch.save(model, os.path.join(save_dir, f"{now_formatted}_two_abbr_Electra.pt"))
    print("Model saved\n")

def main():

    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu" 
    device = torch.device(dev) 

    # N_CPU_CORES = 2
    # torch.set_num_threads(N_CPU_CORES)
    
    tokenizer = ELECTRA_TOKENIZER
    max = -1

    # Data
    num_abbr = "two_abbr"
    folder = "datasets/medal"
    train_df = pd.read_csv(f"{folder}/{num_abbr}/train.csv")
    dictionary_file = f"{folder}/{num_abbr}/dict.txt"
    output_size = 25 # Should be set to the size of the dictionary
    train_data = MedalDatasetTokenizer(train_df, tokenizer, dictionary_file, device = device)

    valid_df = pd.read_csv(f"{folder}/{num_abbr}/valid.csv")
    valid_data = MedalDatasetTokenizer(valid_df, tokenizer, dictionary_file, device = device)

    # Hyperparameters
    learning_rate = 2e-5
    batch_size = 16
    epochs = 2
    
    # Model
    model = Electra(output_size=output_size, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    

    # Train the model
    for t in range(epochs):
        print(f"\nEpoch {t+1}\n-------------------------------")
        
        loss_array, train_accuracy = train_loop(train_data, model, loss_fn, optimizer, batch_size = batch_size,\
            max = max)
        train_loss = np.mean(loss_array)
        # print(f"\nEpoch {t+1} Finished \nAccuracy: {train_accuracy} | Average Loss: {train_loss:>7f}")
        
        valid_loss, valid_accuracy = valid_loop(valid_data, model, loss_fn, max = max)

        with open("saves/loss.txt", "a") as file:
            file.writelines(f"\n{t+1},{train_loss},{train_accuracy},{valid_loss},{valid_accuracy}")

        save_model(model, f"{folder}/saves")

    

    

if __name__ == "__main__":
    main()