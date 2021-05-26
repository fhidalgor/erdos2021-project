import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from bert.model import Electra, ELECTRA_TOKENIZER
from bert.model import Bert, BERT_TOKENIZER
from bert.preprocess import MedalDatasetTokenizer
from tqdm import tqdm
from datetime import datetime
from time import time

# Using the reduced dictionary 
# DICTIONARY: pd.DataFrame = pd.read_csv("datasets/adam/train_2500_sort_AB_Exp.txt", sep='\t')
# DICTIONARY: pd.DataFrame = pd.read_csv("datasets/adam/valid_adam.txt", sep='\t')

# One epoch of training.
# max is the maximum number of batchs. If max is negative, it runs all batches.
def train_loop(train_data, model, loss_fn, optimizer, train_loader, max = -1):
    
    size = len(train_data)

    # Switches model to training mode.
    model.train()
    
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

        # Record the loss and accuracy
        loss_value = loss.item()
        batch_correct = (pred.argmax(1) == y).type(torch.float).sum().item()
        correct += batch_correct
        loss_list.append(loss_value)

        # Terminate early for testing purposes
        if max > 0:
            if batch >= max:
                print("\nMax iterations reached.")
                break

        # Minibatch loss
        if batch % 20 == 0 and batch != 0:
            print(f"\nBatch loss: {loss_value:>7f}")
    
    loss_list = np.array(loss_list)
    mean_loss = np.mean(loss_list)
    accuracy = correct/size
    print(f"Accuracy: {accuracy:>3f} | Average Loss: {mean_loss:>7f}\n")
    return mean_loss, accuracy

# Tests the model on the validation data
def valid_loop(valid_data, model, loss_fn, valid_loader, max = -1):

    # Switches model to evaluation mode
    model.eval()

    size = len(valid_data)
    loss_list = [] 
    correct = 0

    with torch.no_grad():
        for batch, idx in tqdm(enumerate(valid_loader)):
        # for batch, idx in enumerate(valid_loader):
            # idx = torch.tensor([id])
            # print(id, idx)
            X = valid_data[idx][0]
            loc = valid_data[idx][1]
            y = valid_data[idx][2]
            pred = model(X, loc)
            loss_list.append(loss_fn(pred, y).item())
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            if max > 0:
                if batch > max:
                    break

    valid_loss = np.mean(np.array(loss_list))
    correct /= size
    print(f"=Validation= \nAccuracy: {correct:>3f} | Average loss: {valid_loss:>8f} \n")
    return valid_loss, correct


# Save the model's state_dict in its current state. 
# Saved file name records current time and epoch number
def save_model(model, save_dir):
    now = datetime.now()
    time_formatted = now.strftime("%d")+"_"+now.strftime("%H")+"_"+now.strftime("%M")
    torch.save(model.state_dict(), save_dir + f"_{time_formatted}_StateDict.pt")
    print("Model saved\n")


def main():

    # Use GPU if available
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu" 
    device = torch.device(dev) 
    
    # Maximum number of training batches. Used for debugging.
    max = 5

    # Hyperparameters
    learning_rate = 2e-5
    batch_size = 16
    epochs = 1

    ### Models
    ### The tokenizers are in fact all identical
    output_size = 25 # Should be set to the size of the dictionary

    tokenizer = BERT_TOKENIZER
    # model = Bert(output_size, device)

    # tokenizer = ELECTRA_TOKENIZER
    # tokenizer = ELECTRA_BASE_TOKENIZER
    model = Electra(output_size=output_size, device=device)

    ### Load a saved model. The correct model above must be initialized.
    path = ""
    model.load_state_dict(torch.load(path))

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Data
    num_abbr = "two_abbr"
    folder = "datasets/medal"
    train_df = pd.read_csv(f"{folder}/{num_abbr}/train.csv")
    dictionary_file = f"{folder}/{num_abbr}/dict.txt"
    train_data = MedalDatasetTokenizer(train_df, tokenizer, dictionary_file, device = device)

    valid_df = pd.read_csv(f"{folder}/{num_abbr}/valid.csv")
    valid_data = MedalDatasetTokenizer(valid_df, tokenizer, dictionary_file, device = device)

    # Train the model
    for t in range(epochs):
        print(f"\nEpoch {t+1}\n-------------------------------")
        
        train_loader = DataLoader(
            range(len(train_data)), 
            shuffle=True, 
            batch_size=batch_size
        )

        valid_loader = DataLoader(
            range(len(valid_data)), 
            shuffle=True, 
            batch_size=batch_size
        )

        start = time()
        train_loss, train_accuracy = train_loop(train_data, model, loss_fn, optimizer, train_loader, max = max)
        end = time()
        print(f"Training time: {end-start:>0.1f} sec\n")

        valid_loss, valid_accuracy = valid_loop(valid_data, model, loss_fn, valid_loader, max = max)

        with open(f"{folder}/saves/loss.txt", "a") as file:
            file.writelines(f"\n{t+1},{train_loss},{train_accuracy},{valid_loss},{valid_accuracy}")

        save_model(model, f"{folder}/saves/{num_abbr}_epoch{t+1}")

    
if __name__ == "__main__":
    main()