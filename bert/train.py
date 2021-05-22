from bert.preprocess import Data_Loader
import torch
from torch import nn
from bert.model import Electra
import pandas as pd

# DICTIONARY: pd.DataFrame = pd.read_csv("datasets/adam/train_2500_sort_AB_Exp.txt", sep='\t')
DICTIONARY: pd.DataFrame = pd.read_csv("datasets/adam/valid_adam.txt", sep='\t')

def train_loop(dataloader, model, loss_fn, optimizer, max = -1):
    size = dataloader.size
    for batch, (X, loc, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X, loc)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if max > 0:
            if batch >= max:
                break

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


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
    
    # Hyperparameters
    learning_rate = 1e-3
    # batch_size = 16
    epochs = 1
    
    # Model
    model = Electra()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Data
    train_dataloader = Data_Loader("datasets/medal/train1000.csv")

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, max = 5)
        # test_loop(test_dataloader, model, loss_fn)
    print("Done!")

if __name__ == "__main__":
    main()