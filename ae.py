import os
import math 
import torch 
import pandas as pd 
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

class Autoencoder(nn.Module):
    """
    Autoencoder

    NOTE: with cues from https://www.geeksforgeeks.org/deep-learning/implementing-an-autoencoder-in-pytorch/
    """

    def __init__(self, dims=1000):
        """
        Initialize a new object given an item count 
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(dims, 500),
            nn.ReLU(), 
            nn.Linear(500, 75),
            nn.ReLU(), 
        )
        self.decoder = nn.Sequential(
            nn.Linear(75, 500),
            nn.ReLU(), 
            nn.Linear(500, dims),
            nn.ReLU(), 
        )

    def forward(self, x):
        """
        Implement our forward pass 
        """
        h = self.encoder(x) 
        r = self.decoder(h)

        return r

def train(loader, model, loss_interval=20, epochs=2, lr=0.01, momentum=0.9):
    """
    Train the model with the provided dataset

    NOTE: this is a similar training loop as we used for our vision model in the 
    the vision project, forward pass
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_loss = []

    tqdm.write(f"Starting training run...")    
    # TODO: configure WandB
    # see https://docs.wandb.ai/guides/integrations/pytorch/
    config = {}
    run = wandb.init(config=config) 

    model.train()
    model = model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(epochs):

        running_loss = 0.0
        for i, data in enumerate(loader):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # collect metrics
            running_loss += loss.item()

            if (i % loss_interval) == (loss_interval - 1): 
                train_loss.append(running_loss / loss_interval)
                tqdm.write(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / loss_interval:.3f}")
                running_loss = 0 
    
    tqdm.write("Training complete!") 

    return train_loss 