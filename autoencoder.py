import torch 
import pandas as pd 
import torch
from tqdm import tqdm
import os
import math 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

class Autoencoder(nn.Module):
    """
    Autoencoder
    """

    def __init__(self, w=32):
        """
        Initialize a new object 
        """
        super().__init__()
        
        #TODO: implement

    def forward(self, x):
        """
        Implement our forward pass 
        """

        return x

class DeepCartDataset(torch.utils.data.Dataset): 
    """
    Custom pytorch-compatible dataset. Adapted from 
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    """
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None): 

        self.img_labels = pd.read_csv(annotations_file)

        #TODO: implement

    def __len__(self): 
        return len(self.img_labels) 
    
    def __getitem__(self, idx): 
        #TODO: implement
        pass
    
def get_data_loader(batch_size=5, shuffle=True): 
    """
    Retrieve a pytorch-style dataloader 
    """

    #TODO: implement
    #transform = transforms.Compose([
    #     transforms.ConvertImageDtype(torch.float),
    #     transforms.Normalize(mean=[0.5], std=[0.5])
    #])

    #data = DeepCartDataset(transform=transform)
    #loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    
    #return loader
    pass

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

def predict(loader, model): 
    """
    Run a dataset through the (hopefully trained) model and return outputs
    """

    preds = []
    probas = []

    # Reduce the memory required for a forward pass by disabling the 
    # automatic gradient computation (i.e. commit to not call backward()
    # after this pass)
    with torch.no_grad(): 
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device) 
        model.eval() 

        # Compute the logits for every class and grab the class
        # TODO: switch this to top-k? 
        for i, data in enumerate(loader): 
            inputs, _ = data
            inputs = inputs.to(device)
            outputs = model(inputs) 

            predictions = outputs.to("cpu").flatten()
            preds.append(torch.argmax(predictions))
            probas.append(predictions)

    return preds, probas

def test(model, dataset):
    """
    Test model !
    """

    dataset = pd.read_parquet(dataset)    

    #TODO: implement
    
def save_model(model, path):
    """
    Save the model to a file
    NOTE: borrowed from vision project
    """
    filename = os.path.join(path, "ae.pt")
    torch.save(model, filename)
    tqdm.write(f"Model saved to {filename}")

    return filename

def load_model(path): 
    """
    Load our model 

    NOTE: borrowed from vision project
    """
    model = torch.load(os.path.join(path, "ae.pt"), weights_only=False)

    return model

def summarize_history(history): 
    """
    Mine some salient history 
    """
    df = pd.DataFrame(history) 
    s = (f"***************************\n")
    s += f"Completed {df.epoch.max():.4} epochs ({df.step.max()} steps)\n"
    s += f" - Loss {df.loss.max():.3f} -> {df.loss.min():.3f}\n"
    s += f" - Eval runtime: {df.eval_runtime.sum():.2f}s ({df.eval_runtime.mean():.2f}s/eval)\n"
    return s   
