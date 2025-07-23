import os
import math 
import torch 
import numpy as np 
import pandas as pd 
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
import similarity

class Autoencoder(nn.Module):
    """
    Autoencoder

    NOTE: with cues from https://www.geeksforgeeks.org/deep-learning/implementing-an-autoencoder-in-pytorch/
    """

    def __init__(self, dims):
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

class AutoencoderEstimator(): 

    def __init__(self, tensorboard_dir="./runs"): 
        """
        Initialize an object 
        """
        self.module = None
        self.tensorboard_dir = tensorboard_dir

    def train(self, dataset, val, val_chk, epochs=2, lr=0.005, loss_interval=10):
        """
        Train the model with the provided user-item dataset, optionally furnishing a learning 
        rate and interval to plot loss values
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        train_loss = []

        u_map, i_map = dataset.get_mappings()
        model = Autoencoder(len(i_map))  
            
        # Track progress with tensorboard-style output
        writer = SummaryWriter(self.tensorboard_dir)

        # Rehome, if necessary 
        model = model.to(device)
        
        # We'll use MSE since we're interested in correctly reproducing ground-truth reviews
        # the value of the autoencoder will be in the values it infers where no rating was 
        # provided (which will form our recommendations)
        loss_fn = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=lr)

        loader = dataset.get_data_loader()
        tqdm.write(f"Starting training run...")
        for epoch in tqdm(range(epochs), total=epochs):
        
            running_loss = 0.0
            for i, reviews in tqdm(enumerate(loader), total=len(dataset)):

                # Push our review matrix to whatever device is available
                reviews = reviews.to(device)
                
                # Toss any gradient residue from prior runs
                optimizer.zero_grad()

                # Run the reviews through the network and then propagate the gradient
                # backward to improve our alignment with ground-truth review. 
                # Note we mask out any non-reviews to avoid the network learning 
                # to reconstruct, as our prediction is based entirely on the network's 
                # ability to estimate these 
                outputs = model(reviews)
                loss = loss_fn(outputs, reviews)
                loss.backward()

                #TODO: inspect loss values here and mask before computing gradients at each layer
                #https://stackoverflow.com/questions/78958840/how-to-create-a-gradient-mask-in-pytorch

                optimizer.step()

                # Accumulate metrics for hyperparameter tuning
                running_loss += loss.item()

                if (i % loss_interval) == (loss_interval - 1): 
                    interval_loss = running_loss / loss_interval
                    writer.add_scalar('training loss', interval_loss, epoch*len(loader))
                    tqdm.write(f"[{epoch + 1}, {i + 1:5d}] loss: {interval_loss:.3f}")
                    running_loss = 0 
        
        # Update our object state
        self.model = model 
        self.u_map = u_map
        self.i_map = i_map 

        tqdm.write("Training complete!") 

        return model, train_loss 

    def recommend(self, dataset, k) -> np.ndarray: 
        """
        Generate top k predictions given a list of item ratings (one per user)
        """
        recommendations = []

        preds = []
        probas = []

        u_map, _ = dataset.get_mappings()

        # Avoid gradient bookkeeping
        with torch.no_grad(): 
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device) 
            
            # Avoid training interventions like batch norm and dropout
            model.eval() 

            # Generate recommendations
            for u, reviews in tqdm(enumerate(dataset.get_data_loader()), total=len(dataset)):
                
                # Note the ratings for this user
                rated = np.nonzero(reviews) 

                reviews = reviews.to(device)
                logits = model(reviews) 

                # Find the top_k novel recommendations 
                output = logits.to("cpu").flatten()
                recommended = []
                while len(output) < k: 
                    best_rated = similarity.argmax(output, exclude=rated + recommended) 
                    recommended.append(best_rated)
                                        
                    # Record provided user, recommended item and inferred rating
                    row = [
                        similarity.find_key(u_map, u), 
                        similarity.find_key(self.i_map, best_rated),
                        output[best_rated]
                        ]
                    recommendations.append(row)

        df = pd.DataFrame(recommendations, columns=['user_id', 'item_id', 'rating']) 
        return df

    def score(self, top_ks, test, test_chk, k=10):
        """
        Employ the recommenders library to calculate MAP@K here. 
        NOTE: Recommenders examples used to source call semantics, see e.g.
        https://github.com/recommenders-team/recommenders/blob/main/examples/02_model_collaborative_filtering/standard_vae_deep_dive.ipynb
        """        

        tqdm.write(f"Scoring recommendations... ")

        recs_df = test.map_back_sparse(top_ks, kind='prediction')
        test_df = test.map_back_sparse(test_chk, kind='ratings')

        map = map_at_k(test_df, recs_df, col_prediction='prediction', k=k)
        
        tqdm.write(f"MAP@K (k={k}): {map}")

        return map

def save_model(model, path):
    """
    Save the model to a file
    """
    filename = os.path.join(path, "autoencoder.pt")
    torch.save(model, filename)
    print(f"Model saved to {filename}")

    return filename

def load_model(path): 
    """
    Pull a saved model off disk 
    """
    model = torch.load(os.path.join(path, "autoencoder.pt"), weights_only=False)
    
    if type(model) != Autoencoder: 
        raise ValueError(f"Found unexpected type {type(model)} in {path}!")

    return model

def train(train, epochs, val, val_chk):
    """
    Train the autoencoder given the provided dataset     
    """
    model = AutoencoderEstimator()
    model.train(dataset=train, val=val, val_chk=val_chk, epochs=epochs)
    return model 

def test(model, test, test_chk, top_k):
    """
    Test the autoencoder model 
    """
    top_ks = model.recommend(test, top_k)
    scores = model.score(top_ks, test_chk)
    tqdm.write(f"Autoencoder mean scores for the provided dataset: {np.mean(scores)}")

