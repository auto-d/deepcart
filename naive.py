import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.base import BaseEstimator 
import matplotlib.pyplot as plt
import pickle
import random
import similarity

class NaiveEstimator(BaseEstimator): 
    """
    Estimator that grabs the most popular items out of the user-item matrix 
    and vomits them back at prediction time.
    """
    
    def __init__(self):
        """
        Set up an instance of our naive estimator 
        """
        self.item_ratings = None

    def fit(self, train, val, val_chk): 
        """
        Fit our naive estimator
        """ 
        ui, u_map, i_map = train.gen_affinity_matrix() 

        item_ratings = [0] * len(i_map)
        for i in tqdm(i_map): 
            
            ratings = []
            for u in range(len(u_map)): 
                if ui[u][i] != 0: 
                    ratings.append(ui[u][i])
            
            item_ratings[i] = np.mean(ratings)
        
        self.item_ratings = item_ratings
         
        return self
        
    def predict(self, ui) -> np.ndarray: 
        """
        Generate a prediction given a list of item ratings (one per user)
        """
        preds = []
                
        tqdm.write(f"Running predictions... ")
        ui, u_map, i_map = ui.gen_affinity_matrix() 

        # For each requested prediction (user), find the best-reviewed items that this 
        # user hasn't already reviewed... 
        for u in tqdm(range(len(u_map))): 
            rated = list(np.nonzero(ui[u])[0]) 
            recommended = []
            while len(recommended) < 10: 
                best_rated = similarity.argmax_(ui[0], exclude=rated + recommended)
                recommended.append(best_rated) 

            # Retrieve the item ID and aggregate rating by the cached item index
            recommendations = [ { similarity.find_key(i_map, i): ui[u][i] } for i in recommended ]
            preds.append(recommendations)

        return preds 
    
    def score(self, X, y):
        """
        Sklearn expectation for CV scoring 
        """        
        y_hat = self.predict(X)

        #TODO: implement
        scores = []
        tqdm.write(f"Scoring predictions...")
        for a, b in tqdm(zip(y, y_hat), total=len(y)): 
            scores.append(similarity(a, b)) 

        return scores

def save_model(model, path):
    """
    Save the model to a file
    NOTE: copy/pasta from nlp project 
    """    
    filename = os.path.join(path, 'naive.pkl')

    with open(filename, 'wb') as f: 
        pickle.dump(model, f)
    
    tqdm.write(f"Model saved to {path}")

    return filename

def load_model(path): 
    """
    Load our naive model off disk
    NOTE: copy/pasta from nlp project 
    """
    model = None

    filename = os.path.join(path, 'naive.pkl')
    with open(filename, 'rb') as f: 
        model = pickle.load(f) 
    
    if type(model) != NaiveEstimator: 
        raise ValueError(f"Unexpected type {type(model)} found in {filename}")

    return model

def train(train): 
    """
    'Train' the naive model 
    """

    return NaiveEstimator().fit(train)

def test(model, test, test_chk):
    """
    Test the naive model 
    """
    preds = model.predict(test)
    scores = model.score(preds, test_chk)
    tqdm.write(f"Naive mean scores for the provided dataset: {np.mean(scores)}")