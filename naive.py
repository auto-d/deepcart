import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.base import BaseEstimator 
import matplotlib.pyplot as plt
import pickle
import string 
from similarity import similarity

class NaiveEstimator(BaseEstimator): 
    
    def __init__(self):
        """
        Set up an instance of our naive estimator 
        """
        pass

    def fit(self, X, y): 
        """
        Fit our naive estimator
        """ 

        #TODO: implement 
        
        return self

    def predict(self, X) -> np.ndarray: 
        """
        Generate an answer given a prompt/input/question
        """
        preds = []
        tqdm.write(f"Running predictions... ")
        for x in tqdm(list(X), total=len(X)): 
            #TODO: implement 
            preds.append(None)

        return preds 
    
    def score(self, X, y):
        """
        Sklearn expectation for CV scoring 
        """        
        y_hat = self.predict(X)

        scores = []
        tqdm.write(f"Scoring predictions...")
        for a, b in tqdm(zip(y, y_hat), total=len(y)): 
            scores.append(similarity(a, b)) 

        return scores

def load_dataset(file): 
    """
    Load and return a compatible dataset for the naive classifier
    """
    df = pd.read_parquet(file)
    return df['x'], df['y']

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

def train(dataset, model_dir): 
    """
    'Train' the naive model 
    """
    
    X, y  = load_dataset(dataset)
    model = NaiveEstimator().fit(X, y)
    save_model(model, model_dir)

def test(model_dir, dataset):
    """
    Test the naive model 
    """
    X, y = load_dataset(dataset)
    model = load_model(model_dir)    
    scores = model.score(X, y)
    tqdm.write(f"Naive mean scores for the provided dataset: {np.mean(scores)}")