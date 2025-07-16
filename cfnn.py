from tqdm import tqdm
import random 
import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator 
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import string 
from naive import tokenize, clean, lemmatize, similarity

class CfnnEstimator(BaseEstimator): 
    
    def __init__(self):
        """
        Set up an instance of our hidden markov model estimator 
        """


    def fit(self, X, y): 
        """
        Fit our estimator
        """ 

        #TODO: implement

        return self

    def predict(self, X) -> np.ndarray: 
        """
        Generate an answer given a prompt/input/question
        """
        preds = []
        tqdm.write(f"Running predictions...")
        for x in tqdm(list(X)): 
        
            #TODO: implement
            pass
        
        return preds 
    
    def score(self, X, y):
        """
        Sklearn expectation for CV scoring 
        """        
        y_hat = self.predict(X)

        scores = []
        tqdm.write(f"Scoring predictions...")
        for a, b in tqdm(zip(y, y_hat), total=len(y)): 
            
            #TODO: implement
            pass

        return scores

def load_dataset(file): 
    """
    Load and return a compatible dataset for the naive classifier
    """
    df = pd.read_parquet(file)
    
    #TODO: implement

def save_model(model, path):
    """
    Save the model to a file
    """    
    filename = os.path.join(path, 'cfnn.pkl')

    with open(filename, 'wb') as f:         
        
        #TODO: get model and save

        pickle.dump(model, f)
    
    print (f"Model saved to {path}")

    return filename

def load_model(path): 
    """
    Load our naive model off disk
    NOTE: copy/pasta from vision project 
    """
    model = None

    filename = os.path.join(path, 'cfnn.pkl')
    with open(filename, 'rb') as f: 
        model = pickle.load(f)         
    
    if type(model) != CfnnEstimator: 
        raise ValueError(f"Unexpected type {type(model)} found in {filename}")

    return model

def train(dataset, model_dir): 
    """
    'Train' the model 
    """
    
    X, y  = load_dataset(dataset)
    model = CfnnEstimator().fit(X, y)
    
    save_model(model, model_dir)

    #TODO: implement
    #test(model, ??)

def test(model, dataset):
    """
    Test the hmm model 
    """
    X, y = load_dataset(dataset)
    # See comment in train() above
    # model = load_model(model_dir)    
    scores = model.score(X, y)
    tqdm.write(f"Collaborative nearest neighbor mean scores for the provided dataset: {np.mean(scores)}")