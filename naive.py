import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.base import BaseEstimator 
import matplotlib.pyplot as plt
import pickle
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
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

        # Assemble a list of the mean ratings for reviewed items
        item_ratings = [0] * len(i_map)
        for i in tqdm(i_map): 
            
            ratings = []
            for u in range(len(u_map)): 
                if ui[u][i] != 0: 
                    ratings.append(ui[u][i])
            
            item_ratings[i] = np.mean(ratings)
        
        self.item_ratings = item_ratings
         
        return self
        
    def recommend(self, ui, k) -> np.ndarray: 
        """
        Generate top k predictions given a list of item ratings (one per user)
        """
        recommendations = []
                
        tqdm.write(f"Running predictions... ")
        ui, u_map, i_map = ui.gen_affinity_matrix() 

        # For each requested prediction (user), find the best-reviewed items that this 
        # user hasn't already reviewed... 
        for u in tqdm(range(len(u_map))): 
            rated = list(np.nonzero(ui[u])[0]) 
            recommended = []
            
            while len(recommended) < k: 
                best_rated = similarity.argmax_(self.item_ratings, exclude=rated + recommended)
                recommended.append(best_rated) 
                
                # Recommendations need to be in a format suitable for scoring w/ the 
                # Recommenders MAP@K. I.e. dataframe with cols user, item & rating             
                row = [
                    similarity.find_key(u_map, u), 
                    similarity.find_key(i_map, best_rated), 
                    self.item_ratings[best_rated]
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
    top_ks = model.recommend(test)
    scores = model.score(top_ks, test, test_chk)
    tqdm.write(f"Naive mean scores for the provided dataset: {np.mean(scores)}")