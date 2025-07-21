
from tqdm import tqdm
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator 
from sklearn.metrics.pairwise import cosine_similarity
from recommenders.datasets.sparse import AffinityMatrix
import similarity 
import dataset 

class CfnnEstimator(BaseEstimator):
    """
    A basic collaborative filtering implementation that employs rating 
    distribution similarity measures to produce item recommendations. 
    """ 
    
    def __init__(self):
        """
        Initialize a new instance of the model 
        """
        self.u_map = None
        self.i_map = None
        self.model = None

    def fit(self, reviews, top_k): 
        """
        Fit our estimator
        """ 
        header = {
            "col_user": "user_id",
            "col_item": "item_id",
            "col_rating": "rating",
        }

        ui_sparse = AffinityMatrix(reviews, **header)    

        # This isn't implied by the name, but this densifies the matrix, i.e. we have a contiguous u x i
        # matrix here (user vector of item ratings) ... though it's actually not clear how the memory is 
        # managed underneath in scipy, the 'dense' array might just be a bunch of pointers to the DFs stored 
        # in the AffMat object... 
        ui_dense, u_map, i_map = ui_sparse.gen_affinity_matrix()
        
        # Populating a full user similarity matrix is inherently limited by (and is a questionable 
        # strategy because of) the u^2 memory requirement. Unlike our affinity matrices, 
        # these are not sparse. Since our goal is recommending items, we'll compute the similarity
        # iterativealy and store the most similar users to get down to C * u memory pattern
        similarity_matrix = np.array([[0.] * top_k] * len(u_map))
        for a in range(len(u_map)): 

            # Collect our similarities w/ respect to user A
            sim_a = {}
            for b in range(len(u_map)): 
                if a != b: 
                    
                    # Given the sparsity of our review vectors, cosine similarity is going to be 
                    # effectively zero if we look across the entire item space... compare only those 
                    # items these two users have in common (at least 1 rating between the two).
                    a_item_ix = np.nonzero(ui_dense[a])[0]            
                    b_item_ix = np.nonzero(ui_dense[b])[0]
                    all_ix = np.concatenate([a_item_ix, b_item_ix])
                    a_items = ui_dense[a][all_ix]            
                    b_items = ui_dense[b][all_ix]

                    # Fill non-ratings with middling scores. Non-interactions appear 
                    # dissimilar to positive reviews and similar to negative ones otherwise.
                    a_items[a_items==0] = 3
                    b_items[b_items==0] = 3
                    
                    # TODO: swap this out for pearson to capture magnitude
                    sim_a[b] = cosine_similarity([a_items], [b_items])[0][0]
                    
            # Find and store the top k user matches, in order    
            # NOTE: dict sorting logic courtesy of gpt-4o (https://chatgpt.com/share/687dc72f-54b4-8013-806e-b1de20d0ef12)
            top = sorted(sim_a.items(), key=lambda x: x[1], reverse=True)[:top_k]
            similarity_matrix[a] = [x[0] for x in top]

        self.u_map = u_map 
        self.i_map = i_map 
        self.model = similarity_matrix
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

def train(users, reviews, top_k=5): 
    """
    'Train' the model     
    """
    model = CfnnEstimator() 
    model.fit(reviews, top_k)
    return model     

def test(model, dataset):
    """
    Test the hmm model 
    """
    X, y = load_dataset(dataset)
    # See comment in train() above
    # model = load_model(model_dir)    
    scores = model.score(X, y)
    tqdm.write(f"Collaborative nearest neighbor mean scores for the provided dataset: {np.mean(scores)}")