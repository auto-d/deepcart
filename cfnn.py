
from tqdm import tqdm
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator 
from sklearn.metrics.pairwise import cosine_similarity
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
import similarity

class CfnnEstimator(BaseEstimator):
    """
    A basic collaborative filtering implementation that employs rating 
    distribution similarity measures to produce item recommendations. 
    """ 
    
    def __init__(self):
        """
        Initialize a new instance of the model 
        """
        self.model_u_map = None
        self.model_i_map = None
        self.similarity_matrix = None

    def compare_users(self, ui_a, a_ix, ui_b, b_ix, metric):
        """
        Report a similarity score given the user-item matrices and associated indices for two users
        """
        # Given the sparsity of our review vectors, cosine similarity is going to be 
        # effectively zero if we look across the entire item space... compare only those 
        # items these two users have in common (at least 1 rating between the two).
        a_item_ix = np.nonzero(ui_a[a_ix])[0]            
        b_item_ix = np.nonzero(ui_b[b_ix])[0]
        all_ix = np.concatenate([a_item_ix, b_item_ix])
        a_items = ui_a[a_ix][all_ix]            
        b_items = ui_b[b_ix][all_ix]

        # Fill non-ratings with middling scores. Non-interactions appear 
        # dissimilar to positive reviews and similar to negative ones otherwise.
        a_items[a_items==0] = 3
        b_items[b_items==0] = 3
        
        # Cosine similarity risks insensitivity to rating value, while imperfect here, 
        # Pearson similarity gets us sensitivty to rating magnitude and trends
        sim = None
        if metric=='pearson': 
            sim = similarity.pearson_similarity(a_items, b_items)
        else: 
            sim = cosine_similarity([a_items], [b_items])[0][0]

        return sim 
    
    def fit(self, train, val, val_chk, k=10, metric='pearson'): 
        """
        Fit our estimator given a user-item matrix and validation matrices, only 
        holding k of the most similar users to ease up the memory impact. 
        """ 

        # This isn't implied by the name, but this densifies the matrix, i.e. we have a contiguous u x i
        # matrix here (user vector of item ratings) ... though it's actually not clear how the memory is 
        # managed underneath in scipy, the 'dense' array might just be a bunch of pointers to the DFs stored 
        # in the AffMat object... 
        ui, u_map, i_map = train.gen_affinity_matrix()
        
        # Populating a full user similarity matrix is inherently limited by (and is a questionable 
        # strategy because of) the u^2 memory requirement. Unlike our affinity matrices, 
        # these are not sparse. Since our goal is recommending items, we'll compute the similarity
        # iteratively and store just the top similar users to get down to C * u memory pattern
        similarity_matrix = np.array([[0] * k] * len(u_map))

        # ❗TODO: be careful here, validate that our array multiplication above created distinct arrays 
        # and not copies of the same array which 

        for a in tqdm(range(len(u_map))): 

            # Collect our similarities w/ respect to user A
            sim_a = {}
            for b in range(len(u_map)): 
                if a != b:                     
                    sim_a[b] = self.compare_users(ui, a, ui, b, metric=metric)

            # Find and store the top k user matches, in order    
            # NOTE: dict sorting logic courtesy of gpt-4o (https://chatgpt.com/share/687dc72f-54b4-8013-806e-b1de20d0ef12)
            top = sorted(sim_a.items(), key=lambda x: x[1], reverse=True)[:k]
            similarity_matrix[a] = [x[0] for x in top]

        self.model_ui = ui
        self.model_u_map = u_map 
        self.model_i_map = i_map 
        self.similarity_matrix = similarity_matrix
        return self

    def recommend(self, ui, k) -> np.ndarray: 
        """
        Generate top k predictions given a list of item ratings (one per user)
        """
        recommendations = []
        
        tqdm.write(f"Running predictions... ")
        ui, u_map, i_map = ui.gen_affinity_matrix() 

        # For each provided user, find the most similar user
        for u in tqdm(range(len(u_map))): 
                        
            rated = []
            proxies = []

            target_id = similarity.find_key(self.model_u_map, u)
            
            # We need to cache items this user has already interacted with and find a proxy 
            # for their reviews (to make recommendations). If this user hasn't been seen 
            # previously we'll have to dig around in training data for a similar user.
            if not target_id:
                tqdm.write(f"Unknown user {similarity.find_key(u_map,u)} encountered!")
                
                # Do a live similarity comparison to find a proxy user in the training data
                best_match = 0
                for user_id, user_ix in self.model_u_map.items(): 
                    sim = self.compare_users(ui, u, self.model_ui, user_ix) 
                    if sim > best_match: 
                        target_id = user_id
                        target_ix = user_ix                        
                                
                # Our only evidence of this user's reviews are what came along in the dataset
                # here. Map these ratings to the associated items in our training data. 
                new_rated = list(np.nonzero(ui[u])[0]) 
                rated = similarity.map_keys(i_map, new_rated, self.model_i_map)
            
            else:
                target_ix = self.model_u_map.get(target_id)
                rated = list(np.nonzero(self.model_ui[target_ix])[0])
                
            # Retrieve the most similar users (for the current user or their stand-in) recorded 
            # during training... these will source recommendations 
            proxies = self.similarity_matrix[target_ix]
              
            # Find a/the highest rated items which the target user hasn't yet interacted with
            recommended = []
            
            for proxy in proxies: 

                while True: 
                    best_rated = similarity.argmax(self.model_ui[proxy], exclude=rated + recommended) 
                    
                    # If we're out of 'good' items, move to the next user
                    if self.model_ui[proxy][best_rated] <= 3: 
                        break 

                    # This is worth recommending, stick it on the list
                    row = [
                        similarity.find_key(u_map, u), 
                        similarity.find_key(self.model_i_map, best_rated), 
                        self.model_ui[proxy][best_rated]
                        ]
                    recommendations.append(row)

                    # Iterate until full of recs 
                    recommended.append(best_rated)
                    if len(recommended) >= k: 
                        break
        
        df = pd.DataFrame(recommendations, columns=['user_id', 'item_id', 'rating']) 
        return df 
    
    def score(self, top_ks, test_chk, k):
        """
        Employ the recommenders library to calculate MAP@K here. 
        NOTE: Recommenders examples used to source call semantics, see e.g.
        https://github.com/recommenders-team/recommenders/blob/main/examples/02_model_collaborative_filtering/standard_vae_deep_dive.ipynb
        """        

        tqdm.write(f"Scoring recommendations... ")

        map = map_at_k(
            test_chk.df, 
            top_ks, 
            col_item="item_id", 
            col_user="user_id", 
            col_prediction='prediction', 
            k=k)
        
        tqdm.write(f"MAP@K (k={k}): {map}")

        return map

def save_model(model, path):
    """
    Save the model to a file
    """    
    filename = os.path.join(path, 'cfnn.pkl')

    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    
    tqdm.write(f"Model saved to {path}")

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

def train(train, val, val_chk): 
    """
    Train the model     
    """
    model = CfnnEstimator() 
    model.fit(train, val, val_chk)
    return model     

def test(model, test, test_chk, top_k):
    """
    Test the CFNN model 
    """
    top_ks = model.recommend(test, top_k)
    scores = model.score(top_ks, test_chk, top_k)
    tqdm.write(f"CF NN mean scores for the provided dataset: {np.mean(scores)}")