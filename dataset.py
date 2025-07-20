import os
import pandas as pd

def extract_users(reviews): 
    """
    Given reviews, generate a user dataframe     
    """
    users = reviews.groupby(['user_id']).rating.count()
    users = pd.DataFrame(users).reset_index()
    users.rename(columns={'rating':'ratings'}, inplace=True)
    return users 

def build(items_path, reviews_path, tag, output_dir, min_ratings, min_interactions, sample_n): 
    """
    Given review and item data, prepare a cleaned dataset for training
    """
    
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating dataset based on {items_path} and {reviews_path}...")

    if not items_path.endswith(".parquet") or not reviews_path.endswith(".parquet"): 
        raise ValueError("Unexpected file type!")
    
    reviews = pd.read_parquet(reviews_path) 
    items = pd.read_parquet(items_path)
    
    users = extract_users(reviews)
    users_small = users[users.ratings > min_ratings]    
        
    # narrow down reviews, too many to operate on with limited resources
    reviews_small = reviews[reviews.user_id.isin(users_small.user_id.unique())]

    # sample to make this computationally approachable 
    print(len(reviews_small.user_id.unique())) 

    # TODO: add this to the pipeline as a variable
    sample_n = 1000
    sampled_users = users_small.sample(sample_n)
    reviews_sampled = reviews_small[reviews_small.user_id.isin(sampled_users.user_id)]
    print(len(reviews_sampled.user_id.unique())) 
    print(len(reviews_sampled))

    reviews_file = os.path.join(output_dir,f"reviews_{tag}.parquet")
    reviews_sampled.to_parquet(reviews_file)

    items_small = items[items.parent_asin.isin(reviews_sampled.parent_asin.unique())]
    
    items_file = os.path.join(output_dir,f"items_{tag}.parquet")
    items_small.to_parquet(items_file)
        
    print(f"Wrote {tag} dataset to {output_dir}.")
    
    print(f"Generation complete!")

def load(items_path, reviews_path): 
    """
    Read our datasets and return 
    """
    print(f"Loading reviews... ")
    reviews = pd.read_parquet(reviews_path) 
    
    print(f"Loading items... ")
    items = pd.read_parquet(items_path)
    
    print(f"Extracting users ... ")
    users = extract_users(reviews)

    return users, reviews, items