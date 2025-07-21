import sys 
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

def build(items_path, reviews_path, tag, output_dir, min_interactions, min_ratings, sample_n): 
    """
    Given review and item data, prepare a cleaned dataset for training

    NOTE: the full Amazon dataset processing consumes more RAM than I have available, this
    was preprocessed on the command-line to discard non-essential features and move to a 
    parquet format (from jsonl) to make it manageable. We operate on these preprocessed files
    here. 
    """
    
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating dataset based on {items_path} and {reviews_path}...")

    if not items_path.endswith(".parquet") or not reviews_path.endswith(".parquet"): 
        raise ValueError("Unexpected file type!")
    
    reviews = pd.read_parquet(reviews_path) 
    items = pd.read_parquet(items_path)    
    users = extract_users(reviews)
    print(f"Found {len(users):,} users with {len(reviews):,} ratings of {len(items):,} items.")

    # Per dataset documentation we should always use parent_asin for correlations as 
    # children are just color, etc... variations of the same product
    reviews.rename(columns={'parent_asin':'item_id'}, inplace=True)
    items.rename(columns={'parent_asin':'item_id'}, inplace=True)

    items_small = items[items.rating_number > min_ratings]
    print(f"Dropped {len(items)-len(items_small):,} items (<{min_ratings} ratings)")

    users_small = users[users.ratings >= min_interactions]
    print(f"Dropped {len(users)-len(users_small):,} users (reviews <{min_interactions})")

    reviews_small = reviews[reviews.user_id.isin(users_small.user_id.unique())]

    sampled_users = users_small.sample(sample_n)
    reviews_sampled = reviews_small[reviews_small.user_id.isin(sampled_users.user_id)]
    print(f"Dropped {len(reviews_small)-len(reviews_sampled):,} reviews (no user associated)")

    items_smaller = items_small[items_small.item_id.isin(reviews_sampled.item_id.unique())]    
    print(f"Dropped {len(items_small)-len(items_smaller):,} items (no review associated)")
        
    reviews_file = os.path.join(output_dir,f"reviews_{tag}.parquet")
    print(f"Writing {len(reviews_sampled):,} reviews as {reviews_file}...")
    reviews_sampled.to_parquet(reviews_file)

    items_file = os.path.join(output_dir,f"items_{tag}.parquet")
    print(f"Writing {len(items_small):,} items as {items_file}...")    
    items_small.to_parquet(items_file)

    print(f"Wrote '{tag}' dataset to {output_dir}.")
    
    print(f"Generation complete!")

def load(dir, tag): 
    """
    Read our datasets and return 
    """    
    reviews_path = os.path.join(dir, f"reviews_{tag}.parquet")
    items_path = os.path.join(dir, f"items_{tag}.parquet")

    print(f"Loading reviews... ")
    reviews = pd.read_parquet(reviews_path) 
        
    print(f"Loading items... ")
    items = pd.read_parquet(items_path)    
    
    print(f"Extracting users ... ")
    users = extract_users(reviews)

    print(f"Memory usage:") 
    print(f" - reviews ~{sys.getsizeof(reviews):,} bytes)")
    print(f" - items ~{sys.getsizeof(items):,} bytes)")
    print(f" - users ~{sys.getsizeof(users):,} bytes)")

    print(f"Non-sparse user-item matrix will be {len(users) * len(items):,} elements!")

    return users, reviews, items