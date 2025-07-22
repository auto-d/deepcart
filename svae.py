import numpy as np 
import pandas as pd 
import keras 
from tqdm import tqdm
import os
from recommenders.datasets.sparse import AffinityMatrix
from recommenders.datasets.python_splitters import python_random_split, python_stratified_split 
from recommenders.utils.python_utils import binarize
from recommenders.models.vae.standard_vae import StandardVAE

def train(users, reviews, epochs=2, batch=10):
    """
    Train the model with the provided dataset

    NOTE: this is a similar training loop as we used for our vision model in the 
    the vision project, forward pass
    """
    tqdm.write(f"Starting training run...")        

    # We are trying to teach the model what a good interaction is like, and we'll 
    # ultimately be interested only in whether to recommend an item or not ... 
    # low reviews are not something we want the model suggesting... 
    reviews_low = reviews[reviews.rating < 3]
    reviews = reviews[reviews.rating >= 3]

    # NOTE: Strategy adapted from tutorials available in the Recommenders project, see 
    # https://github.com/recommenders-team/recommenders/tree/main
    # Split along user boundaries to ensure no leakage of preference between train and test
    train_users, test_users, val_users = python_random_split(users, [.9, .05, .05])
    print(train_users.shape, test_users.shape, val_users.shape)

    train = reviews[reviews.user_id.isin(train_users.user_id)]
    val = reviews[reviews.user_id.isin(val_users.user_id)]
    test = reviews[reviews.user_id.isin(test_users.user_id)]
    print(train.shape, val.shape, test.shape)
    
    # Technique from Recommenders (see https://github.com/recommenders-team/recommenders/blob/45e1b215a35e69b92390e16eb818d4528d0a33a2/examples/02_model_collaborative_filtering/standard_vae_deep_dive.ipynb) 
    # to improve utility of validation set during training - only allow items in
    # the validation set that are also present in the train set
    val = val[val.item_id.isin(train.item_id.unique())]
    print(val.shape)

    # Another technique employed in Recommenders (see above link for notebook), for in-flight validation to be 
    # meaningful during training, our validation set needs not just ground truth, but unseen validation samples 
    # to see if predictions for validation users are relevant (to those users). Anyway, break down our val and test 
    # sets again to support this strategy
    val_src, val_target = python_stratified_split(
        data=val, 
        ratio=0.8, 
        filter_by="item", 
        col_user="user_id", 
        col_item="item_id"
        )
    test_src, test_target = python_stratified_split(
        data=test, 
        ratio=0.8, 
        filter_by="item", 
        col_user="user_id", 
        col_item="item_id"
        )
    
    print(val.shape, " -> ", val_src.shape, val_target.shape)
    print(test.shape, " -> ", test_src.shape, test_target.shape)

    #to use standard names across the analysis 
    header = {
            "col_user": "user_id",
            "col_item": "item_id",
            "col_rating": "rating",
            # Unclear why this doesn't also eat a timestamp, but many of the functions that split temporally use, fortunately 
            # the column 'timestamp' (i.e. DEFAULT_TIMESTAMP_COL='timestamp') so I think we're fine. 
            # "col_timestamp" : "timestamp"
        }

    train_matrix = AffinityMatrix(df=train, **header)
    val_matrix = AffinityMatrix(df=val, **header)
    val_src_matrix = AffinityMatrix(df=val_src, **header)
    val_tgt_matrix = AffinityMatrix(df=val_target, **header)
    test_src_matrix = AffinityMatrix(df=test_src, **header)
    test_tgt_matrix = AffinityMatrix(df=test_target, **header)

    # This generates a sparse array of user vectors, aka user-item matrix
    # X[0] is the first user in the list, with entries for all items known when the matrix was constructed in that row
    train, _, _ = train_matrix.gen_affinity_matrix()
    val, _, _ = val_matrix.gen_affinity_matrix() 
    val_src, _, _ = val_src_matrix.gen_affinity_matrix()
    val_tgt, _, _ = val_tgt_matrix.gen_affinity_matrix()
    test_src, _, _ = test_src_matrix.gen_affinity_matrix()
    test_tgt, _, _ = test_src_matrix.gen_affinity_matrix()    

    train = binarize(train, 3)
    val = binarize(train, 3)
    val_src = binarize(val_src, 3) 
    val_tgt = binarize(val_tgt, 3)
    test_src = binarize(test_src, 3)
    test_tgt = binarize(test_tgt, 3)

    sparsity = np.count_nonzero(train)/(train.shape[0]*train.shape[1])*100
    print(f"sparsity: {sparsity:.2f}%")

    model = StandardVAE(
        n_users = train.shape[0], 
        original_dim = train.shape[1],
        intermediate_dim=250, 
        latent_dim=50, 
        n_epochs=1, 
        batch_size=1, 
        k=10, 
        verbose=1, 
        seed=4, 
        save_path="models/svae.hdf5", 
        drop_encoder=0.5, 
        drop_decoder=0.5, 
        annealing=False, 
        beta=1.0) 

    model.fit(
        x_train=train, 
        x_valid=val, 
        x_val_tr=val_src, 
        x_val_te=val_tgt, 
        mapper=val_matrix,
        )
    
    tqdm.write("Training complete!") 

    return model

def predict(loader, model): 
    """
    Run a dataset through the (hopefully trained) model and return outputs
    """

    preds = []
    probas = []

    #TODO: implement

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
    filename = os.path.join(path, "ae.xyz")
    
    #TODO: implement

    tqdm.write(f"Model saved to {filename}")

    return filename

def load_model(path): 
    """
    Load our model 

    NOTE: borrowed from vision project
    """
    model = None

    #TODO: implement

    return model

def summarize_history(history): 
    """
    Mine some salient history 
    """
    #TODO: update or discard 

    df = pd.DataFrame(history) 
    s = (f"***************************\n")
    s += f"Completed {df.epoch.max():.4} epochs ({df.step.max()} steps)\n"
    s += f" - Loss {df.loss.max():.3f} -> {df.loss.min():.3f}\n"
    s += f" - Eval runtime: {df.eval_runtime.sum():.2f}s ({df.eval_runtime.mean():.2f}s/eval)\n"
    return s   
