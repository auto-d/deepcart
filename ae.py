import os
import math 
import torch 
import pandas as pd 
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import sys
sys.path.append('..')
from recommenders.datasets.sparse import AffinityMatrix
from recommenders.datasets.python_splitters import python_random_split
from recommenders.datasets.python_splitters import python_stratified_split 

class Autoencoder(nn.Module):
    """
    Autoencoder

    NOTE: with cues from https://www.geeksforgeeks.org/deep-learning/implementing-an-autoencoder-in-pytorch/
    """

    def __init__(self, dims=1000):
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
    
class DeepCartDataset(torch.utils.data.Dataset): 
    """
    Custom pytorch-compatible dataset. Adapted from 
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    """
    def __init__(self, users, reviews): 
        """
        Initialize a new instance

        Oof. The ideal pattern here is for the dataset to be blissfully ignorant of our split strategy and just 
        make a dataset available to its client based on the raw data passed. However the split strategy is rather intricate below ... 
        note the five splits. Can we easily raise that up to a higher level? The refactoring might not be trivial and it may result in 
        residue of this split strategy bleeding over to the other models -- what's common and what's not? 

        common: 
        - train, val, test split
        - val test and test test bonus splits - for all validation stages we need to check performance, however non-NN techniques 
          really only need a test split, right? if we do train holdout for validation, we essentially have three test sets (train-test, val, test)
        - a need to operate on the same validation or at least test data, lest the comparison be biased by the selection method each model applies

        unique
        - logic to prune reviews < 3.5 -- we don't do this in cfnn, and naive doesn't care (predicts highest review in the matrix), if this is 
          done during training, it will also need to be done during inference
        - need for a pytorch-style dataset ... the naive method is doing a O(n) search, the cfnn needs dataframes -- while refactoring is 
          possible, why understake the risk it will be a disjoint and inelegant fit? 
        - the VAE implementation wants all train and val, but doesn't require a test dataset. we ou


        we could: 
        - pass train and val, hold test out
        - pass test to predict function, which we need for the demo anyway
        - keep the pytorch dataset unique to the pytorch-compatible class... doesn't make sense to try and foist on other algos... we are 
        doing this in the wrong order, filtering and then splitting... we need to outsource the splitting and then do the filtering inside each 
        model 

        right now this is speculation, just get something working! we can figure out how to streamline after -- oh, but we need a dataset 
        implementation
        """
        self.users = users 
        self.reviews = reviews 
        self.matrix = 

    def build_affinity_matrices(): 
        """
        
        """
        oof

    def split(users, reviews, items):
        """
        Generate splits 
        """
        print(f"Full user-item matrix is {len(users) * len(items)}")



# generic need
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

# really all of these models just need the review data to train, the users and item lists can be extracted from that

# AE specific ... 
        # We are trying to teach the model what a good interaction is like, and we'll 
        # ultimately be interested only in whether to recommend an item or not ... 
        # low reviews are not something we want the model suggesting... 
        reviews_low = reviews[reviews.rating < 3]
        reviews = reviews[reviews.rating >= 3]

        header = {
            "col_user": "user_id",
            "col_item": "item_id",
            "col_rating": "rating",
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
    def __len__(self): 
        """
        Retrieve length of the dataset
        """
        return len(self.img_labels) 
    
    def __getitem__(self, idx): 
        """
        Retrieve an item at the provided index
        """
        #TODO: implement
        pass

def get_data_loader(batch_size=5, shuffle=True): 
    """
    Retrieve a pytorch-style dataloader 
    """

    #TODO: implement
    #transform = transforms.Compose([
    #     transforms.ConvertImageDtype(torch.float),
    #     transforms.Normalize(mean=[0.5], std=[0.5])
    #])

    #data = DeepCartDataset(transform=transform)
    #loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    
    #return loader
    pass

def train(loader, model, loss_interval=20, epochs=2, lr=0.01, momentum=0.9):
    """
    Train the model with the provided dataset

    NOTE: this is a similar training loop as we used for our vision model in the 
    the vision project, forward pass
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_loss = []

    tqdm.write(f"Starting training run...")    
    # TODO: configure WandB
    # see https://docs.wandb.ai/guides/integrations/pytorch/
    config = {}
    run = wandb.init(config=config) 

    model.train()
    model = model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(epochs):

        running_loss = 0.0
        for i, data in enumerate(loader):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # collect metrics
            running_loss += loss.item()

            if (i % loss_interval) == (loss_interval - 1): 
                train_loss.append(running_loss / loss_interval)
                tqdm.write(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / loss_interval:.3f}")
                running_loss = 0 
    
    tqdm.write("Training complete!") 

    return train_loss 