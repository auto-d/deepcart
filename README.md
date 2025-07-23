# DeepCart - Amazon Electronics Recommender System

DeepCart is a tool to sift through Amazon electronics purchase historicals and source recommendations to drive customer engagement. 

## TODO 

- be clear on how this will be presented
- implement autoencoder from scratch 
- implment naive approach that picks the most popular item
- set up evaluation for all models
- build a web app to demo
- write the readme 
- record a video

## Problem 

TODO: flip this around and make it a system for telling the user what not to buy. That is, recommend the worst reviewed items and mine the review information for potentially hilarious results. Make sure to cite: https://chatgpt.com/share/687f9234-23a8-8013-9779-969676cf35b1 

We would just need to reverse the data on load... add a command-line switch that enables this during training. 

## Data Sources

The University of California at San Diego (UCSD) Amazon Review 2023 dataset is the sole source of user interaction data mined for this project. From the project page, the dataset consists of 
- User Reviews (ratings, text, helpfulness votes, etc.);
- Item Metadata (descriptions, price, raw image, etc.);
- Links (user-item / bought together graphs).

We are specifically concerned with the Electronics subset of this expansive dataset, which consists of 18.3 million user reviews, 1.6M items and 43.9 million ratings. Subsampling on users was conducted to reduce the computational complexity of generating recommendations across all models tested. 
  
## Prior Efforts 

To the author's knowledge, no prior efforts exist to leverage this reviews dataset to make recommendations using an autoencoder or to recommend negative reviews for comedic effect. 

## Model Evaluation and Selection 

We apply mean average precision @ k (MAP@K) to baseline and compare our modeling approaches. 

### Data Processing Pipeline 

The Amazon dataset is 

1. Download review and product datasets
2. Leverage command-line JSON parser for effiency to filter non-essential columns from both (maxi -> mini)
3. Ingest JSON, run needed type conversions and write as Parquet to gain efficiency on future read/write ops and reduce memory pressure
4. 

### Models

## Repository Layout

  
## Quickstart 

All testing done with Python 3.9 

1. `pip install -r requirements.txt` 
2. 

## Usage 

```usage: deepcart build [-h] [--items-file ITEMS_FILE] [--reviews-file REVIEWS_FILE] [--min-interactions MIN_INTERACTIONS] [--min-ratings MIN_RATINGS] [--sample-n SAMPLE_N] [--output-dir OUTPUT_DIR] --tag TAG```

**Generate a Dataset** 

To generate a dataset, run the `build` command and specify the number of interactions, ratings, and user subsample. For example: 

```
deepcart$ python main.py build --min-interactions 10 --min-ratings 100 --sample-n 10000 --tag small

Generating dataset based on data/2023/items_1.6M.parquet and data/2023/reviews_10M.parquet...
Found 2,294,450 users with 10,000,000 ratings of 1,610,012 items.
Dropped 1,350,025 items (<100 ratings)
Dropped 2,069,321 users (reviews <10)
Dropped 4,261,903 reviews (no user associated)
Dropped 198,411 items (no review associated)
Writing 196,325 reviews as data/processed/reviews_small.parquet...
Writing 259,987 items as data/processed/items_small.parquet...
Wrote 'small' dataset to data/processed.
Generation complete!
```

**Train Models** 

**Evaluate Models** 

**Deploy a Demo** 

## Demo Application

## Results and Conclusions

### Challenges 

## Ethics Statement

### Provenance

@TODO: update 

**Data** 
This project was developed as a proof-of-concept to aid the rapid uptake and ultimate mastery of complex software projects. The data used to construct the synthetic training sets was sourced entirely from the Linux open source software project [5] which is licensed under GPL2, and other permissive licenses. 

**Reproducability** 
The code written in this project is the author's work, made possible by a host of righteous open source software packages, tools, and the Ubuntu Linux distribution. Code snippets sourced from articles, tutorials and large-language model chat sessions are annotated in the source code where appropriate. All results here should be reproducible freely, without any licensing implications. 

**Harmful Content** 
The synthetic datasets generated are based purely on Linux kernel symbols and their associated references. Source code and comments are ingeseted into the question datasets used for training. A thorough review of this material has not been conducted, and latent bias, offensive content, or malicious code may have been unintentionally incorporated into the resulting dataset accordingly. 

## References

[1] Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024). Bridging language and items for retrieval and recommendation. arXiv preprint arXiv:2403.03952.