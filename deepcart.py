#!/usr/bin/env python3

import argparse 
import os
import naive
import autoencoder
import cfnn 
import naive
import tempfile
import glob
from dataset import DeepCartDataset, DeepCartTorchDataset
from process import run_subprocess
from demo import demo

def deploy(share=False, data_tag="test"): 
    """
    Deploy, optionally pushing to the cloud (share = True)
    """
    demo(share, data_tag)

def load_secrets(): 
    """
    Find our secrets and return them
    NOTE: reused from NLP project
    """
            
    # TODO: update or discard 
    # hf_token = os.environ.get("HF_TOKEN")
    # if not hf_token: 
    #     with open("secrets/huggingface.token", "r") as file: 
    #         hf_token = file.read()
    
    return None

def readable_file(path):
    """
    Test for a readable file
    NOTE: reused from NLP project
    """
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"'{path}' doesn't exist.")
    return path

def nonexistent_file(path):
    """
    Test for a non-existent file (to help avoid overwriting important stuff)
    NOTE: reused from NLP project
    """
    if os.path.exists(path):
        raise argparse.ArgumentTypeError(f"'{path}' already exists.")
    return path

def readable_dir(path):
    """
    Test for a readable dir
    NOTE: reused from NLP project
    """
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"'{path}' is not a valid directory.")
    
def nonexistent_dir(path): 
    """
    Test to ensure directory doesn't exist
    NOTE: reused from NLP project
    """    
    if os.path.exists(path):
        if os.path.isdir(path):
            raise argparse.ArgumentTypeError(f"Directory '{path}' already exists.")
        else:
            raise argparse.ArgumentTypeError(f"Path '{path}' exists and is not a directory.")
    return path

def build_parser(): 
    """
    Apply a command-line schema, returning a parser
    """
    parser = argparse.ArgumentParser("deepcart", description="Amazon electronics recommendations via variational autoencoder")

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Build mode 
    build_parser = subparsers.add_parser("build") 
    build_parser.add_argument("--items-file", type=readable_file, help="File to containing item metadata", default="data/2023/items_1.6M.parquet", required=False)
    build_parser.add_argument("--reviews-file", type=readable_file, help="File to contining review data", default="data/2023/reviews_10M.parquet", required=False)
    build_parser.add_argument("--min-interactions", type=int, help="Minimum threshold for number of ratings by a user", default=10, required=False)
    build_parser.add_argument("--min-ratings", type=int, help="Minimum number of product reviews an item must have", default=10)
    build_parser.add_argument("--sample-n", type=int, help="Number of users to sample from the total", default=10000, required=False)
    build_parser.add_argument("--output-dir", type=readable_dir, help="Directory to write resulting dataset to", default="data/processed", required=False)
    build_parser.add_argument("--tag", type=str, help="Friendly name to tag dataset names with", required=True)

    # Train mode 
    train_parser = subparsers.add_parser("train") 
    train_parser.add_argument("--data-dir", type=readable_dir, help="Directory to look for tagged dataset", default="data/processed", required=False)
    train_parser.add_argument("--data-tag", type=str, help="Dataset tag to look for (set during creation)", required=True)
    train_parser.add_argument("--model-dir", help="Directory to write resulting model to", default="models")
    train_parser.add_argument("--nn-epochs", type=int, default=1)
    train_parser.add_argument("--nn-batch", type=int, default=1)
    train_parser.add_argument("--type", choices=['naive', 'classic', 'neural'], default='neural')

    # Test mode 
    test_parser = subparsers.add_parser("test") 
    test_parser.add_argument("--model_dir", type=readable_dir, help="Directory to load model from", default="models")
    test_parser.add_argument("--data-dir", type=readable_dir, help="Directory to look for tagged dataset", default="data/processed", required=False)    
    test_parser.add_argument("--data-tag", type=str, help="Dataset tag to look for (set during creation)", required=True)
    test_parser.add_argument("--top-k", type=int, help="Number of recommendations to evaluate", default=10)
    test_parser.add_argument("--type", choices=['naive', 'classic', 'neural'], default='neural')

    # Deploy mode 
    deploy_parser = subparsers.add_parser("deploy")
    deploy_parser.add_argument("--data-tag", type=str, help="Data tag associated with the ref data")
    deploy_parser.add_argument("--share", type=bool, default=False, help="Whether or not to deploy to gradio hosting")
    
    return parser
    
def router(): 
    """
    Argument processor and router

    @NOTE: Argparsing with help from chatgpt: https://chatgpt.com/share/685ee2c0-76c8-8013-abae-304aa04b0eb1
    @NOTE: arg parsing logic incorporates work from NLP assignment
    """

    parser = build_parser() 
    args = parser.parse_args()    
    token = load_secrets()
    
    match(args.mode):
        case "build":
            dataset = DeepCartDataset(args.tag)
            dataset.extract(
                args.items_file, 
                args.reviews_file,
                args.min_interactions, 
                args.min_ratings, 
                args.sample_n)
            dataset.store(args.output_dir)

        case "train":
            dataset = DeepCartDataset(args.data_tag)
            dataset.load(args.data_dir)
            dataset.split()

            match(args.type): 
                case 'naive':
                    model = naive.train(dataset.train, dataset.val, dataset.val_chk)
                    naive.save_model(model, args.model_dir)
                case 'classic':
                    model = cfnn.train(dataset.train, dataset.val, dataset.val_chk) 
                    cfnn.save_model(model, args.model_dir)
                case 'neural': 
                    torch_dataset = DeepCartTorchDataset(matrix=dataset.train, batch_size=args.nn_batch)
                    model = autoencoder.train(torch_dataset, args.nn_epochs, dataset.val, dataset.val_chk)
                    autoencoder.save_model(model, args.model_dir)

        case  "test":
            dataset = DeepCartDataset(args.data_tag)
            dataset.load(args.data_dir)
            dataset.split() 
            match (args.type): 
                case 'naive':
                    model = naive.load_model(args.model_dir)
                    naive.test(model, dataset.test, dataset.test_chk, top_k=args.top_k)
                case 'classic':
                    model = cfnn.load_model(args.model_dir)
                    cfnn.test(model, dataset.test, dataset.test_chk, top_k=args.top_k) 
                case 'neural': 
                    model = autoencoder.load_model(args.model_dir)
                    torch_dataset = DeepCartTorchDataset(matrix=dataset.test)
                    autoencoder.test(model, torch_dataset, dataset.test_chk, top_k=args.top_k)

        case "deploy":
            deploy(args.share, args.data_tag)
        case _:
            parser.print_help()

if __name__ == "__main__": 
    router()
