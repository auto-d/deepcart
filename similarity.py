import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

client = OpenAI(api_key="none", base_url="http://localhost:11434/v1")

def embed(text): 
    """
    Generate a text embedding for the input sequence using our trusty ollama 
    nomin embedding model 

    NOTE: snippet from https://platform.openai.com/docs/guides/embeddings
    """    
    response = client.embeddings.create(model="nomic-embed-text:latest", input=text)
    return response.data[0].embedding

def similarity(a, b): 
    """
    Return pairwise similarity between elements in passed arrays
    """
    similarity_matrix = cosine_similarity([embed(a)],[embed(b)])
    pairwise_similarity = np.diag(similarity_matrix)[0]

    return float(pairwise_similarity)