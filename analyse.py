import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from configfile import *

if __name__ == "__main__":
    if not isfile("embeddings.pkl"):
        exit("Embeddings not found. Please evaluate model first!")
    embeddings_df = pd.read_pickle("embeddings.pkl")
    print(f"Loaded embeddings of {embeddings_df.shape[0]} images in {embeddings_df.shape[1] - 2} dimensions.")
    
