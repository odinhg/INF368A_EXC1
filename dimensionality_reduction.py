import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import isfile
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
#from configfile import *

def sample_df(df, n=100):
    if n > df.shape[0]:
        n = df.shape[0]
    return df.sample(n, random_state=420)

if __name__ == "__main__":
    if not (isfile("embeddings_train.pkl") and isfile("embeddings_test.pkl") and isfile("embeddings_test.pkl")):
        exit("Embeddings not found. Please evaluate model first!")
   
    # Load embeddings
    df_test = pd.read_pickle("embeddings_test.pkl")
    df_train = pd.read_pickle("embeddings_train.pkl")
    df_unseen = pd.read_pickle("embeddings_unseen.pkl")
    
    # Randomly subsample images
    number_of_samples = 1000
    df_test = sample_df(df_test, number_of_samples)
    df_train = sample_df(df_train, number_of_samples)
    df_unseen = sample_df(df_unseen, number_of_samples)
    df_all = pd.concat([df_test, df_train, df_unseen])
    
    # Standardize features
    standard_scaler = StandardScaler().fit(df_all.iloc[:,1:])
    df_test.iloc[:,1:] = standard_scaler.transform(df_test.iloc[:,1:])
    df_train.iloc[:,1:] = standard_scaler.transform(df_train.iloc[:,1:])
    df_unseen.iloc[:,1:] = standard_scaler.transform(df_unseen.iloc[:,1:])
    df_all.iloc[:,1:] = standard_scaler.transform(df_all.iloc[:,1:])

    # Fit UMAP and reduce dimensions
    reducer = umap.UMAP(verbose=True)
    reducer.fit(df_all.iloc[:,1:])
    df_projection_test = reducer.transform(df_test.iloc[:,1:])
    df_projection_train = reducer.transform(df_train.iloc[:,1:])
    df_projection_unseen = reducer.transform(df_unseen.iloc[:,1:])

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
    ax[0].scatter(df_projection_test[:,0], df_projection_test[:,1], c=df_test.label_idx, s=5)
    ax[0].set_aspect("equal", "datalim")
    ax[0].set_title("UMAP (embedded test data)")
    plt.show()
