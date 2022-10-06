import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import isfile
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
from configfile import class_names, class_names_unseen

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
    number_of_samples = 1500
    df_test = sample_df(df_test, number_of_samples)
    df_train = sample_df(df_train, number_of_samples)
    df_unseen = sample_df(df_unseen, number_of_samples)
    df_all = pd.concat([df_test, df_train, df_unseen])
    
    # Standardize features
    standard_scaler = StandardScaler().fit(df_all.iloc[:,2:])
    df_test.iloc[:,2:] = standard_scaler.transform(df_test.iloc[:,2:])
    df_train.iloc[:,2:] = standard_scaler.transform(df_train.iloc[:,2:])
    df_unseen.iloc[:,2:] = standard_scaler.transform(df_unseen.iloc[:,2:])
    df_all.iloc[:,2:] = standard_scaler.transform(df_all.iloc[:,2:])

    # Fit UMAP and reduce dimensions
    reducer = umap.UMAP(verbose=True)
    reducer.fit(df_all.iloc[:,2:])
    df_projection_test = reducer.transform(df_test.iloc[:,2:])
    df_projection_train = reducer.transform(df_train.iloc[:,2:])
    df_projection_unseen = reducer.transform(df_unseen.iloc[:,2:])

    # Generate and save plot
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    s = ax[0].scatter(df_projection_test[:,0], df_projection_test[:,1], c=df_test.label_idx, s=5)
    ax[0].set_aspect("equal", "datalim")
    ax[0].set_title("UMAP (embedded test data)")
    ax[0].legend(handles=s.legend_elements()[0], labels=class_names)
    s = ax[1].scatter(df_projection_train[:,0], df_projection_train[:,1], c=df_train.label_idx, s=5)
    ax[1].set_aspect("equal", "datalim")
    ax[1].set_title("UMAP (embedded train data)")
    ax[1].legend(handles=s.legend_elements()[0], labels=class_names)
    s = ax[2].scatter(df_projection_unseen[:,0], df_projection_unseen[:,1], c=df_unseen.label_idx, s=5)
    ax[2].set_aspect("equal", "datalim")
    ax[2].set_title("UMAP (embedded unseen classes)")
    ax[2].legend(handles=s.legend_elements()[0], labels=class_names_unseen)
    fig.tight_layout()
    plt.savefig("umap_embeddings.png")
