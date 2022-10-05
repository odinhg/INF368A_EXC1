import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from configfile import *
from scipy.spatial.distance import cdist

def compute_average_distances(classes):
    avg_euclidean_distances = np.zeros((len(classes), len(classes)))
    avg_angular_distances = np.zeros((len(classes), len(classes)))
    for i in tqdm(range(len(classes))):
        for j in range(len(classes)):
            avg_euclidean_distance = np.mean(cdist(classes[i], classes[j], metric="euclidean"))
            avg_euclidean_distances[i,j] = avg_euclidean_distance
            avg_angular_distance = np.mean(cdist(classes[i], classes[j], metric="cosine"))
            avg_angular_distances[i,j] = avg_angular_distance
    return (avg_euclidean_distances, avg_angular_distances)

if __name__ == "__main__":
    if not (isfile("embeddings_train.pkl") and isfile("embeddings_test.pkl") and isfile("embeddings_test.pkl")):
        exit("Embeddings not found. Please evaluate model first!")
    #Task 4
    df = pd.read_pickle("embeddings_test.pkl")
    classes = [df[df["label_idx"] == i].iloc[:,1:] for i in class_idx]
    avg_euclidean_distances, avg_angular_distances = compute_average_distances(classes)
    print("Average euclidean distances (test dataset):")
    print(avg_euclidean_distances)
    print("Average angular distances (test dataset):")
    print(avg_angular_distances)
    
    #Task 5
    df = pd.read_pickle("embeddings_unseen.pkl")
    classes = [df[df["label_idx"] == i].iloc[:,1:] for i in class_idx_unseen]
    avg_euclidean_distances, avg_angular_distances = compute_average_distances(classes)
    print("Average euclidean distances (unseen classes):")
    print(avg_euclidean_distances)
    print("Average angular distances (unseen classes):")
    print(avg_angular_distances)
