import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from configfile import *
from utilities import save_train_plot
from dataloader import FlowCamDataLoader
from trainer import train_model
from backbone import BackBone

def save_embeddings(classifier, class_idx, dataloader, filename):
    embeddings = []
    classifier.eval()
    with torch.no_grad():
        for data in tqdm(dataloader):
            images, labels, indicies = data[0].to(device), data[1].to(device), data[2].to(device)
            _, activations_second_last_layer = classifier(images) #We don't care about predictions, just embeddings
            embeddings += [[int(class_idx[label]), int(index)]  + activation for activation, label, index in zip(activations_second_last_layer.cpu().detach().tolist(), labels.cpu().detach().tolist(), indicies.cpu().detach().tolist())]
    df = pd.DataFrame(data=embeddings)
    df.columns = ["label_idx", "image_idx"] + [f"X{i}" for i in range(1, df.shape[1] - 1)]
    df.to_pickle(filename)
    print(f"Dataframe ({df.shape[0]} x {df.shape[1]}) saved to {filename}")

if __name__ == "__main__":
    if not isfile("./checkpoints/best.pth"):
        print("No checkpoint found! Please run training before evaluating model.")
    else:
        number_of_classes = len(class_names)
        classifier = BackBone(number_of_classes)
        classifier.to(device)
        #Load custom dataset
        train_dataloader, val_dataloader, test_dataloader = FlowCamDataLoader(class_names, image_size, val, test,  batch_size)
        unseen_dataloader = FlowCamDataLoader(class_names_unseen, image_size=image_size, batch_size=batch_size, split=False)
        print("Loading checkpoint.")
        classifier.load_state_dict(torch.load("./checkpoints/best.pth"))
        
        print("Embedding data.")
        save_embeddings(classifier, class_idx, train_dataloader, "embeddings_train.pkl")
        save_embeddings(classifier, class_idx, test_dataloader, "embeddings_test.pkl")
        save_embeddings(classifier, class_idx_unseen, unseen_dataloader, "embeddings_unseen.pkl")
