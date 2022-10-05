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

if __name__ == "__main__":
    #Use custom backbone based on EfficientNet v2
    number_of_classes = len(class_names)
    classifier = BackBone(number_of_classes)
    classifier.to(device)
    #Load custom dataset
    train_dataloader, val_dataloader, test_dataloader = FlowCamDataLoader(class_names, image_size, val, test,  batch_size)

    if not isfile("./checkpoints/best.pth"):
        print("No checkpoint found! Please run training before evaluating model.")
    else:
        print("Loading checkpoint.")
        classifier.load_state_dict(torch.load("./checkpoints/best.pth"))
        
        correct_pred = {classname: 0 for classname in class_names}
        total_pred = {classname: 0 for classname in class_names}
        
        embeddings = []
        with torch.no_grad():
            print("Evaluating model on test data.")
            for data in tqdm(test_dataloader):
                images, labels = data[0].to(device), data[1].to(device)
                outputs, activations_second_last_layer = classifier(images)
                embeddings += [[int(label), 0]  + activation for activation, label in zip(activations_second_last_layer.cpu().detach().tolist(), labels.cpu().detach().tolist())]
                _, predictions = torch.max(outputs, 1)
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[class_names[label]] += 1
                    total_pred[class_names[label]] += 1
        
        #Calculate accuracies
        correct = 0
        total = 0
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * correct_count / total_pred[classname]
            print(f"\"{classname}\" {accuracy:.2f}%")
            total += total_pred[classname]
            correct += correct_count
        total_accuracy = 100 * correct / total
        print(f"Total accuracy: {total_accuracy:.2f}%")
        
        print("Loading dataset with unseen classes.")
        unseen_dataloader = FlowCamDataLoader(class_names_unseen, image_size=image_size, batch_size=batch_size, split=False)
        with torch.no_grad():
            print("Computing embeddings for unseen data.")
            for data in tqdm(unseen_dataloader):
                images, labels = data[0].to(device), data[1].to(device)
                _, activations_second_last_layer = classifier(images) #We don't care about predictions, just embeddings
                embeddings += [[int(label), 1]  + activation for activation, label in zip(activations_second_last_layer.cpu().detach().tolist(), labels.cpu().detach().tolist())]

        #Convert embeddings to pandas dataframe
        embeddings_df = pd.DataFrame(data=embeddings)
        embeddings_df.columns = ["label_idx", "unseen"] + [f"X{i}" for i in range(0, len(embeddings_df.columns) - 2)]
        embeddings_df = embeddings_df.loc[:, embeddings_df.any()] #Remove all-zero columns
        embeddings_df.to_pickle("embeddings.pkl")
        print(f"Saved embeddings of {embeddings_df.shape[0]} images in {embeddings_df.shape[1] - 2} dimensions.")

#TODO
# - Create analyse.py which loads the embeddings, compute average distances (both angular and euclidean) between classes and within classes, also do dimensionality reduction and visualize
# - retrieve_examples.py: for each class show closest and furthest in-class objects and closest objects from other classes
# - transfer_learning.py: load the unseen classes into train, validation and test set, train a classifier after the embedding (linear, SVM and/or kNN classifier)
