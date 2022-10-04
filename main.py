import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from utilities import save_train_plot
from dataloader import FlowCamDataLoader
from trainer import train_model
from backbone import BackBone

if __name__ == "__main__":
    #class_names = ["Bacillariophyceae", "Rhizosoleniaceae", "Melosiraceae", "Coscinodiscaceae", "Dinophyceae", "Chaetoceros", "nauplii"]
    class_names = ["Bacillariophyceae", "Rhizosoleniaceae", "Melosiraceae", "Coscinodiscaceae"]
    batch_size = 64
    epochs = 50 
    lr = 0.0014
    val = 0.05 #Use 5% for validation data 
    test = 0.2 #Use 20% for test data
    image_size = (200, 200)
    number_of_classes = len(class_names)
    
    device = torch.device('cuda:4') 
    print(f"Device: {device}")
    torch.manual_seed(0)
    
    #Load custom dataset
    train_dataloader, val_dataloader, test_dataloader = FlowCamDataLoader(class_names, image_size, val, test,  batch_size)

    print(f"Training data: {len(train_dataloader)*batch_size} images.")
    print(f"Validation data: {len(val_dataloader)*batch_size} images.")
    print(f"Test data: {len(test_dataloader)*batch_size} images.")
    
    #Use custom backbone based on EfficientNet v2
    classifier = BackBone(number_of_classes)
    classifier.to(device)

    if not isfile("./checkpoints/best.pth"):
        print("Training...")
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=lr)
        train_history = train_model(classifier, train_dataloader, val_dataloader, loss_function, optimizer, epochs, device)
        save_train_plot("training_plot.png", train_history)
    else:
        print("Loading checkpoint...")
        classifier.load_state_dict(torch.load("./checkpoints/best.pth"))
        
        correct_pred = {classname: 0 for classname in class_names}
        total_pred = {classname: 0 for classname in class_names}
        
        embeddings = []
        with torch.no_grad():
            print("Evaluating model on test data...")
            for data in tqdm(test_dataloader):
                images, labels = data[0].to(device), data[1].to(device)
                outputs, activations_second_last_layer = classifier(images)
                embeddings += activations_second_last_layer.cpu().detach().tolist()
                _, predictions = torch.max(outputs, 1)
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[class_names[label]] += 1
                    total_pred[class_names[label]] += 1

        print("Class accuracies:")
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'\"{classname}\" {accuracy:.1f} %')
        
        # evaluate on unseen classes here
        embeddings_df = pd.DataFrame(data=embeddings)
        embeddings_df.describe()
        # pickle dataframe

#TODO
# - Create a settings file with class_names, class_names_unseen, epochs, learning_rate, etc...
# - Split main into train.py (only train if no checkpoint exists), evaluate.py (load checkpoint, 
#   evaluate on test set and print accuracies, also saves embeddings of test data and unseen classes 
#   to datafram pickle file)
# - Create analyse.py which loads the embeddings, compute average distances (both angular and euclidean) between classes and within classes, also do dimensionality reduction and visualize
# - retrieve_examples.py: for each class show closest and furthest in-class objects and closest objects from other classes
# - transfer_learning.py: load the unseen classes into train, validation and test set, train a classifier after the embedding (linear, SVM and/or kNN classifier)
