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
from torchsummary import summary

if __name__ == "__main__":
    #Use custom backbone based on EfficientNet v2
    number_of_classes = len(class_names)
    classifier = BackBone(number_of_classes)
    summary(classifier, (3, *image_size))
    classifier.to(device)
    #Load custom dataset
    train_dataloader, val_dataloader, test_dataloader = FlowCamDataLoader(class_names, image_size, val, test,  batch_size)

    if not isfile("./checkpoints/best.pth"):
        print("Training...")
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=lr)
        train_history = train_model(classifier, train_dataloader, val_dataloader, loss_function, optimizer, epochs, device)
        save_train_plot("training_plot.png", train_history)
    else:
        print("Chechpoint found! Please delete checkpoint and run training again.")
