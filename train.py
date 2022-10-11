import torch
import numpy as np
import pandas as pd
from os.path import isfile, join
from tqdm import tqdm
from configfile import *
from utilities import save_train_plot
from dataloader import FlowCamDataLoader
from trainer import train_model
from torchsummary import summary

if __name__ == "__main__":
    #Use custom backbone based on EfficientNet v2
    summary(classifier, (3, *image_size), device=device)
    classifier.to(device)
    #Load custom dataset
    train_dataloader, val_dataloader, test_dataloader, _ = FlowCamDataLoader(class_names, image_size, val, test,  batch_size)

    if not isfile("./checkpoints/best.pth"):
        print("Training...")
        train_history = train_model(classifier, train_dataloader, val_dataloader, loss_function, optimizer, epochs, device)
        save_train_plot(join("figs", "training_plot.png"), train_history)
    else:
        print("Chechpoint found! Please delete checkpoint and run training again.")
