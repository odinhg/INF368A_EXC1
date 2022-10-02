import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from utilities import FlowCamDataLoader
from backbone import BackBone
from tqdm import tqdm

if __name__ == "__main__":
    #class_names = ["Bacillariophyceae", "Rhizosoleniaceae", "Melosiraceae", "Coscinodiscaceae", "Dinophyceae", "Chaetoceros", "nauplii"]
    class_names = ["Bacillariophyceae", "Rhizosoleniaceae", "Melosiraceae", "Coscinodiscaceae"]
    batch_size = 64
    epochs = 50
    lr = 0.0014
    val = 0.05 #Use 10% for validation data 
    test = 0.2 #Use 20% for test data
    image_size = (300, 300)
    number_of_classes = len(class_names)
    
    device = torch.device('cuda:4') 
    print(f"Device: {device}")
    
    #Load custom dataset
    train_dataloader, val_dataloader, test_dataloader = FlowCamDataLoader(class_names, image_size, val, test,  batch_size)

    print(f"Training data: {len(train_dataloader)*batch_size} images.")
    print(f"Validation data: {len(val_dataloader)*batch_size} images.")
    print(f"Test data: {len(test_dataloader)*batch_size} images.")

    #Use custom backbone based on EfficientNet v2
    classifier = BackBone(number_of_classes)
    classifier.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    steps = len(train_dataloader) // 4 #Compute validation and train loss 4 times every epoch
    
    train_history = {"train_loss":[], "train_accuracy":[], "val_loss":[], "val_accuracy":[]}

    for epoch in range(epochs):
        train_losses = []
        train_accuracies = []
        for i, data in enumerate((pbar := tqdm(train_dataloader))):
            images, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = classifier(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            _, predicted = torch.max(outputs.data, 1)
            train_accuracies.append((predicted == labels).sum().item() / labels.size(0))
            if i % steps == steps - 1:
                train_loss = np.mean(train_losses)
                train_losses = []
                train_accuracy = 100.0 * np.mean(train_accuracies)
                correct = 0
                total = 0
                val_losses = []
                with torch.no_grad():
                    for data in val_dataloader:
                        images, labels = data[0].to(device), data[1].to(device)
                        outputs = classifier(images)
                        val_losses.append(loss_function(outputs, labels).item())
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    val_accuracy = 100.0 * correct / total
                    val_loss = np.mean(val_losses)
                if train_history["val_accuracy"] and val_accuracy > np.max(train_history["val_accuracy"]):
                    torch.save(classifier.state_dict(), "./checkpoints/best.pth")
                train_history["train_loss"].append(train_loss)
                train_history["train_accuracy"].append(train_accuracy)
                train_history["val_loss"].append(val_loss)
                train_history["val_accuracy"].append(val_accuracy)
                pbar.set_description(f"Epoch {epoch}/{epochs-1} | Loss: Train={round(train_loss, 3)} Val={round(val_loss, 3)} | Acc.: Train={round(train_accuracy,1)}% Val={round(val_accuracy, 1)}%")
    #Plot losses and accuracies
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    axes[0].plot(train_history["train_loss"], 'b', label="Train")
    axes[0].plot(train_history["val_loss"], 'g', label="Val")
    axes[1].plot(train_history["train_accuracy"], 'b', label="Train")
    axes[1].plot(train_history["val_accuracy"], 'g', label="Val")
    axes[0].title.set_text('Loss')
    axes[1].title.set_text('Accuracy')
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    plt.savefig("training_plot.png")
