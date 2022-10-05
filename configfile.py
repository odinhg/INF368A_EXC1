import torch
from dataloader import FlowCamDataLoader

torch.manual_seed(0)

# Settings
class_names = ["Bacillariophyceae", "Rhizosoleniaceae", "Melosiraceae", "Coscinodiscaceae"]
class_names_unseen = ["Dinophyceae", "Chaetoceros", "nauplii"]
batch_size = 64
epochs = 50 
lr = 0.0014
val = 0.05 #Use 5% for validation data 
test = 0.2 #Use 20% for test data
image_size = (200, 200)
device = torch.device('cuda:4') 
