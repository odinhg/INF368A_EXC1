import torch
from dataloader import FlowCamDataLoader

torch.manual_seed(0)

# Settings

#class_names_all =  ["Bacillariophyceae", "Rhizosoleniaceae", "Melosiraceae", "Coscinodiscaceae", "Dinophyceae", "Chaetoceros"]
#class_idx = [0, 1, 2, 3]
#class_idx_unseen = [4, 5]

#class_names_all = ["Rhabdonellidae", "Odontella", "Codonellopsis", "Neoceratium", "Retaria", "Thalassionematales", "Chaetoceros"]
#class_idx = [0, 1, 2, 3]
#class_idx_unseen = [4, 5, 6]

class_names_all = ["chainthin", "darksphere", "Rhabdonellidae", "Odontella", "Codonellopsis", "Neoceratium", "Retaria", "Thalassionematales", "Chaetoceros"]
class_idx = [0, 1, 2, 3, 4, 5]
class_idx_unseen = [6, 7, 8]

class_names = [class_names_all[i] for i in class_idx]
class_names_unseen = [class_names_all[i] for i in class_idx_unseen]

batch_size = 64
epochs = 50 
lr = 0.0014
val = 0.05 #Use 5% for validation data 
test = 0.2 #Use 20% for test data
#image_size = (200, 200)
image_size = (128, 128)
device = torch.device('cuda:4') 
