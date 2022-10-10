import torch
from dataloader import FlowCamDataLoader

torch.manual_seed(0)

# Settings
#class_names_all = ["chainthin", "darksphere", "Rhabdonellidae", "Odontella", "Codonellopsis", "Neoceratium", "Retaria", "Thalassionematales", "Chaetoceros"]
class_names_all = ["chainthin", "darksphere", "Rhabdonellidae", "Odontella", "Codonellopsis", "Neoceratium", "Retaria", "Thalassionematales", "Chaetoceros", "Neoceratium pentagonum", "Dinophyceae", "Coscinodiscaceae", "Melosiraceae"]
class_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8]
class_idx_unseen = [9, 10, 11, 12]

class_names = [class_names_all[i] for i in class_idx]
class_names_unseen = [class_names_all[i] for i in class_idx_unseen]

batch_size = 64
epochs = 50 
lr = 0.0014
val = 0.05 #Use 5% for validation data 
test = 0.2 #Use 20% for test data
image_size = (128, 128)
device = torch.device('cuda:4') 
