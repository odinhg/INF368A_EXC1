import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from os import listdir
from os.path import isfile, join
from PIL import Image

class FlowCamDataSet(Dataset):
    def __init__(self, class_names, image_size = (300, 300)): 
        self.base_dir = "/data/ansatte/kma026/data/FlowCamNet/imgs"
        self.class_names = class_names
        self.images = []
        self.class_sizes = []
        
        for i, class_name in enumerate(self.class_names):
            image_dir = join(self.base_dir, class_name)
            images = [(join(image_dir,f), i) for f in listdir(image_dir) if isfile(join(image_dir, f))]
            self.images += images
            self.class_sizes.append((class_name, len(images)))
        self.transformations = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
        self.number_of_images = len(self.images)
        for class_name, class_size in self.class_sizes:
            print(f"\"{class_name}\": {class_size} images ({round(100*class_size/self.number_of_images, 2)}%).")
        print(f"Total: {self.number_of_images} images.")

    def __getitem__(self, index):
        filename, label = self.images[index]
        image = Image.open(filename).convert("RGB")
        image = self.transformations(image)
        return (image, label)

    def __len__(self):
        return self.number_of_images

def FlowCamDataLoader(class_names, image_size = (300, 300), val = 0.1, test = 0.2, batch_size = 32):
    dataset = FlowCamDataSet(class_names, image_size)
    #Split into train and test data
    val_size = int(val * len(dataset))
    test_size = int(test * len(dataset))
    train_size = len(dataset) - val_size - test_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training data: {len(train_dataloader)*batch_size} images.")
    print(f"Validation data: {len(val_dataloader)*batch_size} images.")
    print(f"Test data: {len(test_dataloader)*batch_size} images.")

    return train_dataloader, val_dataloader, test_dataloader
