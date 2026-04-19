import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# PIL = Python Imaging Library
# In Python, the glob module is part of the standard library and 
# is primarily used for finding file and directory pathnames 
# that match a specific pattern

# function for single image processing and feedign to Resnet model
def get_resnet_transforms():
    """Returns the standard ImageNet preprocessing pipeline"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])

class ImagePathDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
        # Recursively find all jpg/jpeg/png files in the dataset folder
        self.image_paths = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            self.image_paths.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))
            
        # Standard ResNet50 preprocessing, call our function here
        self.transform = get_resnet_transforms()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            # Convert to RGB to ensure 3 channels (handles greyscale/RGBA safely)
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img)
            return img_tensor, img_path
        except Exception as e:
            print(f"Skipping {img_path} due to error: {e}")
            # Return empty tensor and path so we can filter it out later
            return torch.zeros((3, 224, 224)), img_path




