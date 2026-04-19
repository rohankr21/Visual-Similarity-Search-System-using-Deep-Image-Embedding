import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image  
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Import our custom dataset
from src.dataset import ImagePathDataset, get_resnet_transforms

class ResNetExtractor:
    def __init__(self):
        # Auto-detect GPU/MPS/CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load pretrained ResNet50
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Strip the final classification layer (FC) to get the 2048-d avgpool output
        self.model = torch.nn.Sequential(*list(base_model.children())[:-1])
        self.model = self.model.to(self.device).eval() # Set to evaluation mode

        # preprocessing required for single image inference in the app
        self.preprocess = get_resnet_transforms()

    def extract_batch(self, img_tensors):
        """Used for building the database in bulk"""
        img_tensors = img_tensors.to(self.device)
        with torch.no_grad():
            features = self.model(img_tensors)
        # Flatten from (Batch, 2048, 1, 1) to (Batch, 2048)
        return features.squeeze(-1).squeeze(-1).cpu().numpy()

    def extract_features(self, img_path):
        """Used for extracting a single image in the Gradio app."""
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature = self.model(img_tensor)
        return feature.squeeze().cpu().numpy()

def main():
    data_dir = "data/deepfashion"
    embeddings_dir = "embeddings"
    batch_size = 32 # Adjust down to 16 if you get Memory Errors, up to 64/128 if you have a strong GPU
    
    print(f"Scanning directory: {data_dir}...")
    dataset = ImagePathDataset(data_dir)
    
    if len(dataset) == 0:
        print(f"❌ No images found in {data_dir}. Add some images and try again.")
        return
        
    print(f"Found {len(dataset)} images. Building DataLoader...")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    extractor = ResNetExtractor()
    
    # made changes to add category to metadata.csv

    all_embeddings, all_paths, all_categories = [], [], []
    zero_tensor = torch.zeros((3, 224, 224))
    
    print("Extracting features...")
    for img_tensors, img_paths in tqdm(dataloader):
        # Extract embeddings for the batch
        for i, (tensor, path) in enumerate(zip(img_tensors, img_paths)):
            # Skip corrupted images (zero tensor fallback from dataset.py)
            if torch.equal(tensor, zero_tensor):
                continue
            all_paths.append(path)
            # Derive category from immediate parent folder name 
            all_categories.append(os.path.basename(os.path.dirname(path)))

        # Filter out zero tensors before GPU inference
        valid_mask = ~torch.all(img_tensors.view(img_tensors.size(0), -1) == 0, dim=1)
        if valid_mask.sum() == 0:
            continue
        batch_embeddings = extractor.extract_batch(img_tensors[valid_mask])
        all_embeddings.append(batch_embeddings)
        
    # Stack list of batch arrays into one massive (N, 2048) matrix
    final_embeddings = np.vstack(all_embeddings)
    
    # Save the files
    np.save(os.path.join(embeddings_dir, "embeddings.npy"), final_embeddings)
    pd.DataFrame({
        "img_path": all_paths,
        "category": all_categories
    }).to_csv(os.path.join(embeddings_dir, "metadata.csv"), index=False)
    
    print(f"\n✅ Saved {final_embeddings.shape} matrix")
    print(f"Categories found: {pd.Series(all_categories).unique().tolist()}")
    

if __name__ == "__main__":
    main()