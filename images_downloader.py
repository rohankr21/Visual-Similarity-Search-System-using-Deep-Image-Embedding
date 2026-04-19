import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
import shutil

print("Downloading Fashion-MNIST...")
# Download to a temporary folder
dataset = torchvision.datasets.FashionMNIST(root='./temp_fmnist', train=True, download=True)

class_names = ['tshirt', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boot']

# Save first 200 images per class = 2000 images total
counts = {c: 0 for c in class_names}
limit = 200

print("Saving images to data/deepfashion/...")
for img, label in dataset:
    cls = class_names[label]
    
    # Skip if we already have 200 for this category
    if counts[cls] >= limit:
        continue
        
    folder = os.path.join('data', 'deepfashion', cls)
    os.makedirs(folder, exist_ok=True)
    
    # Convert to RGB so it works perfectly with our ResNet50 pipeline
    img_rgb = img.convert('RGB')
    
    # Save the image
    img_path = os.path.join(folder, f"{counts[cls]}.jpg")
    img_rgb.save(img_path)
    
    counts[cls] += 1
    
    # Stop the loop once we hit 200 for every single class
    if all(v >= limit for v in counts.values()):
        break

# Clean up the raw dataset files
print("Cleaning up temporary download files...")
shutil.rmtree('./temp_fmnist', ignore_errors=True)

print("✅ Done! Final counts:")
for cls, count in counts.items():
    print(f"  - {cls}: {count} images")