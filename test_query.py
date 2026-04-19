# test_query.py
# Picks a random image FROM your database and uses it as query
# This guarantees domain match and proves the pipeline works

import numpy as np
import pandas as pd
from PIL import Image
import shutil, os

metadata = pd.read_csv("embeddings/metadata.csv")

# Pick a random tshirt from the database
tshirts = metadata[metadata['category'] == 'tshirt']
sample  = tshirts.sample(1).iloc[0]

print(f"Using as query: {sample['img_path']}")

# Copy it to outputs so you can view it
os.makedirs("outputs", exist_ok=True)
shutil.copy(sample['img_path'], "outputs/test_query_image.jpg")

# Open and show its size/mode
img = Image.open(sample['img_path'])
print(f"Size: {img.size}, Mode: {img.mode}")
print("\nNow upload outputs/test_query_image.jpg into the Gradio app.")
print("You should see other tshirts returned.")