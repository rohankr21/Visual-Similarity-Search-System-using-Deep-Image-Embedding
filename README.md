# 👕 Visual Similarity Search System

A local image retrieval engine that finds visually similar clothing items 
using deep CNN embeddings, approximate nearest-neighbour search, and an 
auxiliary classifier — built entirely with PyTorch and FAISS.

## Demo
![Retrieval Examples](assets/retrieval_examples.png)
![t-SNE Embedding Space](assets/tsne_embeddings.png)

## Results
| Metric | Score |
|---|---|
| Precision@10 | 0.78 |
| mAP | 0.85 |
| Classifier Val Accuracy | 82% |
| Dataset | Myntra Fashion (4,068 real product images, 15 categories) |

## Architecture
Query Image → ResNet50 (no FC) → 2048-d Embedding → FAISS IndexFlatIP
↓
MLP Classifier → Category Filter
↓
rembg → Background Removal
↓
OpenCV K-Means → Color Filter
↓
Top-K Similar Items → Gradio UI

## Features
- **Category Filter** — MLP predicts clothing type, restricts search space
- **Smart Background Removal** — U2-Net via rembg cleans noisy query photos
- **Color Match Filter** — K-Means dominant color extraction with adjustable threshold
- **Interactive UI** — Gradio app with toggles, K slider, live preview

## Setup

```bash
conda create -n visual-search python=3.11
conda activate visual-search
pip install -r requirements.txt
```

Download the [Myntra Fashion dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) 
and place images in `data/raw_myntra/images/`.

```bash
python reorganize_data.py   # organise into category folders
python -m src.extract       # extract ResNet embeddings
python -m src.classify      # train MLP classifier
python app.py               # launch Gradio UI
```

## Tech Stack
PyTorch · FAISS · ResNet50 · Gradio · rembg · OpenCV · scikit-learn

## Project Structure
src/
├── dataset.py      # Custom PyTorch Dataset
├── extract.py      # ResNet50 feature extractor
├── index.py        # FAISS similarity search
├── classify.py     # MLP auxiliary classifier
├── evaluate.py     # Precision@K, Recall@K, mAP
├── visualize.py    # t-SNE + similarity heatmap
└── attributes.py   # Background removal + color extraction
app.py              # Gradio demo