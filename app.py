import gradio as gr
import numpy as np
import torch
from PIL import Image
import tempfile, os

from src.extract import ResNetExtractor
from src.index import SimilaritySearch
from src.classify import MLPClassifier, predict_category

# ── Load backend ─────────────────────────────
print("Loading models and FAISS index...")
extractor = ResNetExtractor()
searcher  = SimilaritySearch()
embeddings = searcher.load_data()
searcher.build_index(embeddings)

# ── Load MLP classifier ───────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load("models/mlp_classifier.pth", map_location=DEVICE)
classes    = checkpoint['classes']
classifier = MLPClassifier(
    input_dim=checkpoint['input_dim'],
    num_classes=checkpoint['num_classes']
)
classifier.load_state_dict(checkpoint['model_state_dict'])
classifier = classifier.to(DEVICE).eval()
print(f"Classifier loaded. Classes: {classes}")


# ── Search function ───────────────────────────
def perform_search(image, use_filter, k_results):
    if image is None:
        return [], "No image uploaded."

    # Handle numpy array from Gradio
    if not isinstance(image, str):
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        Image.fromarray(image.astype(np.uint8)).save(tmp.name)
        img_path = tmp.name
    else:
        img_path = image

    # 1. Extract embedding
    query_embedding = extractor.extract_features(img_path)

    # 2. Predict category
    predicted_class, confidence = predict_category(
        query_embedding, classifier, classes, DEVICE
    )

    # 3. Search — with or without category filter
    k_fetch = int(k_results) + 1  # +1 to skip exact self-match

    if use_filter:
        # Filter metadata to predicted category only, search subset
        import faiss, pandas as pd
        meta = searcher.metadata
        cat_indices = meta[meta['category'] == predicted_class].index.tolist()

        if len(cat_indices) == 0:
            status = f"Predicted: {predicted_class} ({confidence:.0%}) — No items found in category."
            return [], status

        # Build a temporary sub-index for this category
        cat_embeddings = embeddings[cat_indices].copy().astype('float32')
        faiss.normalize_L2(cat_embeddings)
        sub_index = faiss.IndexFlatIP(cat_embeddings.shape[1])
        sub_index.add(cat_embeddings)

        q_vec = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(q_vec)
        distances, local_indices = sub_index.search(q_vec, min(k_fetch, len(cat_indices)))

        results = []
        for dist, local_idx in zip(distances[0], local_indices[0]):
            global_idx = cat_indices[local_idx]
            results.append({
                "img_path": meta.iloc[global_idx]['img_path'],
                "score": float(dist)
            })
        status = f"Predicted: {predicted_class} ({confidence:.0%}) | Filter ON — searching {len(cat_indices)} {predicted_class} items"

    else:
        results = searcher.search(query_embedding, k=k_fetch)
        status  = f"Predicted: {predicted_class} ({confidence:.0%}) | Filter OFF — searching all {len(embeddings)} items"

    # 4. Format for Gradio gallery
    gallery = []
    for res in results[1:]:   # skip exact self-match
        if res['score'] > 0.0:
            gallery.append((res['img_path'], f"{res['score']:.4f}"))

    return gallery, status


# ── UI ────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(), title="Visual Similarity Search") as demo:
    gr.Markdown("# 👕 Visual Similarity Search System")
    gr.Markdown("Upload a clothing item to find visually similar products from the database.")

    with gr.Row():
        # Left column — controls
        with gr.Column(scale=1):
            input_image = gr.Image(label="Upload Query Image")
            use_filter  = gr.Checkbox(
                label="Category Filter (use MLP classifier to search within predicted category only)",
                value=True
            )
            k_slider = gr.Slider(
                minimum=3, maximum=15, value=6, step=1,
                label="Number of results (K)"
            )
            search_btn = gr.Button("🔍 Find Similar Items", variant="primary")
            status_box = gr.Textbox(label="Search Info", interactive=False)

        # Right column — results
        with gr.Column(scale=2):
            output_gallery = gr.Gallery(
                label="Similar Items Found",
                columns=3,
                object_fit="contain",
                height=500
            )

    search_btn.click(
        fn=perform_search,
        inputs=[input_image, use_filter, k_slider],
        outputs=[output_gallery, status_box]
    )

if __name__ == "__main__":
    demo.launch()