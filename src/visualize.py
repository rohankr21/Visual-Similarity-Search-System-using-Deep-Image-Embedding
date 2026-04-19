"""
src/visualize.py
t-SNE visualization of the embedding space.
Plots 2048-d ResNet embeddings reduced to 2D, colored by category.
Visually proves that the embeddings form meaningful clusters.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import os


def plot_tsne(embeddings_path: str = "embeddings/embeddings.npy",
              metadata_path:   str = "embeddings/metadata.csv",
              sample_size:     int = 1500,
              perplexity:      int = 40,
              max_iter:          int = 1000,
              save_path:       str = "outputs/tsne_embeddings.png"):
    """
    Reduces 2048-d embeddings to 2D using t-SNE and plots them.

    Args:
        sample_size: Number of embeddings to visualize (full set is slow)
        perplexity:  t-SNE perplexity — controls cluster tightness (try 30-50)
        max_iter:      Optimization iterations — more = better but slower
    """
    os.makedirs("outputs", exist_ok=True)

    # ── Load data ─────────────────────────────
    print("Loading embeddings and metadata...")
    embeddings = np.load(embeddings_path).astype(np.float32)
    metadata   = pd.read_csv(metadata_path)

    assert len(embeddings) == len(metadata), "Embedding/metadata row mismatch."

    # ── Sample for speed ──────────────────────
    n = min(sample_size, len(embeddings))
    np.random.seed(42)
    indices    = np.random.choice(len(embeddings), size=n, replace=False)
    emb_sample = embeddings[indices]
    labels     = metadata.iloc[indices]['category'].values

    print(f"Running t-SNE on {n} embeddings → 2D...")
    print(f"  perplexity={perplexity}, max_iter={max_iter}")
    print(f"  (This takes 1-3 minutes on CPU — please wait...)")

    # ── Run t-SNE ─────────────────────────────
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=42,
        verbose=1                # prints progress
    )
    emb_2d = tsne.fit_transform(emb_sample)   # shape: (n, 2)

    # ── Assign colors per category ────────────
    unique_classes = sorted(set(labels))
    num_classes    = len(unique_classes)

    # Use a colormap with enough distinct colors
    cmap    = plt.colormaps.get_cmap('tab20').resampled(num_classes)
    cat_to_color = {cls: cmap(i) for i, cls in enumerate(unique_classes)}

    # ── Plot ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')

    # Plot each category as a separate scatter layer
    for cls in unique_classes:
        mask = labels == cls
        ax.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            c=[cat_to_color[cls]],
            label=cls,
            alpha=0.7,
            s=18,
            edgecolors='none'
        )

    # ── Category centroid labels ──────────────
    for cls in unique_classes:
        mask    = labels == cls
        cx      = emb_2d[mask, 0].mean()
        cy      = emb_2d[mask, 1].mean()
        ax.text(
            cx, cy, cls,
            fontsize=9,
            fontweight='bold',
            color='white',
            ha='center', va='center',
            path_effects=[
                pe.Stroke(linewidth=2, foreground='black'),
                pe.Normal()
            ]
        )

    # ── Styling ───────────────────────────────
    ax.set_title(
        "t-SNE Visualization of ResNet50 Embedding Space\n"
        "Each point = one clothing image | Clusters = similar visual features",
        fontsize=13, fontweight='bold', color='white', pad=15
    )
    ax.set_xlabel("t-SNE Dimension 1", color='#aaaaaa', fontsize=10)
    ax.set_ylabel("t-SNE Dimension 2", color='#aaaaaa', fontsize=10)
    ax.tick_params(colors='#555555')

    legend = ax.legend(
        title="Category",
        bbox_to_anchor=(1.01, 1),
        loc='upper left',
        framealpha=0.2,
        facecolor='#1a1a2e',
        labelcolor='white',
        title_fontsize=10,
        fontsize=9
    )
    legend.get_title().set_color('white')

    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"\n✅ Saved → {save_path}")
    plt.show()


def plot_category_distances(embeddings_path: str = "embeddings/embeddings.npy",
                             metadata_path:  str = "embeddings/metadata.csv",
                             save_path:      str = "outputs/category_similarity_matrix.png"):
    """
    Bonus plot: heatmap of average cosine similarity BETWEEN categories.
    Shows which categories are visually closest to each other in embedding space.
    """
    print("\nBuilding category similarity matrix...")
    embeddings = np.load(embeddings_path).astype(np.float32)
    metadata   = pd.read_csv(metadata_path)

    # L2 normalize for cosine similarity
    norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)

    categories = sorted(metadata['category'].unique())
    n_cats     = len(categories)
    sim_matrix = np.zeros((n_cats, n_cats))

    # Compute mean centroid per category, then pairwise cosine similarity
    centroids = []
    for cat in categories:
        idx      = metadata[metadata['category'] == cat].index
        centroid = embeddings[idx].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        centroids.append(centroid)

    centroids = np.array(centroids)
    sim_matrix = centroids @ centroids.T   # pairwise cosine similarity

    # ── Plot heatmap ──────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=0.5, vmax=1.0)
    plt.colorbar(im, ax=ax, label='Cosine Similarity')

    ax.set_xticks(range(n_cats))
    ax.set_yticks(range(n_cats))
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(categories, fontsize=9)

    # Annotate cells with values
    for i in range(n_cats):
        for j in range(n_cats):
            ax.text(j, i, f"{sim_matrix[i, j]:.2f}",
                    ha='center', va='center', fontsize=7,
                    color='black' if sim_matrix[i, j] > 0.7 else 'white')

    ax.set_title(
        "Category-to-Category Cosine Similarity\n"
        "(Centroid of each category's embeddings)",
        fontsize=12, fontweight='bold', pad=15
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved → {save_path}")
    plt.show()


if __name__ == "__main__":
    # Plot 1 — t-SNE cluster visualization
    plot_tsne(sample_size=1500, perplexity=40, max_iter=1000)

    # Plot 2 — Category similarity heatmap (fast, ~5 seconds)
    plot_category_distances()