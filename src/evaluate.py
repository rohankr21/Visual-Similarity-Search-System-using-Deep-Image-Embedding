"""
src/evaluate.py
Evaluation metrics for the Visual Similarity Search System.
Computes Precision@K, Recall@K, and mAP over the embedding database.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from src.index import SimilaritySearch


# ─────────────────────────────────────────────
# 1. CORE METRICS
# ─────────────────────────────────────────────

def precision_at_k(retrieved_labels: list, query_label: str, k: int) -> float:
    """
    Of the top-K retrieved items, what fraction share the query's category?
    Perfect score = 1.0
    """
    top_k = retrieved_labels[:k]
    relevant = sum(1 for label in top_k if label == query_label)
    return relevant / k


def recall_at_k(retrieved_labels: list, query_label: str,
                all_labels: list, k: int) -> float:
    """
    Of ALL relevant items in the database, what fraction appear in top-K?
    Perfect score = 1.0
    """
    top_k = retrieved_labels[:k]
    relevant_retrieved = sum(1 for label in top_k if label == query_label)
    # Total relevant items in DB (excluding the query itself)
    total_relevant = sum(1 for label in all_labels if label == query_label) - 1
    if total_relevant == 0:
        return 0.0
    return relevant_retrieved / total_relevant


def average_precision(retrieved_labels: list, query_label: str) -> float:
    """
    Average Precision (AP) for a single query.
    Rewards systems that rank relevant items higher.
    mAP = mean of AP across all queries.
    """
    hits, precision_sum = 0, 0.0
    for i, label in enumerate(retrieved_labels, start=1):
        if label == query_label:
            hits += 1
            precision_sum += hits / i
    if hits == 0:
        return 0.0
    return precision_sum / hits


# ─────────────────────────────────────────────
# 2. FULL EVALUATION LOOP
# ─────────────────────────────────────────────

def evaluate(embeddings_path: str = "embeddings/embeddings.npy",
             metadata_path:   str = "embeddings/metadata.csv",
             k: int = 10,
             num_queries: int = 200) -> dict:
    """
    Runs Precision@K, Recall@K, and mAP over a random sample of queries.

    Args:
        k:           How many results to retrieve per query
        num_queries: How many random images to use as queries (for speed)

    Returns:
        Dictionary of metric results
    """
    # ── Load data ────────────────────────────
    print("Loading embeddings and metadata...")
    embeddings = np.load(embeddings_path).astype(np.float32)
    metadata   = pd.read_csv(metadata_path)
    all_labels = metadata['category'].tolist()

    # ── Build FAISS index ────────────────────
    searcher = SimilaritySearch()
    searcher.load_data()
    searcher.build_index(embeddings)

    # ── Sample queries ───────────────────────
    np.random.seed(42)
    query_indices = np.random.choice(len(embeddings), size=num_queries, replace=False)

    # ── Collect metrics per query ────────────
    precisions, recalls, aps = [], [], []

    print(f"Evaluating {num_queries} queries at K={k}...")
    for q_idx in query_indices:
        query_vec   = embeddings[q_idx].reshape(1, -1)
        query_label = all_labels[q_idx]

        # Search k+1 because top result is always the query image itself
        results = searcher.search(query_vec, k=k + 1)

        # Skip the first result (exact match, score ~1.0)
        retrieved_labels = [
            all_labels[metadata[metadata['img_path'] == r['img_path']].index[0]]
            for r in results[1:]
            if r['img_path'] in metadata['img_path'].values
        ]

        precisions.append(precision_at_k(retrieved_labels, query_label, k))
        recalls.append(recall_at_k(retrieved_labels, query_label, all_labels, k))
        aps.append(average_precision(retrieved_labels, query_label))

    # ── Aggregate ────────────────────────────
    results_dict = {
        f"Precision@{k}": np.mean(precisions),
        f"Recall@{k}":    np.mean(recalls),
        "mAP":            np.mean(aps),
    }

    # ── Print report ─────────────────────────
    print("\n" + "═" * 40)
    print("   EVALUATION RESULTS")
    print("═" * 40)
    for metric, value in results_dict.items():
        bar_len = int(value * 30)
        bar     = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {metric:<15} {value:.4f}  |{bar}|")
    print("═" * 40)

    return results_dict


# ─────────────────────────────────────────────
# 3. VISUAL RESULTS GRID
# ─────────────────────────────────────────────

def show_retrieval_examples(embeddings_path: str = "embeddings/embeddings.npy",
                             metadata_path:  str = "embeddings/metadata.csv",
                             num_examples:   int = 5,
                             k:              int = 5):
    """
    Plots a grid: each row = 1 query image + its top-K retrieved results.
    Correct matches are highlighted with a green border, wrong with red.
    """
    embeddings = np.load(embeddings_path).astype(np.float32)
    metadata   = pd.read_csv(metadata_path)
    all_labels = metadata['category'].tolist()

    searcher = SimilaritySearch()
    searcher.load_data()
    searcher.build_index(embeddings)

    np.random.seed(7)
    query_indices = np.random.choice(len(embeddings), size=num_examples, replace=False)

    fig = plt.figure(figsize=(3 * (k + 1), 3 * num_examples))
    fig.suptitle("Query → Top-K Retrieved Results\n(Green = correct class, Red = wrong class)",
                 fontsize=13, fontweight='bold', y=1.01)

    for row, q_idx in enumerate(query_indices):
        query_label = all_labels[q_idx]
        query_path  = metadata.iloc[q_idx]['img_path']
        results     = searcher.search(embeddings[q_idx].reshape(1, -1), k=k + 1)[1:]

        # ── Query image (leftmost column) ────
        ax = fig.add_subplot(num_examples, k + 1, row * (k + 1) + 1)
        _show_image(ax, query_path,
                    title=f"QUERY\n{query_label}",
                    border_color="blue", border_width=4)

        # ── Retrieved images ──────────────────
        for col, res in enumerate(results[:k]):
            img_path  = res['img_path']
            try:
                ret_idx   = metadata[metadata['img_path'] == img_path].index[0]
                ret_label = all_labels[ret_idx]
            except IndexError:
                continue

            is_correct   = (ret_label == query_label)
            border_color = "green" if is_correct else "red"
            score_str    = f"{res['score']:.3f}"

            ax = fig.add_subplot(num_examples, k + 1, row * (k + 1) + col + 2)
            _show_image(ax, img_path,
                        title=f"{ret_label}\n{score_str}",
                        border_color=border_color, border_width=3)

    plt.tight_layout()
    plt.savefig("outputs/retrieval_examples.png", dpi=150, bbox_inches='tight')
    print("Saved → outputs/retrieval_examples.png")
    plt.show()


def _show_image(ax, img_path: str, title: str,
                border_color: str, border_width: int):
    """Helper: display one image in a matplotlib axis with a colored border."""
    try:
        img = Image.open(img_path).convert('RGB')
        ax.imshow(img)
    except Exception:
        ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
    ax.set_title(title, fontsize=8, pad=2)
    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(border_width)
        spine.set_visible(True)


# ─────────────────────────────────────────────
# 4. ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    # Run numerical evaluation
    metrics = evaluate(k=10, num_queries=200)

    # Generate visual results grid
    show_retrieval_examples(num_examples=5, k=5)