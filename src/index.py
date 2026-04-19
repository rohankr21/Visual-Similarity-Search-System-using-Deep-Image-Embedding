import os
import faiss
import numpy as np
import pandas as pd

class SimilaritySearch:
    def __init__(self, embeddings_dir="embeddings"):
        self.embeddings_path = os.path.join(embeddings_dir, "embeddings.npy")
        self.metadata_path = os.path.join(embeddings_dir, "metadata.csv")
        self.index = None
        self.metadata = None
        
    def load_data(self):
        """Loads embeddings and metadata from disk."""
        print("Loading embeddings and metadata...")
        if not os.path.exists(self.embeddings_path) or not os.path.exists(self.metadata_path):
            raise FileNotFoundError("Embeddings or metadata not found. Run extract.py first.")
            
        embeddings = np.load(self.embeddings_path)
        self.metadata = pd.read_csv(self.metadata_path)
        return embeddings

    def build_index(self, embeddings):
        """Builds a FAISS index for Cosine Similarity."""
        print(f"Building FAISS index for {embeddings.shape[0]} vectors...")
        
        # 1. L2 Normalize the vectors for Cosine Similarity
        faiss.normalize_L2(embeddings)
        
        # 2. Get the dimension size (2048 for ResNet50)
        dimension = embeddings.shape[1]
        
        # 3. Use Inner Product (IP) index. 
        # Normalized Vectors + Inner Product = Cosine Similarity
        self.index = faiss.IndexFlatIP(dimension)
        
        # 4. Add vectors to the index
        self.index.add(embeddings)
        print(f"✅ Index built successfully! Total vectors in index: {self.index.ntotal}")

    def search(self, query_embedding, k=5):
        """Searches the index for the top K most similar vectors."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
            
        # Reshape to (1, 2048) if it's a 1D array
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            
        # L2 normalize the query vector! (Crucial for accurate cosine similarity)
        faiss.normalize_L2(query_embedding)
        
        # Search the index
        # distances = cosine similarity scores (closer to 1.0 is better)
        # indices = the row numbers of the closest matches
        distances, indices = self.index.search(query_embedding, k)
        
        # Fetch the image paths from metadata
        results = []
        for i, idx in enumerate(indices[0]):
            row = self.metadata.iloc[idx]
            img_path = row['img_path']
            category = row['category']
            score = distances[0][i]
            results.append({"img_path": img_path, "category": category, "score": float(score)})
            
        return results

# Quick test script
if __name__ == "__main__":
    searcher = SimilaritySearch()
    
    try:
        # Load and build
        embeddings = searcher.load_data()
        searcher.build_index(embeddings)
        
        # Test search using the FIRST embedding in the dataset as a fake "query"
        print("\nTesting search with the first image in the dataset...")
        test_query = embeddings[0].copy() # Copy to avoid modifying original array
        
        # We search for k=6 because the 1st result will be the image itself (score ~1.0)
        results = searcher.search(test_query, k=6)
        
        print("\nTop 5 Similar Images:")
        for rank, res in enumerate(results[1:6], start=1): # Skip the exact match at index 0
            print(f"Rank {rank}: {res['img_path']} (Score: {res['score']:.4f})")
            
    except Exception as e:
        print(f"❌ Error: {e}")