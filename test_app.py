# test_app.py
from src.extract import ResNetExtractor
from src.index import SimilaritySearch
print("Imports OK")

extractor = ResNetExtractor()
print("Extractor OK")

searcher = SimilaritySearch()
embeddings = searcher.load_data()
searcher.build_index(embeddings)
print("Index OK — app backend is working")